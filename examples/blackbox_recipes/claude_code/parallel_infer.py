"""Standalone inference runner for the blackbox claude-code recipe.

Spins up vLLM + gateway + a reward worker, runs agent sessions in parallel,
and reports resolve rate. Does NOT start the Megatron trainer.

Reuses the recipe's existing training config
(config/claude_code_megatron_v1.yaml); its megatron/optimizer sections are
inert here since this driver never builds the actor worker group — only the
rollout, agent_framework, model, and reward sections are read.

Usage:
    python examples/blackbox_recipes/claude_code/parallel_infer.py \
        --model-path ~/models/Qwen3.5-9B \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --max-samples 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch

from uni_agent.framework.entry import build_agent_framework, build_gateway_manager
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import tq
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    force=True,
)
logger = logging.getLogger(__name__)

# ── Recipe-specific constants (only these two differ between recipes) ──────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
_CONFIG_NAME = "claude_code_megatron_v1"
_DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest"
_ARTIFACT_SCHEMA_VERSION = 1
_SAFE_REWARD_EXTRA_KEYS = frozenset({"score", "claude_code_exit_code", "resolved", "eval_completed"})


# =====================================================================
# Dataset loading (inlined; keeps the driver self-contained)
# =====================================================================


def _remap_image_to_local(image_name: str) -> str:
    parts = image_name.split("/")
    if len(parts) > 1 and "." in parts[0]:
        basename = parts[-1]
    else:
        basename = image_name
    basename = basename.replace("_1776_", "__")
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return f"{basename}:latest"


def _remap_sample_images(sample: dict[str, Any]) -> dict[str, Any]:
    extra_info = sample.get("extra_info")
    if not extra_info:
        return sample
    tools_kwargs = extra_info.get("tools_kwargs", {})
    env = tools_kwargs.get("env", {})
    image = env.get("image")
    if not image:
        return sample
    local_image = _remap_image_to_local(image)
    if local_image != image:
        logger.debug("Remapping image: %s -> %s", image, local_image)
        env["image"] = local_image
    return sample


def _inject_reward_fields(sample: dict[str, Any]) -> None:
    extra_info = sample.get("extra_info", {})
    tools_kwargs = extra_info.get("tools_kwargs", {})
    reward_config = tools_kwargs.get("reward", {})
    sample.setdefault("data_source", reward_config.get("name", "unknown"))
    sample.setdefault("reward_model", {"ground_truth": {}})


def load_swe_dataset(data_path: str, max_samples: int = -1) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = os.path.expanduser(data_path)
    logger.info("Loading dataset from: %s", path)
    samples = pq.read_table(path).to_pylist()
    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])
    if max_samples > 0:
        samples = samples[:max_samples]
    logger.info("Loaded %d samples", len(samples))
    return samples


# =====================================================================
# Config
# =====================================================================


def _load_config(
    *,
    model_path: str,
    engine: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
    enable_subagents: bool,
    require_subagent: bool,
) -> Any:
    """Compose the recipe's training config and override inference fields.

    The megatron/actor/optimizer sections are left untouched and never read.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        config = compose(config_name=_CONFIG_NAME)

    OmegaConf.set_struct(config, False)

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)

    ro = config.actor_rollout_ref.rollout
    ro.name = engine
    ro.mode = "async"
    ro.prompt_length = prompt_length
    ro.response_length = response_length
    ro.max_model_len = prompt_length + response_length + 1024
    ro.max_num_batched_tokens = ro.max_model_len
    ro.n = n
    ro.temperature = temperature
    ro.top_p = top_p
    ro.tensor_model_parallel_size = tensor_parallel_size
    ro.gpu_memory_utilization = float(os.getenv("ROLLOUT_GPU_MEM_UTIL", "0.7"))
    ro.nnodes = nnodes
    ro.n_gpus_per_node = n_gpus_per_node
    ro.calculate_log_probs = True
    ro.enable_sleep_mode = False

    af = ro.custom.agent_framework
    af.gateway_count = gateway_count
    runner_name = next(iter(af.agent_runners.keys()))
    runner_cfg = af.agent_runners[runner_name]
    runner_cfg.max_concurrent_sessions = max_concurrent_sessions
    if tool_image:
        runner_cfg.runner_kwargs.tool_image = tool_image
    runner_cfg.runner_kwargs.run_timeout = run_timeout
    runner_cfg.runner_kwargs.enable_subagents = enable_subagents
    runner_cfg.runner_kwargs.require_subagent = require_subagent

    config.trainer.nnodes = nnodes
    config.trainer.n_gpus_per_node = n_gpus_per_node

    OmegaConf.set_struct(config, True)
    return config


# =====================================================================
# Batch + trajectory capture
# =====================================================================


def _build_prompts(samples: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    raw_prompts = [sample["prompt"] for sample in samples]
    uids = [str(uuid4()) for _ in samples]
    tools_kwargs_list = [dict((sample.get("extra_info") or {}).get("tools_kwargs", {})) for sample in samples]
    prompts = tu.get_tensordict(
        tensor_dict={
            "raw_prompt": raw_prompts,
            "uid": uids,
            "data_source": [sample["data_source"] for sample in samples],
            "reward_model": [sample["reward_model"] for sample in samples],
            "tools_kwargs": tools_kwargs_list,
        },
        non_tensor_dict={"global_steps": 0},
    )
    return prompts, uids


@dataclass
class InferenceCapture:
    scores: dict[str, float] = field(default_factory=dict)
    uid_status: dict[str, str] = field(default_factory=dict)
    trajectories: list[dict[str, Any]] = field(default_factory=list)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.dim() == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _field_item(fields, key: str, index: int, default=None) -> Any:
    value = tu.get(fields, key, default)
    if value is default:
        return default
    if isinstance(value, torch.Tensor):
        if value.dim() > 0:
            value = value[index]
    elif isinstance(value, (list, tuple)):
        value = value[index]
    return _json_safe(value)


def _parse_trajectory_key(key: str) -> tuple[str, int, int]:
    try:
        uid, session_index, trajectory_index = key.rsplit("_", 2)
        return uid, int(session_index), int(trajectory_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected trajectory key format: {key!r}") from exc


def _sanitize_reward_extra_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: _json_safe(value[key]) for key in _SAFE_REWARD_EXTRA_KEYS if key in value}


def _sanitize_tag(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {"status", "prompt_len", "response_len", "seq_len", "global_steps", "materialization_reason"}
    return {key: _json_safe(value[key]) for key in allowed if key in value}


def _reward_from_rm_scores(row: Any) -> float:
    if isinstance(row, list) and row:
        return float(row[-1])
    if isinstance(row, (int, float)):
        return float(row)
    return 0.0


def _install_tq_capture() -> InferenceCapture:
    """Monkeypatch the process-local TransferQueue to capture inference outputs.

    Runner dispatch is a Ray task, but session finalize/score/TQ-writes happen
    in this driver process, so patching ``tq`` here captures every write.
    """
    capture = InferenceCapture()

    async def _fake_put(*, key, partition_id=None, tag=None, **kwargs):
        if isinstance(tag, dict) and "status" in tag:
            capture.uid_status[str(key)] = str(tag["status"])

    async def _fake_batch_put(*, keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        if fields is None or keys is None:
            return
        tag_list = list(tags or [{} for _ in keys])
        for i, key in enumerate(keys):
            uid, session_index, trajectory_index = _parse_trajectory_key(str(key))
            rm_scores = _field_item(fields, "rm_scores", i, [])
            reward_score = _reward_from_rm_scores(rm_scores)
            capture.scores[str(key)] = reward_score
            capture.trajectories.append(
                {
                    "uid": uid,
                    "session_index": session_index,
                    "trajectory_index": trajectory_index,
                    "is_final_trajectory": i == len(keys) - 1,
                    "data_source": _field_item(fields, "data_source", i),
                    "prompt_ids": _field_item(fields, "prompts", i, []),
                    "response_ids": _field_item(fields, "responses", i, []),
                    "response_mask": _field_item(fields, "response_mask", i, []),
                    "response_logprobs": _field_item(fields, "rollout_log_probs", i, []),
                    "reward_score": reward_score,
                    "num_turns": int(_field_item(fields, "num_turns", i, 0)),
                    "reward_extra_info": _sanitize_reward_extra_info(
                        _field_item(fields, "reward_extra_info", i, {})
                    ),
                    "tags": _sanitize_tag(tag_list[i]),
                }
            )

    tq.async_kv_put = _fake_put
    tq.async_kv_batch_put = _fake_batch_put
    return capture


def _records_for_output(capture: InferenceCapture, uids: list[str]) -> list[dict[str, Any]]:
    uid_to_index = {uid: index for index, uid in enumerate(uids)}
    records = []
    for trajectory in capture.trajectories:
        record = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "sample_index": uid_to_index.get(trajectory["uid"]),
            **trajectory,
        }
        records.append(record)
    return sorted(
        records,
        key=lambda record: (
            record["sample_index"] if record["sample_index"] is not None else len(uids),
            record["session_index"],
            record["trajectory_index"],
        ),
    )


def _report(samples, uids, captured_scores) -> dict[str, Any]:
    uid_to_index = {uid: i for i, uid in enumerate(uids)}
    per_sample_sum = [0.0] * len(samples)
    per_sample_cnt = [0] * len(samples)
    for key, score in captured_scores.items():
        # key format: {uid}_{session_index}_{index}
        uid = key.rsplit("_", 2)[0]
        idx = uid_to_index.get(uid)
        if idx is None:
            continue
        per_sample_sum[idx] += score
        per_sample_cnt[idx] += 1
    per_sample_scores = [
        per_sample_sum[i] / per_sample_cnt[i] if per_sample_cnt[i] else 0.0 for i in range(len(samples))
    ]
    resolved = sum(1 for s in per_sample_scores if s > 0)
    mean = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0
    logger.info(
        "Resolved %d / %d samples (%.2f%%), mean score: %.4f",
        resolved,
        len(samples),
        100.0 * resolved / max(len(samples), 1),
        mean,
    )
    return {"resolved": resolved, "total": len(samples), "mean_score": mean, "per_sample_scores": per_sample_scores}


def _group_trajectory_records(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["uid"]), int(record["session_index"]))].append(record)
    return grouped


def _validate_trajectory_records(
    records: list[dict[str, Any]],
    *,
    require_subagent: bool,
) -> list[str]:
    errors: list[str] = []
    groups = _group_trajectory_records(records)
    multiple_chain_sessions = 0

    for (uid, session_index), session_records in groups.items():
        session_records.sort(key=lambda record: int(record["trajectory_index"]))
        prefix = f"uid={uid} session_index={session_index}"
        indexes = [int(record["trajectory_index"]) for record in session_records]
        if indexes != list(range(len(session_records))):
            errors.append(f"{prefix}: non-contiguous trajectory indexes {indexes}")

        final_records = [record for record in session_records if record["is_final_trajectory"]]
        if len(final_records) != 1:
            errors.append(f"{prefix}: expected exactly one final trajectory, got {len(final_records)}")
        elif final_records[0] is not session_records[-1]:
            errors.append(f"{prefix}: final trajectory is not ordered last")

        rewards = {float(record["reward_score"]) for record in session_records}
        if len(rewards) > 1:
            errors.append(f"{prefix}: reward was not broadcast consistently: {sorted(rewards)}")

        for record in session_records:
            response_length = len(record["response_ids"])
            mask_length = len(record["response_mask"])
            logprob_length = len(record["response_logprobs"])
            if response_length != mask_length or response_length != logprob_length:
                errors.append(
                    f"{prefix} trajectory_index={record['trajectory_index']}: "
                    f"unaligned response fields ids={response_length} mask={mask_length} logprobs={logprob_length}"
                )

        if len(session_records) >= 2:
            multiple_chain_sessions += 1
            if require_subagent and final_records:
                mask_values = set(final_records[0]["response_mask"])
                if not {0, 1}.issubset(mask_values):
                    errors.append(f"{prefix}: final main trajectory does not contain both context and output masks")

    if require_subagent and multiple_chain_sessions == 0:
        errors.append("No successful session produced multiple trajectories; required subagent was not observed")
    return errors


def _build_artifact_summary(
    *,
    report: dict[str, Any],
    records: list[dict[str, Any]],
    capture: InferenceCapture,
    run_config: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    per_session = []
    groups = _group_trajectory_records(records)
    for (uid, session_index), session_records in sorted(groups.items()):
        ordered = sorted(session_records, key=lambda record: int(record["trajectory_index"]))
        per_session.append(
            {
                "uid": uid,
                "session_index": session_index,
                "trajectory_count": len(ordered),
                "trajectory_indexes": [record["trajectory_index"] for record in ordered],
                "final_trajectory_count": sum(bool(record["is_final_trajectory"]) for record in ordered),
                "reward_scores": [record["reward_score"] for record in ordered],
            }
        )

    output_dir = run_config.get("output_dir")
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "run_config": _json_safe(run_config),
        "artifacts": (
            {
                "trajectories": str(Path(output_dir) / "trajectories.jsonl"),
                "summary": str(Path(output_dir) / "summary.json"),
            }
            if output_dir
            else None
        ),
        "result": _json_safe(report),
        "num_trajectories": len(records),
        "num_sessions": len(per_session),
        "multiple_chains_sessions": sum(item["trajectory_count"] >= 2 for item in per_session),
        "uid_status": dict(sorted(capture.uid_status.items())),
        "per_session": per_session,
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
        },
    }


def _write_artifacts(
    output_dir: str | os.PathLike[str],
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    trajectories_path = directory / "trajectories.jsonl"
    summary_path = directory / "summary.json"

    trajectories_tmp = directory / ".trajectories.jsonl.tmp"
    summary_tmp = directory / ".summary.json.tmp"
    jsonl = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    trajectories_tmp.write_text(jsonl, encoding="utf-8")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trajectories_tmp.replace(trajectories_path)
    summary_tmp.replace(summary_path)
    logger.info("Saved %d trajectories to %s", len(records), trajectories_path)
    logger.info("Saved inference summary to %s", summary_path)
    return trajectories_path, summary_path


# =====================================================================
# Runner
# =====================================================================


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    max_samples: int,
    engine: str,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
    enable_subagents: bool,
    require_subagent: bool,
    output_dir: str | None,
) -> dict[str, Any]:
    if require_subagent and not enable_subagents:
        raise ValueError("require_subagent=True requires enable_subagents=True")

    if not ray.is_initialized():
        ray.init()

    config = _load_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        temperature=temperature,
        top_p=top_p,
        n=n,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
        gateway_count=gateway_count,
        max_concurrent_sessions=max_concurrent_sessions,
        tool_image=tool_image,
        run_timeout=run_timeout,
        enable_subagents=enable_subagents,
        require_subagent=require_subagent,
    )

    samples = load_swe_dataset(data_path, max_samples=max_samples)
    if not samples:
        raise ValueError("No samples to process")

    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)
    llm_client = llm_server_manager.get_client()

    gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
    reward_worker = ray.remote(RewardLoopWorker).remote(config, None)
    framework = build_agent_framework(
        config=config,
        gateway_manager=gateway_manager,
        reward_loop_worker_handles=[reward_worker],
    )

    prompts, uids = _build_prompts(samples)
    capture = _install_tq_capture()

    logger.info("Starting %d sample(s), %d session(s) each...", len(samples), n)
    try:
        try:
            asyncio.run(framework.generate_sequences(prompts))
        except RuntimeError as exc:
            # Framework raises RuntimeError when every rollout fails; downgrade to a
            # warning so we still report a (zero) resolve rate instead of crashing.
            logger.warning("generate_sequences failed: %s", exc)

        if not capture.scores:
            logger.warning(
                "No trajectory scores captured — all rollouts may have failed (see the "
                "generate_sequences summary above), or the TransferQueue monkeypatch did not "
                "reach the writer; resolve rate will be reported as 0."
            )

        report = _report(samples, uids, capture.scores)
        records = _records_for_output(capture, uids)
        validation_errors = _validate_trajectory_records(records, require_subagent=require_subagent)
        summary = _build_artifact_summary(
            report=report,
            records=records,
            capture=capture,
            run_config={
                "model_path": os.path.expanduser(model_path),
                "data_path": os.path.expanduser(data_path),
                "engine": engine,
                "n": n,
                "max_samples": max_samples,
                "tool_image": tool_image,
                "enable_subagents": enable_subagents,
                "require_subagent": require_subagent,
                "output_dir": str(Path(output_dir).expanduser().resolve()) if output_dir else None,
            },
            validation_errors=validation_errors,
        )
        if output_dir:
            _write_artifacts(output_dir, records=records, summary=summary)
        if validation_errors:
            raise RuntimeError("Trajectory validation failed: " + "; ".join(validation_errors))
        return summary
    finally:
        # Always tear down gateway actors, even if generate/report raised, so a
        # failed run does not leak the Ray actor pool.
        asyncio.run(gateway_manager.shutdown())


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Blackbox claude-code standalone inference")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3.5-9B")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-count", type=int, default=1)
    parser.add_argument("--max-concurrent-sessions", type=int, default=8)
    parser.add_argument("--tool-image", type=str, default=_DEFAULT_TOOL_IMAGE)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--enable-subagents", action="store_true")
    parser.add_argument(
        "--require-subagent",
        action="store_true",
        help="Append a mandatory subagent instruction and fail validation if no multiple-chain session is captured",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/claude_code_infer")
    args = parser.parse_args()

    if args.require_subagent and not args.enable_subagents:
        parser.error("--require-subagent requires --enable-subagents")

    # Set before ray.init so runner Ray tasks inherit it.
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        gateway_count=args.gateway_count,
        max_concurrent_sessions=args.max_concurrent_sessions,
        tool_image=args.tool_image,
        run_timeout=args.run_timeout,
        enable_subagents=args.enable_subagents,
        require_subagent=args.require_subagent,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
