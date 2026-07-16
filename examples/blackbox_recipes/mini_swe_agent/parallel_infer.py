"""Standalone inference runner for the blackbox mini-swe-agent recipe.

Spins up vLLM + gateway + a reward worker, runs agent sessions in parallel,
and reports resolve rate. Does NOT start the Megatron trainer.

Reuses the recipe's existing training config
(config/swe_agent_blackbox_megatron_v1.yaml); its megatron/optimizer sections
are inert here since this driver never builds the actor worker group — only
the rollout, agent_framework, model, and reward sections are read.

Usage:
    python examples/blackbox_recipes/mini_swe_agent/parallel_infer.py \
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# ── Recipe-specific constants ─────────────────────────────────────────────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
_CONFIG_NAME = "swe_agent_blackbox_megatron_v1"
_DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest"
_ARTIFACT_SCHEMA_VERSION = 1
_SEQUENCE_TYPES = (list, tuple)
_JSON_PRIMITIVE_TYPES = (str, int, float, bool)
_NUMBER_TYPES = (int, float)
_SAFE_TAG_KEYS = frozenset(
    {
        "status",
        "global_steps",
        "min_global_steps",
        "max_global_steps",
        "prompt_len",
        "response_len",
        "seq_len",
        "finish_reason",
    }
)


# =====================================================================
# Dataset loading (inlined; keeps the driver self-contained)
# =====================================================================


def _parse_sample_split(value: str) -> tuple[int, int] | None:
    value = value.strip()
    if not value:
        return None

    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("--sample-split must use START:END with non-negative integers")

    start, end = (int(part) for part in parts)
    if start >= end:
        raise argparse.ArgumentTypeError("--sample-split requires START < END")
    return start, end


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


def load_swe_dataset(
    data_path: str,
    max_samples: int = -1,
    sample_split: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    import pyarrow.parquet as pq

    path = os.path.expanduser(data_path)
    logger.info("Loading dataset from: %s", path)
    table = pq.read_table(path)
    dataset_size = len(table)
    if sample_split is not None:
        start, end = sample_split
        if start < 0 or start >= end:
            raise ValueError(f"Invalid sample split {start}:{end}; expected 0 <= START < END")
        if end > dataset_size:
            raise ValueError(f"Sample split {start}:{end} exceeds dataset size {dataset_size}")
        table = table.slice(start, end - start)
        dataset_indices = list(range(start, end))
    elif max_samples > 0:
        sample_count = min(max_samples, dataset_size)
        table = table.slice(0, sample_count)
        dataset_indices = list(range(sample_count))
    else:
        dataset_indices = list(range(dataset_size))

    samples = table.to_pylist()
    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])
    logger.info("Loaded %d samples", len(samples))
    return samples, dataset_indices


# =====================================================================
# Config
# =====================================================================


def _load_config(
    *,
    model_path: str,
    engine: str,
    prompt_length: int,
    response_length: int,
    max_num_batched_tokens: int | None,
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
    ro.max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else ro.max_model_len
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
    uid_status: dict[str, str] = field(default_factory=dict)
    sessions: dict[tuple[str, int], SessionArtifact] = field(default_factory=dict)
    write_error: BaseException | None = None

    def raise_if_write_failed(self) -> None:
        if self.write_error is not None:
            raise self.write_error


@dataclass(frozen=True)
class SessionArtifact:
    dataset_index: int
    uid: str
    session_index: int
    trajectory_count: int
    trajectory_indexes: tuple[int, ...]
    final_trajectory_index: int
    score: float
    resolved: bool
    path: Path

    def metadata_record(self) -> dict[str, Any]:
        return {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "record_type": "session_metadata",
            "dataset_index": self.dataset_index,
            "uid": self.uid,
            "session_index": self.session_index,
            "trajectory_count": self.trajectory_count,
            "final_trajectory_index": self.final_trajectory_index,
            "score": self.score,
            "resolved": self.resolved,
        }


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
    if isinstance(value, _SEQUENCE_TYPES):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, _JSON_PRIMITIVE_TYPES):
        return value
    return str(value)


def _field_item(fields, key: str, index: int, default=None) -> Any:
    value = tu.get(fields, key, default)
    if value is default:
        return default
    if isinstance(value, torch.Tensor):
        if value.dim() > 0:
            value = value[index]
    elif isinstance(value, _SEQUENCE_TYPES):
        value = value[index]
    return _json_safe(value)


def _parse_trajectory_key(key: str) -> tuple[str, int, int]:
    try:
        uid, session_index, trajectory_index = key.rsplit("_", 2)
        return uid, int(session_index), int(trajectory_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected trajectory key format: {key!r}") from exc


def _sanitize_tag(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: _json_safe(value[key]) for key in _SAFE_TAG_KEYS if key in value}


def _reward_from_rm_scores(row: Any) -> float:
    if isinstance(row, list) and row:
        return float(row[-1])
    if isinstance(row, _NUMBER_TYPES):
        return float(row)
    return 0.0


def _write_jsonl_record(handle, record: dict[str, Any]) -> None:
    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
    handle.write("\n")


def _write_session_artifact_atomic(
    *,
    run_dir: Path,
    uid_to_dataset_index: dict[str, int],
    keys: list[str],
    fields: Any,
    tags: list[Any],
) -> SessionArtifact:
    parsed_keys = [_parse_trajectory_key(key) for key in keys]
    uid, session_index, _ = parsed_keys[0]
    if any(
        parsed_uid != uid or parsed_session_index != session_index
        for parsed_uid, parsed_session_index, _ in parsed_keys
    ):
        raise ValueError("A TransferQueue batch must contain exactly one session")

    trajectory_indexes = tuple(trajectory_index for _, _, trajectory_index in parsed_keys)
    if trajectory_indexes != tuple(range(len(trajectory_indexes))):
        raise ValueError(f"Unexpected trajectory indexes for uid={uid} session={session_index}: {trajectory_indexes}")
    if uid not in uid_to_dataset_index:
        raise ValueError(f"Unknown trajectory uid: {uid}")

    dataset_index = uid_to_dataset_index[uid]
    final_trajectory_index = trajectory_indexes[-1]
    final_score = _reward_from_rm_scores(_field_item(fields, "rm_scores", len(keys) - 1, []))
    path = run_dir / _trajectory_filename(dataset_index, session_index)
    artifact = SessionArtifact(
        dataset_index=dataset_index,
        uid=uid,
        session_index=session_index,
        trajectory_count=len(keys),
        trajectory_indexes=trajectory_indexes,
        final_trajectory_index=final_trajectory_index,
        score=final_score,
        resolved=final_score > 0.0,
        path=path,
    )
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            _write_jsonl_record(handle, artifact.metadata_record())
            for index in range(len(keys)):
                parsed_uid, parsed_session_index, trajectory_index = parsed_keys[index]
                _write_jsonl_record(
                    handle,
                    {
                        "schema_version": _ARTIFACT_SCHEMA_VERSION,
                        "record_type": "trajectory",
                        "dataset_index": dataset_index,
                        "uid": parsed_uid,
                        "session_index": parsed_session_index,
                        "trajectory_index": trajectory_index,
                        "is_final_trajectory": index == len(keys) - 1,
                        "data_source": _field_item(fields, "data_source", index),
                        "prompt_ids": _field_item(fields, "prompts", index, []),
                        "response_ids": _field_item(fields, "responses", index, []),
                        "response_mask": _field_item(fields, "response_mask", index, []),
                        "response_logprobs": _field_item(fields, "rollout_log_probs", index, []),
                        "num_turns": int(_field_item(fields, "num_turns", index, 0)),
                        "tags": _sanitize_tag(tags[index]),
                    },
                )
        temporary_path.replace(path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove incomplete session artifact %s", temporary_path, exc_info=True)
        raise
    return artifact


def _install_tq_capture(
    *,
    run_dir: Path,
    uids: list[str],
    dataset_indices: list[int],
) -> InferenceCapture:
    """Monkeypatch the process-local TransferQueue to capture inference outputs.

    Runner dispatch is a Ray task, but session finalize/score/TQ-writes happen
    in this driver process, so patching ``tq`` here captures every write.
    """
    if len(uids) != len(dataset_indices):
        raise ValueError("UID and dataset index counts must match")
    uid_to_dataset_index = dict(zip(uids, dataset_indices, strict=True))
    capture = InferenceCapture()

    async def _fake_put(*, key, partition_id=None, tag=None, **kwargs):
        if isinstance(tag, dict) and "status" in tag:
            capture.uid_status[str(key)] = str(tag["status"])

    async def _fake_batch_put(*, keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        if fields is None or keys is None:
            return
        key_list = [str(key) for key in keys]
        if not key_list:
            return
        tag_list = list(tags or [{} for _ in key_list])
        try:
            artifact = await asyncio.to_thread(
                _write_session_artifact_atomic,
                run_dir=run_dir,
                uid_to_dataset_index=uid_to_dataset_index,
                keys=key_list,
                fields=fields,
                tags=tag_list,
            )
        except BaseException as exc:
            if capture.write_error is None:
                capture.write_error = exc
            raise
        capture.sessions[(artifact.uid, artifact.session_index)] = artifact

    tq.async_kv_put = _fake_put
    tq.async_kv_batch_put = _fake_batch_put
    return capture


def _report(samples, uids, sessions: dict[tuple[str, int], SessionArtifact]) -> dict[str, Any]:
    uid_to_index = {uid: i for i, uid in enumerate(uids)}
    per_sample_sum = [0.0] * len(samples)
    per_sample_cnt = [0] * len(samples)
    for session in sessions.values():
        idx = uid_to_index.get(session.uid)
        if idx is None:
            continue
        per_sample_sum[idx] += session.score
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


def _trajectory_filename(dataset_index: int, session_index: int) -> str:
    return f"trajectory_session_{session_index}_task_{dataset_index}.jsonl"


def _build_artifact_summary(
    *,
    report: dict[str, Any],
    capture: InferenceCapture,
    planned_sessions: int,
    run_config: dict[str, Any],
    created_at: datetime,
    run_dir: Path,
) -> dict[str, Any]:
    ordered_sessions = sorted(
        capture.sessions.values(),
        key=lambda session: (session.dataset_index, session.session_index, session.uid),
    )
    per_session = []
    for session in ordered_sessions:
        per_session.append(
            {
                "dataset_index": session.dataset_index,
                "uid": session.uid,
                "session_index": session.session_index,
                "trajectory_count": session.trajectory_count,
                "trajectory_indexes": list(session.trajectory_indexes),
                "final_trajectory_index": session.final_trajectory_index,
                "score": session.score,
                "resolved": session.resolved,
                "path": str(session.path),
            }
        )

    successful_sessions = len(per_session)
    if successful_sessions > planned_sessions:
        raise ValueError(
            f"Captured {successful_sessions} successful sessions, exceeding planned session count {planned_sessions}"
        )

    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "run_config": _json_safe(run_config),
        "artifacts": {
            "run_dir": str(run_dir),
            "summary": str(run_dir / "summary.json"),
            "trajectories": per_session,
        },
        "result": _json_safe(report),
        "num_trajectories": sum(session.trajectory_count for session in ordered_sessions),
        "num_planned_sessions": planned_sessions,
        "num_successful_sessions": successful_sessions,
        "num_failed_sessions": planned_sessions - successful_sessions,
        "multiple_chains_sessions": sum(item["trajectory_count"] >= 2 for item in per_session),
        "uid_status": dict(sorted(capture.uid_status.items())),
        "per_session": per_session,
    }


def _create_run_directory(output_dir: str | os.PathLike[str], created_at: datetime) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / created_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(exist_ok=False)
    return output_root, run_dir


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove incomplete artifact %s", temporary_path, exc_info=True)
        raise


def _write_summary_atomic(run_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = run_dir / "summary.json"
    _write_text_atomic(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    logger.info(
        "Saved summary for %d trajectories across %d session files to %s",
        summary["num_trajectories"],
        summary["num_successful_sessions"],
        run_dir,
    )
    return summary_path


# =====================================================================
# Runner
# =====================================================================


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int,
    response_length: int,
    max_num_batched_tokens: int | None,
    temperature: float,
    top_p: float,
    n: int,
    max_samples: int,
    sample_split: tuple[int, int] | None,
    engine: str,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
    output_dir: str,
) -> dict[str, Any]:
    if not ray.is_initialized():
        ray.init()

    config = _load_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        max_num_batched_tokens=max_num_batched_tokens,
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
    )

    samples, dataset_indices = load_swe_dataset(
        data_path,
        max_samples=max_samples,
        sample_split=sample_split,
    )
    if not samples:
        raise ValueError("No samples to process")

    prompts, uids = _build_prompts(samples)
    created_at = datetime.now(timezone.utc)
    output_root, run_dir = _create_run_directory(output_dir, created_at)
    capture = _install_tq_capture(
        run_dir=run_dir,
        uids=uids,
        dataset_indices=dataset_indices,
    )

    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)
    llm_client = llm_server_manager.get_client()

    gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
    primary_error: BaseException | None = None
    try:
        reward_worker = ray.remote(RewardLoopWorker).remote(config, None)
        framework = build_agent_framework(
            config=config,
            gateway_manager=gateway_manager,
            reward_loop_worker_handles=[reward_worker],
        )

        logger.info("Starting %d sample(s), %d session(s) each...", len(samples), n)
        try:
            asyncio.run(framework.generate_sequences(prompts))
        except RuntimeError as exc:
            logger.warning("generate_sequences failed: %s", exc)

        capture.raise_if_write_failed()
        if not capture.sessions:
            logger.warning(
                "No session results captured — all rollouts may have failed (see the "
                "generate_sequences summary above), or the TransferQueue monkeypatch did not "
                "reach the writer; resolve rate will be reported as 0."
            )

        result = _report(samples, uids, capture.sessions)
        summary = _build_artifact_summary(
            report=result,
            capture=capture,
            planned_sessions=len(samples) * n,
            run_config={
                "model_path": os.path.expanduser(model_path),
                "data_path": os.path.expanduser(data_path),
                "sample_split": f"{sample_split[0]}:{sample_split[1]}" if sample_split else None,
                "max_samples": max_samples,
                "selected_sample_count": len(samples),
                "prompt_length": prompt_length,
                "response_length": response_length,
                "max_num_batched_tokens": (
                    max_num_batched_tokens
                    if max_num_batched_tokens is not None
                    else prompt_length + response_length + 1024
                ),
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "engine": engine,
                "tensor_parallel_size": tensor_parallel_size,
                "nnodes": nnodes,
                "n_gpus_per_node": n_gpus_per_node,
                "gateway_count": gateway_count,
                "max_concurrent_sessions": max_concurrent_sessions,
                "tool_image": tool_image,
                "run_timeout": run_timeout,
                "agent_max_turns": int(os.environ.get("AGENT_MAX_TURNS", "100")),
                "output_root": str(output_root),
            },
            created_at=created_at,
            run_dir=run_dir,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            asyncio.run(gateway_manager.shutdown())
        except BaseException:
            if primary_error is None:
                raise
            logger.exception("Gateway shutdown failed while handling an earlier inference error")

    _write_summary_atomic(run_dir, summary)
    return result


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Blackbox mini-swe-agent standalone inference")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3.5-9B")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--sample-split",
        type=_parse_sample_split,
        default=None,
        metavar="START:END",
        help="Zero-based half-open dataset row range; overrides --max-samples",
    )
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
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
    parser.add_argument("--output-dir", type=str, default="outputs/mini_swe_agent_infer")
    args = parser.parse_args()

    # Set before ray.init so runner Ray tasks inherit it.
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        max_num_batched_tokens=args.max_num_batched_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        sample_split=args.sample_split,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        gateway_count=args.gateway_count,
        max_concurrent_sessions=args.max_concurrent_sessions,
        tool_image=args.tool_image,
        run_timeout=args.run_timeout,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
