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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _validate_max_samples(max_samples: int) -> None:
    if max_samples == 0 or max_samples < -1:
        raise ValueError(f"max_samples must be -1 or a positive integer, got {max_samples}")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _validate_engine_options(engine: str, vllm_language_model_only: bool, max_num_batched_tokens: int) -> None:
    if max_num_batched_tokens <= 0:
        raise ValueError(f"max_num_batched_tokens must be a positive integer, got {max_num_batched_tokens}")
    if vllm_language_model_only and engine != "vllm":
        raise ValueError("vllm_language_model_only requires engine='vllm'")


def _select_sample_prefix(samples: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    _validate_max_samples(max_samples)
    return samples if max_samples == -1 else samples[:max_samples]


def load_swe_dataset(data_path: str, max_samples: int = -1) -> list[dict[str, Any]]:
    _validate_max_samples(max_samples)
    import pyarrow.parquet as pq

    path = os.path.expanduser(data_path)
    logger.info("Loading dataset from: %s", path)
    samples = pq.read_table(path).to_pylist()
    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])
    samples = _select_sample_prefix(samples, max_samples)
    logger.info("Dataset prefix: row indices [0, %d)", len(samples))
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
    output_dir: str | None = None,
    capture_messages: bool = False,
    vllm_language_model_only: bool = False,
    max_num_batched_tokens: int = _DEFAULT_MAX_NUM_BATCHED_TOKENS,
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
    ro.max_num_batched_tokens = max_num_batched_tokens
    ro.n = n
    ro.temperature = temperature
    ro.top_p = top_p
    ro.tensor_model_parallel_size = tensor_parallel_size
    ro.gpu_memory_utilization = float(os.getenv("ROLLOUT_GPU_MEM_UTIL", "0.7"))
    ro.nnodes = nnodes
    ro.n_gpus_per_node = n_gpus_per_node
    ro.calculate_log_probs = True
    ro.enable_sleep_mode = False
    if engine == "vllm":
        ro.engine_kwargs.vllm.language_model_only = vllm_language_model_only
    # Standalone inference is validation-like: it has no trainer global step,
    # so session artifacts are written directly below log_dir instead of step_0.
    ro.val_kwargs.n = n
    ro.val_kwargs.temperature = temperature
    ro.val_kwargs.top_p = top_p

    af = ro.custom.agent_framework
    af.gateway_count = gateway_count
    if output_dir is not None:
        af.log_dir = output_dir
    af.capture_messages = capture_messages
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
# Batch + score capture
# =====================================================================


def _build_prompts(samples: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    from verl.utils import tensordict_utils as tu

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
        non_tensor_dict={"global_steps": None, "validate": True},
    )
    return prompts, uids


def _install_tq_capture(tq_module=None) -> tuple[dict[str, float], dict[str, str]]:
    """Monkeypatch the process-local TransferQueue to capture rm_scores in-memory.

    Runner dispatch is a Ray task, but session finalize/score/TQ-writes happen
    in this driver process, so patching ``tq`` here captures every write.
    """
    if tq_module is None:
        from verl.utils.transferqueue_utils import tq as tq_module

    captured_scores: dict[str, float] = {}
    uid_status: dict[str, str] = {}

    async def _fake_put(*, key, partition_id=None, tag=None, **kwargs):
        if isinstance(tag, dict) and "status" in tag:
            uid_status[str(key)] = str(tag["status"])

    async def _fake_batch_put(*, keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        if fields is None or keys is None or "rm_scores" not in fields:
            return
        rm = fields["rm_scores"]  # nested tensor; rm[i] is trajectory i's response scores
        for i, key in enumerate(keys):
            row = rm[i]
            captured_scores[str(key)] = float(row[-1].item()) if row.numel() else 0.0

    tq_module.async_kv_put = _fake_put
    tq_module.async_kv_batch_put = _fake_batch_put
    return captured_scores, uid_status


def _extract_instance_id(sample: dict[str, Any], sample_index: int) -> str:
    """Return only the stable public identifier needed by the run summary.

    Dataset rows have used both task.metadata and reward.metadata over time.
    Deliberately do not copy either metadata mapping into artifacts: they may
    contain large task payloads or credentials unrelated to reporting.
    """
    extra_info = sample.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        return str(sample_index)
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    if not isinstance(tools_kwargs, dict):
        return str(sample_index)
    for owner in ("task", "reward"):
        owner_config = tools_kwargs.get(owner) or {}
        if not isinstance(owner_config, dict):
            continue
        metadata = owner_config.get("metadata") or {}
        instance_id = metadata.get("instance_id") if isinstance(metadata, dict) else None
        if isinstance(instance_id, str) and instance_id:
            return instance_id
        if isinstance(instance_id, int) and not isinstance(instance_id, bool):
            return str(instance_id)
    return str(sample_index)


def _select_session_scores(
    captured_scores: dict[str, float],
) -> tuple[dict[tuple[str, int], tuple[int, float]], dict[tuple[str, int], int]]:
    """Select one deterministic reward per session from trajectory-keyed TQ writes."""
    selected: dict[tuple[str, int], tuple[int, float]] = {}
    trajectory_counts: dict[tuple[str, int], int] = {}
    for key, score in captured_scores.items():
        # key format: {uid}_{session_index}_{trajectory_index}
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            continue
        uid, raw_session_index, raw_trajectory_index = parts
        try:
            session_index = int(raw_session_index)
            trajectory_index = int(raw_trajectory_index)
        except ValueError:
            continue
        if session_index < 0 or trajectory_index < 0:
            continue
        session_key = (uid, session_index)
        trajectory_counts[session_key] = trajectory_counts.get(session_key, 0) + 1
        previous = selected.get(session_key)
        if previous is None or trajectory_index > previous[0]:
            selected[session_key] = (trajectory_index, float(score))
    return selected, trajectory_counts


def _report(
    samples: list[dict[str, Any]],
    uids: list[str],
    captured_scores: dict[str, float],
    uid_status: dict[str, str] | None = None,
    *,
    planned_sessions: int = 1,
) -> dict[str, Any]:
    uid_status = uid_status or {}
    selected, trajectory_counts = _select_session_scores(captured_scores)
    per_sample_scores: list[float] = []
    sample_results: list[dict[str, Any]] = []
    for sample_index, (sample, uid) in enumerate(zip(samples, uids, strict=True)):
        sessions = []
        for (session_uid, session_index), (trajectory_index, score) in sorted(selected.items()):
            if session_uid != uid:
                continue
            sessions.append(
                {
                    "session_index": session_index,
                    "status": "success",
                    "score": score,
                    "selected_trajectory_index": trajectory_index,
                    "num_captured_trajectories": trajectory_counts[(session_uid, session_index)],
                }
            )
        sample_score = sum(session["score"] for session in sessions) / len(sessions) if sessions else 0.0
        per_sample_scores.append(sample_score)
        sample_results.append(
            {
                "sample_index": sample_index,
                "instance_id": _extract_instance_id(sample, sample_index),
                "status": uid_status.get(uid, "unknown"),
                "num_planned_sessions": planned_sessions,
                "num_captured_sessions": len(sessions),
                "score": sample_score,
                "sessions": sessions,
            }
        )

    resolved = sum(1 for s in per_sample_scores if s > 0)
    mean = sum(per_sample_scores) / len(per_sample_scores) if per_sample_scores else 0.0
    num_planned_sessions = len(samples) * planned_sessions
    num_captured_sessions = sum(result["num_captured_sessions"] for result in sample_results)
    logger.info(
        "Resolved %d / %d samples (%.2f%%), mean score: %.4f",
        resolved,
        len(samples),
        100.0 * resolved / max(len(samples), 1),
        mean,
    )
    return {
        "resolved": resolved,
        "total": len(samples),
        "mean_score": mean,
        "num_planned_sessions": num_planned_sessions,
        "num_captured_sessions": num_captured_sessions,
        "num_uncaptured_sessions": num_planned_sessions - num_captured_sessions,
        "per_sample_scores": per_sample_scores,
        "samples": sample_results,
    }


def _write_summary_atomic(output_dir: str, summary: dict[str, Any]) -> Path:
    artifact_root = Path(output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    destination = artifact_root / "summary.json"
    temporary = artifact_root / f".summary-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _default_output_dir() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(_REPO_ROOT / "outputs" / "claude_code_infer" / f"{timestamp}-{os.getpid()}")


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
    max_turns: int = 100,
    output_dir: str | None = None,
    capture_messages: bool = True,
    vllm_language_model_only: bool = False,
    max_num_batched_tokens: int = _DEFAULT_MAX_NUM_BATCHED_TOKENS,
) -> dict[str, Any]:
    # Keep invalid range checks ahead of Ray import/initialization and model setup.
    _validate_max_samples(max_samples)
    _validate_engine_options(engine, vllm_language_model_only, max_num_batched_tokens)
    output_dir = str(Path(output_dir or _default_output_dir()).expanduser().resolve())
    os.environ["AGENT_MAX_TURNS"] = str(max_turns)

    samples = load_swe_dataset(data_path, max_samples=max_samples)
    if not samples:
        raise ValueError("No samples to process")

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
        output_dir=output_dir,
        capture_messages=capture_messages,
        vllm_language_model_only=vllm_language_model_only,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    import ray

    from uni_agent.framework.entry import build_agent_framework, build_gateway_manager
    from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
    from verl.utils.transferqueue_utils import tq
    from verl.workers.rollout.llm_server import LLMServerManager

    if not ray.is_initialized():
        ray.init()

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
    captured_scores, uid_status = _install_tq_capture(tq)

    logger.info("Starting %d sample(s), %d session(s) each...", len(samples), n)
    try:
        try:
            asyncio.run(framework.generate_sequences(prompts))
        except RuntimeError as exc:
            # Framework raises RuntimeError when every rollout fails; downgrade to a
            # warning so we still report a (zero) resolve rate instead of crashing.
            logger.warning("generate_sequences failed: %s", exc)

        if not captured_scores:
            logger.warning(
                "No trajectory scores captured — all rollouts may have failed (see the "
                "generate_sequences summary above), or the TransferQueue monkeypatch did not "
                "reach the writer; resolve rate will be reported as 0."
            )

        report = _report(
            samples,
            uids,
            captured_scores,
            uid_status,
            planned_sessions=n,
        )
    finally:
        # Always tear down gateway actors, even if generate/report raised, so a
        # failed run does not leak the Ray actor pool.
        asyncio.run(gateway_manager.shutdown())

    summary = {
        "schema_version": 1,
        "status": "completed",
        "artifact_root": output_dir,
        "run": {
            "model_path": os.path.expanduser(model_path),
            "engine": engine,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "nnodes": nnodes,
            "n_gpus_per_node": n_gpus_per_node,
            "tensor_parallel_size": tensor_parallel_size,
            "gateway_count": gateway_count,
            "max_concurrent_sessions": max_concurrent_sessions,
            "tool_image": tool_image,
            "run_timeout": run_timeout,
            "max_turns": max_turns,
            "capture_messages": capture_messages,
            "vllm_language_model_only": vllm_language_model_only,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
        "dataset": {
            "path": os.path.expanduser(data_path),
            "selection": "prefix",
            "requested_max_samples": max_samples,
            "prefix_start": 0,
            "prefix_stop_exclusive": len(samples),
            "num_selected_samples": len(samples),
        },
        **report,
    }
    summary_path = _write_summary_atomic(output_dir, summary)
    logger.info("Wrote completed run summary: %s", summary_path)
    return summary


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
    parser.add_argument(
        "--vllm-language-model-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable all vLLM multimodal inputs and modules (default: disabled for direct CLI calls).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=_positive_int,
        default=_DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help=f"Maximum tokens scheduled in one engine batch (default: {_DEFAULT_MAX_NUM_BATCHED_TOKENS}).",
    )
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-count", type=int, default=1)
    parser.add_argument("--max-concurrent-sessions", type=int, default=8)
    parser.add_argument("--tool-image", type=str, default=_DEFAULT_TOOL_IMAGE)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--capture-messages", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    try:
        _validate_max_samples(args.max_samples)
        _validate_engine_options(
            args.engine,
            args.vllm_language_model_only,
            args.max_num_batched_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir or _default_output_dir()

    logger.info(
        "Requested dataset prefix: %s; output_dir=%s; capture_messages=%s; "
        "vllm_language_model_only=%s; max_num_batched_tokens=%d",
        "all rows" if args.max_samples == -1 else f"[0, {args.max_samples})",
        output_dir,
        bool(args.capture_messages),
        args.vllm_language_model_only,
        args.max_num_batched_tokens,
    )

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
        max_turns=args.max_turns,
        output_dir=output_dir,
        capture_messages=bool(args.capture_messages),
        vllm_language_model_only=args.vllm_language_model_only,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )


if __name__ == "__main__":
    main()
