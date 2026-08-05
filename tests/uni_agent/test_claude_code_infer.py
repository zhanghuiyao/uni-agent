from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from examples.blackbox_recipes.claude_code import parallel_infer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_INFER = _REPO_ROOT / "examples" / "blackbox_recipes" / "claude_code" / "run_infer.sh"


def _fake_python_env(tmp_path: Path, **overrides: str) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args = tmp_path / "python-args.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${CAPTURE_ARGS:?}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ARGS": str(captured_args),
        "REPO_ROOT": str(_REPO_ROOT),
        **overrides,
    }
    return env, captured_args


def _argument_value(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_sample_prefix_selection_and_validation():
    samples = [{"row": i} for i in range(4)]

    assert parallel_infer._select_sample_prefix(samples, 2) == samples[:2]
    assert parallel_infer._select_sample_prefix(samples, 10) == samples
    assert parallel_infer._select_sample_prefix(samples, -1) == samples
    for invalid in (0, -2, -10):
        with pytest.raises(ValueError, match="-1 or a positive integer"):
            parallel_infer._select_sample_prefix(samples, invalid)


def test_invalid_max_samples_stops_before_dataset_or_ray(monkeypatch):
    def _unexpected_dataset_load(*args, **kwargs):
        raise AssertionError("dataset/runtime setup must not run")

    monkeypatch.setattr(parallel_infer, "load_swe_dataset", _unexpected_dataset_load)

    with pytest.raises(ValueError, match="got 0"):
        parallel_infer.run_inference(
            model_path="model",
            data_path="dataset.parquet",
            prompt_length=16,
            response_length=32,
            temperature=1.0,
            top_p=1.0,
            n=1,
            max_samples=0,
            engine="vllm",
            nnodes=1,
            n_gpus_per_node=1,
            tensor_parallel_size=1,
            gateway_count=1,
            max_concurrent_sessions=1,
            tool_image=None,
            run_timeout=60,
        )


def test_load_config_wires_artifact_and_message_capture(tmp_path):
    config = parallel_infer._load_config(
        model_path="~/model",
        engine="vllm",
        prompt_length=16,
        response_length=32,
        temperature=0.7,
        top_p=0.9,
        n=3,
        nnodes=1,
        n_gpus_per_node=2,
        tensor_parallel_size=2,
        gateway_count=1,
        max_concurrent_sessions=4,
        tool_image=None,
        run_timeout=60,
        output_dir=str(tmp_path),
        capture_messages=True,
        vllm_language_model_only=True,
        max_num_batched_tokens=16384,
    )

    rollout = config.actor_rollout_ref.rollout
    framework = rollout.custom.agent_framework
    assert framework.log_dir == str(tmp_path)
    assert framework.capture_messages is True
    assert rollout.val_kwargs.n == 3
    assert rollout.val_kwargs.temperature == 0.7
    assert rollout.val_kwargs.top_p == 0.9
    assert rollout.max_num_batched_tokens == 16384
    assert rollout.engine_kwargs.vllm.language_model_only is True


def test_invalid_vllm_engine_options_stop_before_dataset_or_ray(monkeypatch):
    def _unexpected_dataset_load(*args, **kwargs):
        raise AssertionError("dataset/runtime setup must not run")

    monkeypatch.setattr(parallel_infer, "load_swe_dataset", _unexpected_dataset_load)

    with pytest.raises(ValueError, match="requires engine='vllm'"):
        parallel_infer.run_inference(
            model_path="model",
            data_path="dataset.parquet",
            prompt_length=16,
            response_length=32,
            temperature=1.0,
            top_p=1.0,
            n=1,
            max_samples=1,
            engine="sglang",
            nnodes=1,
            n_gpus_per_node=1,
            tensor_parallel_size=1,
            gateway_count=1,
            max_concurrent_sessions=1,
            tool_image=None,
            run_timeout=60,
            vllm_language_model_only=True,
        )


def test_report_counts_each_session_once_and_omits_private_metadata():
    samples = [
        {
            "extra_info": {
                "tools_kwargs": {
                    "task": {"metadata": {"instance_id": "task-shape", "secret": "secret-value"}},
                    "reward": {"metadata": {"instance_id": "older-id", "token": "reward-token"}},
                }
            }
        },
        {
            "extra_info": {
                "tools_kwargs": {"reward": {"metadata": {"instance_id": "reward-shape", "secret": "legacy-secret"}}}
            }
        },
        {"extra_info": {"tools_kwargs": {"task": {"metadata": {"secret": "fallback-secret"}}}}},
    ]
    uids = ["uid_with_underscores", "legacy", "fallback"]
    captured_scores = {
        "uid_with_underscores_0_0": 0.1,
        "uid_with_underscores_0_2": 0.8,
        "uid_with_underscores_1_0": 0.4,
        "legacy_0_0": 0.0,
        "malformed": 99.0,
    }

    report = parallel_infer._report(
        samples,
        uids,
        captured_scores,
        {"uid_with_underscores": "finished", "legacy": "finished", "fallback": "failure"},
        planned_sessions=2,
    )

    assert report["per_sample_scores"] == pytest.approx([0.6, 0.0, 0.0])
    assert report["resolved"] == 1
    assert report["num_planned_sessions"] == 6
    assert report["num_captured_sessions"] == 3
    first = report["samples"][0]
    assert first["instance_id"] == "task-shape"
    assert first["num_captured_sessions"] == 2
    assert first["sessions"][0] == {
        "session_index": 0,
        "status": "success",
        "score": 0.8,
        "selected_trajectory_index": 2,
        "num_captured_trajectories": 2,
    }
    assert report["samples"][1]["instance_id"] == "reward-shape"
    assert report["samples"][2]["instance_id"] == "2"
    assert parallel_infer._extract_instance_id({"extra_info": "malformed"}, 7) == "7"
    serialized = json.dumps(report)
    for private_value in ("secret-value", "reward-token", "legacy-secret", "fallback-secret"):
        assert private_value not in serialized


def test_summary_is_atomically_replaced_without_jsonl(tmp_path):
    destination = parallel_infer._write_summary_atomic(str(tmp_path), {"status": "first"})
    parallel_infer._write_summary_atomic(str(tmp_path), {"status": "completed", "samples": []})

    assert destination == tmp_path / "summary.json"
    assert json.loads(destination.read_text()) == {"status": "completed", "samples": []}
    assert not list(tmp_path.glob(".summary-*.tmp"))
    assert not list(tmp_path.glob("*.jsonl"))


def test_run_infer_shell_passes_explicit_artifact_options(tmp_path):
    output_dir = tmp_path / "artifacts"
    env, captured_args = _fake_python_env(
        tmp_path,
        MAX_SAMPLES="10",
        OUTPUT_DIR=str(output_dir),
        SAVE_TRAJECTORY_MESSAGES="0",
        VLLM_LANGUAGE_MODEL_ONLY="0",
        MAX_NUM_BATCHED_TOKENS="16384",
    )

    completed = subprocess.run(
        ["bash", str(_RUN_INFER)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = captured_args.read_text().splitlines()
    assert _argument_value(arguments, "--max-samples") == "10"
    assert _argument_value(arguments, "--output-dir") == str(output_dir)
    assert _argument_value(arguments, "--capture-messages") == "0"
    assert "--no-vllm-language-model-only" in arguments
    assert _argument_value(arguments, "--max-num-batched-tokens") == "16384"
    assert "Sample range: row indices [0, 10)" in completed.stdout
    assert f"Output:      {output_dir}" in completed.stdout


def test_run_infer_shell_creates_timestamped_default_arguments(tmp_path):
    env, captured_args = _fake_python_env(
        tmp_path,
        MAX_SAMPLES="-1",
        SAVE_TRAJECTORY_MESSAGES="1",
    )
    env.pop("OUTPUT_DIR", None)

    completed = subprocess.run(
        ["bash", str(_RUN_INFER)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = captured_args.read_text().splitlines()
    output_dir = Path(_argument_value(arguments, "--output-dir"))
    assert output_dir.parent == _REPO_ROOT / "outputs" / "claude_code_infer"
    assert re.fullmatch(r"\d{8}T\d{6}Z-\d+", output_dir.name)
    assert _argument_value(arguments, "--capture-messages") == "1"
    assert "--vllm-language-model-only" in arguments
    assert _argument_value(arguments, "--max-num-batched-tokens") == "8192"
    assert "Sample range: all dataset rows" in completed.stdout


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_SAMPLES", "0"),
        ("MAX_SAMPLES", "-2"),
        ("SAVE_TRAJECTORY_MESSAGES", "yes"),
        ("VLLM_LANGUAGE_MODEL_ONLY", "yes"),
        ("MAX_NUM_BATCHED_TOKENS", "0"),
        ("MAX_NUM_BATCHED_TOKENS", "invalid"),
    ],
)
def test_run_infer_shell_rejects_invalid_ranges_and_flags_before_python(tmp_path, name, value):
    env, captured_args = _fake_python_env(tmp_path, **{name: value})
    if name != "MAX_SAMPLES":
        env["MAX_SAMPLES"] = "1"

    completed = subprocess.run(
        ["bash", str(_RUN_INFER)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert not captured_args.exists()
    assert name in completed.stderr
