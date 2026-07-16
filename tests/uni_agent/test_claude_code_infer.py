import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from examples.blackbox_recipes.claude_code.claude_code_runner import build_claude_command, build_claude_task
from examples.blackbox_recipes.claude_code.parallel_infer import (
    InferenceCapture,
    _build_artifact_summary,
    _install_tq_capture,
    _load_config,
    _records_for_output,
    _validate_trajectory_records,
    _write_artifacts,
)
from examples.blackbox_recipes.claude_code.reward import compute_score
from uni_agent.framework.framework import _list_of_tq_fields_to_tensordict
from verl.utils.transferqueue_utils import tq


def test_inference_config_passes_subagent_flags_to_runner():
    config = _load_config(
        model_path="/tmp/model",
        engine="vllm",
        prompt_length=128,
        response_length=256,
        temperature=1.0,
        top_p=1.0,
        n=1,
        nnodes=1,
        n_gpus_per_node=1,
        tensor_parallel_size=1,
        gateway_count=1,
        max_concurrent_sessions=1,
        tool_image="claude-code-tool:latest",
        run_timeout=60,
        enable_subagents=True,
        require_subagent=True,
        save_trajectory_messages=True,
    )
    runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code

    assert runner.runner_kwargs.tool_image == "claude-code-tool:latest"
    assert runner.runner_kwargs.enable_subagents is True
    assert runner.runner_kwargs.require_subagent is True
    assert config.actor_rollout_ref.rollout.disable_log_stats is False
    assert config.actor_rollout_ref.rollout.custom.agent_framework.capture_messages is True


def test_inference_config_defaults_message_capture_off():
    config = _load_config(
        model_path="Qwen/Qwen3.5-9B",
        prompt_length=1024,
        response_length=2048,
        temperature=0.7,
        top_p=0.9,
        n=1,
        engine="vllm",
        nnodes=1,
        n_gpus_per_node=1,
        tensor_parallel_size=1,
        gateway_count=1,
        max_concurrent_sessions=1,
        tool_image="claude-code-tool:latest",
        run_timeout=60,
        enable_subagents=False,
        require_subagent=False,
    )

    assert config.actor_rollout_ref.rollout.custom.agent_framework.capture_messages is False


@pytest.mark.parametrize(("save_messages", "expected_flag"), [("1", True), ("0", False)])
def test_run_infer_maps_message_capture_env_to_cli(save_messages, expected_flag):
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "MODEL_PATH": "/tmp/model",
        "DATA_PATH": "/tmp/data",
        "REPO_ROOT": str(repo_root),
        "RUN_INFER_SCRIPT": str(repo_root / "examples/blackbox_recipes/claude_code/run_infer.sh"),
        "SAVE_TRAJECTORY_MESSAGES": save_messages,
    }
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'python() { printf "%s\\n" "$@"; }; source "$RUN_INFER_SCRIPT"',
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert ("--save-trajectory-messages" in completed.stdout.splitlines()) is expected_flag


def test_claude_command_subagent_switch_preserves_model_routing_and_web_block():
    disabled = build_claude_command(task="fix it", base_url="http://gateway", max_turns=10)
    assert "CLAUDE_CODE_FORK_SUBAGENT=0" in disabled
    assert "--disallowedTools Agent Task WebFetch WebSearch" in disabled

    enabled = build_claude_command(
        task="fix it",
        base_url="http://gateway",
        max_turns=10,
        model="gateway-model",
        enable_subagents=True,
    )
    assert "CLAUDE_CODE_FORK_SUBAGENT=1" in enabled
    assert "CLAUDE_CODE_SUBAGENT_MODEL=gateway-model" in enabled
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL=gateway-model" in enabled
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=gateway-model" in enabled
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL=gateway-model" in enabled
    assert "--disallowedTools WebFetch WebSearch" in enabled
    assert "--disallowedTools Agent" not in enabled
    assert "--disallowedTools Task" not in enabled


def test_claude_task_only_requires_subagent_in_validation_mode():
    normal = build_claude_task("fix the bug")
    required = build_claude_task("fix the bug", require_subagent=True)

    assert "MUST spawn at least one subagent" not in normal
    assert "MUST spawn at least one subagent" in required
    assert "explicitly use its findings" in required


def test_reward_metadata_is_allowlisted_for_saved_artifacts():
    result = compute_score(
        "swe",
        "solution",
        "ground truth",
        extra_info={
            "reward_score": 1.0,
            "claude_code_exit_code": 0,
            "resolved": True,
            "eval_completed": True,
            "api_key": "must-not-be-saved",
        },
    )

    assert result == {
        "score": 1.0,
        "claude_code_exit_code": 0,
        "resolved": True,
        "eval_completed": True,
    }


@pytest.mark.asyncio
async def test_tq_capture_writes_valid_multiple_chain_artifacts_without_secrets(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tq, "async_kv_put", tq.async_kv_put)
    monkeypatch.setattr(tq, "async_kv_batch_put", tq.async_kv_batch_put)
    uid = "uid-with-hyphens"
    capture = _install_tq_capture(uids=[uid])
    caplog.set_level("INFO")
    fields = _list_of_tq_fields_to_tensordict(
        [
            {
                "prompts": torch.tensor([1, 2]),
                "responses": torch.tensor([3, 4]),
                "response_mask": torch.tensor([1, 1]),
                "rollout_log_probs": torch.tensor([-0.1, -0.2]),
                "rm_scores": torch.tensor([0.0, 1.0]),
                "num_turns": torch.tensor(1),
                "data_source": "swe-bench",
                "messages": [{"role": "user", "content": "subagent request"}],
                "reward_extra_info": {
                    "score": 1.0,
                    "claude_code_exit_code": 0,
                    "api_key": "must-not-be-saved",
                },
            },
            {
                "prompts": torch.tensor([1, 2]),
                "responses": torch.tensor([5, 6, 7]),
                "response_mask": torch.tensor([0, 1, 1]),
                "rollout_log_probs": torch.tensor([0.0, -0.3, -0.4]),
                "rm_scores": torch.tensor([0.0, 0.0, 1.0]),
                "num_turns": torch.tensor(2),
                "data_source": "swe-bench",
                "messages": [
                    {"role": "user", "content": "main request"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_exact",
                                "type": "function",
                                "function": {"name": "Agent", "arguments": {"prompt": "inspect"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_exact", "content": "finding"},
                ],
                "reward_extra_info": {
                    "score": 1.0,
                    "resolved": True,
                    "token": "must-not-be-saved",
                },
            },
        ]
    )
    await tq.async_kv_batch_put(
        keys=[f"{uid}_0_0", f"{uid}_0_1"],
        fields=fields,
        tags=[{"status": "success", "secret": "x"}, {"status": "success", "secret": "y"}],
        partition_id="train",
    )
    await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "finished"})

    assert "[progress] completed=1/1 sample_index=0" in caplog.text
    assert "status=finished sessions=1 trajectories=2" in caplog.text

    records = _records_for_output(capture, [uid])
    errors = _validate_trajectory_records(records, require_subagent=True)
    assert errors == []
    assert [record["trajectory_index"] for record in records] == [0, 1]
    assert [record["is_final_trajectory"] for record in records] == [False, True]
    assert {record["data_source"] for record in records} == {"swe-bench"}
    assert records[0]["reward_extra_info"] == {"score": 1.0, "claude_code_exit_code": 0}
    assert records[1]["reward_extra_info"] == {"score": 1.0, "resolved": True}
    assert records[0]["tags"] == {"status": "success"}
    assert records[0]["messages"] == [{"role": "user", "content": "subagent request"}]
    assert records[1]["messages"][-2]["tool_calls"][0]["id"] == "call_exact"
    assert records[1]["messages"][-1]["tool_call_id"] == "call_exact"

    summary = _build_artifact_summary(
        report={"resolved": 1, "total": 1, "mean_score": 1.0, "per_sample_scores": [1.0]},
        records=records,
        capture=capture,
        run_config={
            "tool_image": "claude-code-tool:latest",
            "save_trajectory_messages": True,
        },
        validation_errors=errors,
    )
    trajectories_path, summary_path = _write_artifacts(tmp_path, records=records, summary=summary)

    saved_records = [json.loads(line) for line in trajectories_path.read_text().splitlines()]
    saved_summary = json.loads(summary_path.read_text())
    assert len(saved_records) == 2
    assert saved_records[1]["schema_version"] == 1
    assert saved_records[1]["messages"] == records[1]["messages"]
    assert "must-not-be-saved" not in trajectories_path.read_text()
    assert saved_summary["multiple_chains_sessions"] == 1
    assert saved_summary["run_config"]["save_trajectory_messages"] is True
    assert saved_summary["validation"] == {"passed": True, "errors": []}


def test_empty_artifacts_are_written_and_required_subagent_validation_fails(tmp_path):
    capture = InferenceCapture()
    errors = _validate_trajectory_records([], require_subagent=True)
    summary = _build_artifact_summary(
        report={"resolved": 0, "total": 1, "mean_score": 0.0, "per_sample_scores": [0.0]},
        records=[],
        capture=capture,
        run_config={},
        validation_errors=errors,
    )
    trajectories_path, summary_path = _write_artifacts(tmp_path, records=[], summary=summary)

    assert trajectories_path.read_text() == ""
    assert json.loads(summary_path.read_text())["validation"]["passed"] is False
    assert errors == ["No successful session produced multiple trajectories; required subagent was not observed"]
