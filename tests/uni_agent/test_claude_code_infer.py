import json

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
    )
    runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code

    assert runner.runner_kwargs.tool_image == "claude-code-tool:latest"
    assert runner.runner_kwargs.enable_subagents is True
    assert runner.runner_kwargs.require_subagent is True


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
async def test_tq_capture_writes_valid_multiple_chain_artifacts_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(tq, "async_kv_put", tq.async_kv_put)
    monkeypatch.setattr(tq, "async_kv_batch_put", tq.async_kv_batch_put)
    capture = _install_tq_capture()
    uid = "uid-with-hyphens"
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

    records = _records_for_output(capture, [uid])
    errors = _validate_trajectory_records(records, require_subagent=True)
    assert errors == []
    assert [record["trajectory_index"] for record in records] == [0, 1]
    assert [record["is_final_trajectory"] for record in records] == [False, True]
    assert {record["data_source"] for record in records} == {"swe-bench"}
    assert records[0]["reward_extra_info"] == {"score": 1.0, "claude_code_exit_code": 0}
    assert records[1]["reward_extra_info"] == {"score": 1.0, "resolved": True}
    assert records[0]["tags"] == {"status": "success"}

    summary = _build_artifact_summary(
        report={"resolved": 1, "total": 1, "mean_score": 1.0, "per_sample_scores": [1.0]},
        records=records,
        capture=capture,
        run_config={"tool_image": "claude-code-tool:latest"},
        validation_errors=errors,
    )
    trajectories_path, summary_path = _write_artifacts(tmp_path, records=records, summary=summary)

    saved_records = [json.loads(line) for line in trajectories_path.read_text().splitlines()]
    saved_summary = json.loads(summary_path.read_text())
    assert len(saved_records) == 2
    assert "must-not-be-saved" not in trajectories_path.read_text()
    assert saved_summary["multiple_chains_sessions"] == 1
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
