import json

import pytest

from examples.blackbox_recipes.claude_code.decode_trajectories import (
    decode_trajectory_jsonl,
    decode_trajectory_record,
    parse_qwen35_chat_template,
)


class _ReadableTokenizer:
    _TOKENS = {
        1: "<|im_start|>system\nYou are helpful.<|im_end|>\n",
        2: "<|im_start|>user\nFix the issue.<|im_end|>\n",
        3: "<|im_start|>assistant\n<think>\n",
        4: (
            "inspect\n</think>\n\n"
            "<tool_call>\n"
            "<function=Agent>\n"
            "<parameter=prompt>\ninspect files\ncarefully\n</parameter>\n"
            '<parameter=options>\n{"depth": 2}\n</parameter>\n'
            '<parameter=tags>\n["python", "tests"]\n</parameter>\n'
            "<parameter=limit>\n3\n</parameter>\n"
            "</function>\n"
            "</tool_call><|im_end|>\n"
        ),
        5: ("<|im_start|>user\n<tool_response>\nfinding\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"),
        6: "done\n</think>\n\nFinal answer.<|im_end|>",
        7: "plain text without ChatML boundaries",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        text = "".join(self._TOKENS[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        return text


def _record(**overrides):
    record = {
        "schema_version": 1,
        "sample_index": 0,
        "uid": "uid-0",
        "session_index": 0,
        "trajectory_index": 1,
        "is_final_trajectory": True,
        "prompt_ids": [1, 2, 3],
        "response_ids": [4, 5, 6],
        "response_mask": [1, 0, 1],
        "response_logprobs": [-0.1, 0.0, -0.3],
        "reward_score": 1.0,
        "messages": [
            {"role": "user", "content": "Fix the issue."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_exact",
                        "type": "function",
                        "function": {"name": "Agent", "arguments": {"prompt": "inspect files"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_exact", "content": "finding"},
        ],
    }
    record.update(overrides)
    return record


def test_decode_trajectory_jsonl_preserves_exact_messages_and_parses_qwen35(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    destination = tmp_path / "trajectories.decoded.jsonl"
    source.write_text(json.dumps(_record()) + "\n")

    count = decode_trajectory_jsonl(source, destination, _ReadableTokenizer())

    assert count == 1
    decoded = json.loads(destination.read_text())
    assert decoded["messages"] == _record()["messages"]
    assert decoded["decoded_message_parser"] == "qwen3.5_chat_template"
    assert [message["role"] for message in decoded["decoded_messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    first_assistant = decoded["decoded_messages"][2]
    assert first_assistant["reasoning_content"] == "inspect"
    assert first_assistant["content"] == ""
    assert first_assistant["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "Agent",
                "arguments": {
                    "prompt": "inspect files\ncarefully",
                    "options": {"depth": 2},
                    "tags": ["python", "tests"],
                    "limit": "3",
                },
            },
        }
    ]
    assert decoded["decoded_messages"][3] == {"role": "tool", "content": "finding"}
    assert decoded["decoded_messages"][4] == {
        "role": "assistant",
        "content": "Final answer.",
        "reasoning_content": "done",
    }
    assert decoded["decoded_message_warnings"] == [
        "Qwen3.5 chat templates do not encode tool call IDs; decoded tool_calls omit id "
        "and tool messages omit tool_call_id."
    ]
    assert [segment["kind"] for segment in decoded["decoded_response"]["segments"]] == [
        "model_output",
        "context",
        "model_output",
    ]
    assert decoded["reward_score"] == 1.0
    assert "prompt_ids" not in decoded
    assert "response_ids" not in decoded
    assert "response_mask" not in decoded
    assert "response_logprobs" not in decoded


def test_skip_special_tokens_only_changes_display_text_not_message_parsing():
    decoded = decode_trajectory_record(_record(), _ReadableTokenizer(), skip_special_tokens=True)

    assert "<|im_start|>" not in decoded["decoded_prompt"]["text"]
    assert [message["role"] for message in decoded["decoded_messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_parse_qwen35_chat_template_handles_multiple_tool_responses_without_ids():
    text = (
        "<|im_start|>assistant\n"
        "<tool_call>\n<function=first>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=second>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\none\n</tool_response>\n"
        "<tool_response>\ntwo\n</tool_response><|im_end|>"
    )

    messages, warnings = parse_qwen35_chat_template(text)

    assert [tool_call["function"]["name"] for tool_call in messages[0]["tool_calls"]] == ["first", "second"]
    assert messages[1:] == [
        {"role": "tool", "content": "one"},
        {"role": "tool", "content": "two"},
    ]
    assert warnings == [
        "Qwen3.5 chat templates do not encode tool call IDs; decoded tool_calls omit id "
        "and tool messages omit tool_call_id."
    ]


def test_parse_qwen35_chat_template_reports_truncation_without_failing():
    messages, warnings = parse_qwen35_chat_template("<|im_start|>assistant\n<think>\npartial")

    assert messages == [{"role": "assistant", "content": "", "reasoning_content": "partial"}]
    assert "Trajectory ended before <|im_end|>; the final decoded message may be truncated." in warnings
    assert "Trajectory contains an unclosed <think> block." in warnings


def test_parse_qwen35_chat_template_reports_truncated_tool_call_without_failing():
    text = "<|im_start|>assistant\n<tool_call>\n<function=search>\n<parameter=query>partial"

    messages, warnings = parse_qwen35_chat_template(text)

    assert messages[0]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": {"query": "partial"},
    }
    assert "Trajectory contains an unclosed <tool_call> block." in warnings
    assert "Trajectory contains an unclosed <function> block." in warnings
    assert "Trajectory contains an unclosed <parameter> block." in warnings


def test_decode_reports_unrecognized_template_without_failing():
    decoded = decode_trajectory_record(
        _record(
            prompt_ids=[7],
            response_ids=[],
            response_mask=[],
            response_logprobs=[],
            messages=None,
        ),
        _ReadableTokenizer(),
    )

    assert decoded["decoded_messages"] == []
    assert decoded["decoded_message_warnings"] == ["No Qwen3.5 ChatML message boundaries found."]


def test_decode_trajectory_record_rejects_misaligned_response_mask():
    record = _record(response_ids=[4, 5], response_mask=[1])

    with pytest.raises(ValueError, match="response_ids/response_mask length mismatch"):
        decode_trajectory_record(record, _ReadableTokenizer())


def test_decode_trajectory_jsonl_rejects_invalid_json(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    destination = tmp_path / "trajectories.decoded.jsonl"
    source.write_text("{not-json}\n")

    with pytest.raises(ValueError, match="invalid JSON"):
        decode_trajectory_jsonl(source, destination, _ReadableTokenizer())

    assert not destination.exists()
