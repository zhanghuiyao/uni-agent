#!/usr/bin/env python3
"""Decode saved Claude Code trajectory token IDs into readable JSONL.

The companion artifact keeps exact Gateway ``messages`` when present, replaces
token arrays with readable text, and adds a best-effort Qwen3.5 ChatML message
parse. Inferred messages are not a lossless reconstruction of provider objects.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_TOKEN_FIELDS = frozenset({"prompt_ids", "response_ids", "response_mask", "response_logprobs"})
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_RESPONSE_START = "<tool_response>"
_TOOL_RESPONSE_END = "</tool_response>"
_TOOL_ID_WARNING = (
    "Qwen3.5 chat templates do not encode tool call IDs; decoded tool_calls omit id "
    "and tool messages omit tool_call_id."
)


def _record_label(record: dict[str, Any]) -> str:
    return (
        f"uid={record.get('uid', '?')} "
        f"session_index={record.get('session_index', '?')} "
        f"trajectory_index={record.get('trajectory_index', '?')}"
    )


def _read_token_ids(record: dict[str, Any], field: str) -> list[int]:
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{_record_label(record)}: {field} must be a list of integers")
    return value


def _read_response_mask(record: dict[str, Any]) -> list[int]:
    value = _read_token_ids(record, "response_mask")
    invalid = sorted(set(value) - {0, 1})
    if invalid:
        raise ValueError(f"{_record_label(record)}: response_mask contains values other than 0/1: {invalid}")
    return value


def _decode(tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))


def _warn_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _parse_parameter_value(value: str) -> Any:
    """Recover unambiguous JSON containers while leaving scalar values as text."""
    stripped = value.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _parse_qwen35_tool_call(body: str, warnings: list[str]) -> dict[str, Any] | None:
    function_match = re.search(r"<function=([^>\n]+)>", body)
    if function_match is not None:
        function_name = function_match.group(1).strip()
        function_end = body.find("</function>", function_match.end())
        if function_end < 0:
            _warn_once(warnings, "Trajectory contains an unclosed <function> block.")
        function_body = body[function_match.end() : function_end if function_end >= 0 else len(body)]
        arguments: dict[str, Any] = {}
        cursor = 0
        while True:
            parameter_match = re.search(r"<parameter=([^>\n]+)>", function_body[cursor:])
            if parameter_match is None:
                break
            value_start = cursor + parameter_match.end()
            value_end = function_body.find("</parameter>", value_start)
            if value_end < 0:
                value_end = len(function_body)
                _warn_once(warnings, "Trajectory contains an unclosed <parameter> block.")
            parameter_name = parameter_match.group(1).strip()
            arguments[parameter_name] = _parse_parameter_value(function_body[value_start:value_end])
            cursor = value_end + len("</parameter>")
            if value_end == len(function_body):
                break
        if function_name:
            return {
                "type": "function",
                "function": {"name": function_name, "arguments": arguments},
            }

    try:
        parsed = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    function = parsed.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = parsed.get("name")
        arguments = parsed.get("arguments", parsed.get("input", {}))
    if not isinstance(name, str) or not name:
        return None
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_assistant_content(content: str, warnings: list[str]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    remaining = content.strip("\n")
    think_start = remaining.find("<think>")
    if think_start >= 0:
        think_end = remaining.find("</think>", think_start + len("<think>"))
        if think_end < 0:
            message["reasoning_content"] = remaining[think_start + len("<think>") :].strip("\n")
            remaining = remaining[:think_start]
            _warn_once(warnings, "Trajectory contains an unclosed <think> block.")
        else:
            message["reasoning_content"] = remaining[think_start + len("<think>") : think_end].strip("\n")
            remaining = remaining[:think_start] + remaining[think_end + len("</think>") :]

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    cursor = 0
    while True:
        call_start = remaining.find(_TOOL_CALL_START, cursor)
        if call_start < 0:
            text_parts.append(remaining[cursor:])
            break
        text_parts.append(remaining[cursor:call_start])
        body_start = call_start + len(_TOOL_CALL_START)
        call_end = remaining.find(_TOOL_CALL_END, body_start)
        if call_end < 0:
            body = remaining[body_start:]
            raw_block = remaining[call_start:]
            _warn_once(warnings, "Trajectory contains an unclosed <tool_call> block.")
            cursor = len(remaining)
        else:
            body = remaining[body_start:call_end]
            raw_block = remaining[call_start : call_end + len(_TOOL_CALL_END)]
            cursor = call_end + len(_TOOL_CALL_END)
        tool_call = _parse_qwen35_tool_call(body, warnings)
        if tool_call is None:
            text_parts.append(raw_block)
            _warn_once(
                warnings,
                "Could not parse a Qwen3.5 <tool_call> block; preserved it as assistant content.",
            )
        else:
            tool_calls.append(tool_call)
        if call_end < 0:
            break

    message["content"] = "".join(text_parts).strip("\n")
    if tool_calls:
        message["tool_calls"] = tool_calls
        _warn_once(warnings, _TOOL_ID_WARNING)
    return message


def _parse_user_content(content: str, warnings: list[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    cursor = 0
    found_tool_response = False
    while True:
        response_start = content.find(_TOOL_RESPONSE_START, cursor)
        if response_start < 0:
            tail = content[cursor:].strip("\n")
            if tail or not found_tool_response:
                messages.append({"role": "user", "content": tail})
            break
        found_tool_response = True
        prefix = content[cursor:response_start].strip("\n")
        if prefix:
            messages.append({"role": "user", "content": prefix})
        body_start = response_start + len(_TOOL_RESPONSE_START)
        response_end = content.find(_TOOL_RESPONSE_END, body_start)
        if response_end < 0:
            response_content = content[body_start:].strip("\n")
            cursor = len(content)
            _warn_once(warnings, "Trajectory contains an unclosed <tool_response> block.")
        else:
            response_content = content[body_start:response_end].strip("\n")
            cursor = response_end + len(_TOOL_RESPONSE_END)
        messages.append({"role": "tool", "content": response_content})
        _warn_once(warnings, _TOOL_ID_WARNING)
        if response_end < 0:
            break
    return messages


def parse_qwen35_chat_template(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Best-effort conversion of decoded Qwen3.5 ChatML into role messages."""
    messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    cursor = 0
    while True:
        message_start = text.find(_IM_START, cursor)
        if message_start < 0:
            break
        role_start = message_start + len(_IM_START)
        role_end = text.find("\n", role_start)
        if role_end < 0:
            _warn_once(warnings, "Trajectory ended before a decoded message role was complete.")
            break
        role = text[role_start:role_end].strip()
        message_end = text.find(_IM_END, role_end + 1)
        if message_end < 0:
            content = text[role_end + 1 :]
            cursor = len(text)
            _warn_once(warnings, "Trajectory ended before <|im_end|>; the final decoded message may be truncated.")
        else:
            content = text[role_end + 1 : message_end]
            cursor = message_end + len(_IM_END)

        if role == "assistant":
            messages.append(_parse_assistant_content(content, warnings))
        elif role == "user":
            messages.extend(_parse_user_content(content, warnings))
        elif role == "system":
            messages.append({"role": "system", "content": content.strip("\n")})
        elif role == "tool":
            messages.append({"role": "tool", "content": content.strip("\n")})
            _warn_once(warnings, _TOOL_ID_WARNING)
        else:
            _warn_once(warnings, f"Skipped unsupported decoded message role: {role!r}.")
        if message_end < 0:
            break

    if not messages:
        _warn_once(warnings, "No Qwen3.5 ChatML message boundaries found.")
    return messages, warnings


def _decode_response_segments(
    tokenizer: Any,
    response_ids: list[int],
    response_mask: list[int],
    *,
    skip_special_tokens: bool,
    label: str,
) -> list[dict[str, Any]]:
    if len(response_ids) != len(response_mask):
        raise ValueError(
            f"{label}: response_ids/response_mask length mismatch: {len(response_ids)} != {len(response_mask)}"
        )
    if not response_ids:
        return []

    segments = []
    start = 0
    mask_value = response_mask[0]
    for index in range(1, len(response_mask) + 1):
        if index < len(response_mask) and response_mask[index] == mask_value:
            continue
        segment_ids = response_ids[start:index]
        segments.append(
            {
                "kind": "model_output" if mask_value == 1 else "context",
                "mask": mask_value,
                "token_start": start,
                "token_end": index,
                "token_count": len(segment_ids),
                "text": _decode(tokenizer, segment_ids, skip_special_tokens=skip_special_tokens),
            }
        )
        if index < len(response_mask):
            start = index
            mask_value = response_mask[index]
    return segments


def decode_trajectory_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    skip_special_tokens: bool = False,
) -> dict[str, Any]:
    """Return a readable companion record for one saved trajectory."""
    prompt_ids = _read_token_ids(record, "prompt_ids")
    response_ids = _read_token_ids(record, "response_ids")
    response_mask = _read_response_mask(record)
    label = _record_label(record)
    segments = _decode_response_segments(
        tokenizer,
        response_ids,
        response_mask,
        skip_special_tokens=skip_special_tokens,
        label=label,
    )

    decoded = {key: value for key, value in record.items() if key not in _TOKEN_FIELDS}
    decoded["decoded_prompt"] = {
        "token_count": len(prompt_ids),
        "text": _decode(tokenizer, prompt_ids, skip_special_tokens=skip_special_tokens),
    }
    decoded["decoded_response"] = {
        "token_count": len(response_ids),
        "text": _decode(tokenizer, response_ids, skip_special_tokens=skip_special_tokens),
        "segments": segments,
    }
    parse_text = _decode(tokenizer, prompt_ids + response_ids, skip_special_tokens=False)
    decoded_messages, decoded_message_warnings = parse_qwen35_chat_template(parse_text)
    decoded["decoded_messages"] = decoded_messages
    decoded["decoded_message_parser"] = "qwen3.5_chat_template"
    decoded["decoded_message_warnings"] = decoded_message_warnings
    return decoded


def decode_trajectory_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Any,
    *,
    skip_special_tokens: bool = False,
) -> int:
    """Stream-decode a trajectory JSONL file and atomically replace the output."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("input and output paths must be different")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    try:
        with source.open(encoding="utf-8") as input_file, temporary.open("w", encoding="utf-8") as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{source}:{line_number}: expected a JSON object")
                decoded = decode_trajectory_record(
                    record,
                    tokenizer,
                    skip_special_tokens=skip_special_tokens,
                )
                output_file.write(json.dumps(decoded, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.decoded{input_path.suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Claude Code trajectory token IDs with the tokenizer that generated them."
    )
    parser.add_argument("--model-path", required=True, help="Exact model/tokenizer path used for inference")
    parser.add_argument("--input", required=True, type=Path, help="Input trajectories.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL (default: <input-stem>.decoded.jsonl next to the input)",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Hide special tokens; disabled by default so chat/tool markers remain visible",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading a trusted local tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    output_path = args.output or _default_output_path(args.input)
    count = decode_trajectory_jsonl(
        args.input,
        output_path,
        tokenizer,
        skip_special_tokens=args.skip_special_tokens,
    )
    print(f"Decoded {count} trajectories to {output_path.expanduser().resolve()}")


if __name__ == "__main__":
    main()
