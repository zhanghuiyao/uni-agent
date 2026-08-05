from __future__ import annotations

import shlex

from examples.blackbox_recipes.claude_code.claude_code_runner import build_claude_command


def _command_argv(command: str) -> list[str]:
    invocation = command.split("cd /testbed; ", maxsplit=1)[1]
    tokens = shlex.split(invocation)
    executable_index = tokens.index("/opt/claude-code/bin/claude")
    return tokens[executable_index:]


def test_build_claude_command_uses_official_subagent_defaults():
    command = build_claude_command(
        task="fix the bug",
        base_url="http://gateway:8000/session/test",
        max_turns=12,
        model="policy",
    )

    unset_prefix = command.split(";", maxsplit=1)[0].split()
    assert unset_prefix[0] == "unset"
    assert {
        "CLAUDE_CODE_FORK_SUBAGENT",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS",
    }.issubset(unset_prefix)
    for assignment in (
        "CLAUDE_CODE_FORK_SUBAGENT=",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=",
        "CLAUDE_CODE_SUBAGENT_MODEL=",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=",
    ):
        assert assignment not in command

    argv = _command_argv(command)
    disallowed_index = argv.index("--disallowedTools")
    assert argv[disallowed_index + 1 :] == ["AskUserQuestion", "WebFetch", "WebSearch"]
    assert "Agent" not in argv
    assert "Task" not in argv
    for key in (
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        assert f"{key}=policy" in command
