# Claude Code In-Sandbox Execution

## Overview

Claude Code runs inside the SWE-bench sandbox through a sidecar tool image. The
external runner creates the sandbox, mounts the tool image at `/opt/claude-code`,
invokes the `claude` binary against the gateway URL, and evaluates the reward in
the same sandbox.

Unlike the mini-swe-agent recipe, there is no in-sandbox Python entrypoint
(`run_agent.py`): the runner builds a single `claude -p ...` command and executes
it directly. The agent reaches the LLM gateway through the sandbox-internal
tunnel (`ANTHROPIC_BASE_URL` rewritten to `http://127.0.0.1:<proxy_port>`).

The Claude Code tool image uses a Node builder to install the
`@anthropic-ai/claude-code` npm package, then copies the result into a minimal
`FROM scratch` final stage. The sandbox base image therefore does not need Node
or npm for the sidecar tool runtime.

**This recipe is self-contained.** It shares only
[`../sandbox_client.py`](../sandbox_client.py) with the mini-swe-agent recipe;
everything else (`dataset.py`, `reward.py`, `build_tool.sh`, `run_train.sh`,
config) lives in this directory and does not depend on `mini_swe_agent/`.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `claude_code` | Claude Code sidecar runner |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `akernel_sdk.Mount` and `sandbox.commands.run()` |

## Architecture

```text
[Rollouter Host: claude_code_runner]
  |
  |-- SandboxClient.create(image, sidecar_image, sidecar_target="/opt/claude-code")
  |     `-- akernel: Sandbox(mounts=[Mount(target="/opt/claude-code", ...)])
  |
  |-- sandbox.run("<env> /opt/claude-code/bin/claude -p <task> ...")
  |     `-- [Inside Sandbox]
  |           claude binary, ANTHROPIC_BASE_URL -> 127.0.0.1:<proxy_port>
  |           commands run inside the SWE-bench sandbox /testbed
  |
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Prerequisites

1. **AKernel** — set `AKERNEL_SERVER_ADDRESS` and `AKERNEL_TOKEN`.
2. **Tool image** — build the claude-code tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

`claude_code` is injected into the SWE-bench sandbox as a sidecar tool image.
Use `build_tool.sh` to build it.

| Default tool image | Dockerfile | Sandbox mount path | Image contents |
|--------------------|------------|--------------------|----------------|
| `claude-code-tool:latest` | `Dockerfile.claude-code-tool` | `/opt/claude-code` | Node-built `@anthropic-ai/claude-code` npm package |

```bash
# Use the default npm registry.
bash examples/blackbox_recipes/claude_code/build_tool.sh

# Use a custom npm mirror.
bash examples/blackbox_recipes/claude_code/build_tool.sh --npm-registry https://registry.npmmirror.com

# Pin a specific claude-code version.
bash examples/blackbox_recipes/claude_code/build_tool.sh --tool-version latest

# Build and push to a remote registry.
bash examples/blackbox_recipes/claude_code/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_IMAGE` | `claude-code-tool` | Image name |
| `TOOL_TAG` | `latest` | Image tag |
| `TOOL_VERSION` | `latest` | `@anthropic-ai/claude-code` package version (`--tool-version`) |
| `NPM_REGISTRY` | unset, use npm default | npm registry URL (`--npm-registry`) |

After pushing, point training at it with `CLAUDE_CODE_TOOL_IMAGE`.

## 2. Training (Fully Async)

```bash
AKERNEL_SERVER_ADDRESS="6.2.179.37:8888" \
AKERNEL_TOKEN="<token>" \
CLAUDE_CODE_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/blackbox_recipes/claude_code/run_train.sh
```

The training YAML keeps `claude_code` as the only runner:

```yaml
agent_runner_fqn: examples.blackbox_recipes.claude_code.claude_code_runner.claude_code_runner
```

## 3. Inference-only Multiple Chains + Subagents Validation

This path runs rollout and reward without starting the trainer. It exercises the
complete Claude Code flow: the main agent invokes an `Agent`/`Task` tool, Claude
Code sends the subagent request through the same Gateway session, and the final
tool result returns to the original main chain.

Two switches control the behavior:

- `ENABLE_SUBAGENTS=1` sets `CLAUDE_CODE_FORK_SUBAGENT=1` and allows the
  `Agent`/`Task` tools. Claude may still decide that a subagent is unnecessary.
- `REQUIRE_SUBAGENT=1` is an acceptance-test mode. It additionally instructs
  Claude to spawn at least one subagent and exits nonzero if no successful
  multiple-chain session is captured. It requires `ENABLE_SUBAGENTS=1`.

`N` is the number of independent rollout sessions per sample; it is not the
number of chains. A single session with one spawned subagent normally produces
at least two trajectory records: the subagent chain followed by the resumed
main chain.

### 3.1 Build and Push the Tool Image

Skip this step when `CLAUDE_CODE_TOOL_IMAGE` already points to an image that the
AKernel sandbox can access.

```bash
TOOL_VERSION=latest TOOL_TAG=latest \
bash examples/blackbox_recipes/claude_code/build_tool.sh \
  --registry <registry>
```

To make a run reproducible, replace `latest` with an explicit Claude Code
version in both variables and use the matching image tag below.

### 3.2 Run One Forced-subagent Sample

Use a healthy Python environment that can import `httpx`, `ray`, and `torch`.
The data file must use the SWE-bench parquet schema expected by this recipe.

```bash
AKERNEL_SERVER_ADDRESS=<server> \
AKERNEL_TOKEN=<token> \
CLAUDE_CODE_TOOL_IMAGE=<registry>/claude-code-tool:latest \
MODEL_PATH=<model> \
DATA_PATH=<swe_parquet> \
MAX_SAMPLES=1 \
N=1 \
ENABLE_SUBAGENTS=1 \
REQUIRE_SUBAGENT=1 \
OUTPUT_DIR=/tmp/uni-agent-claude-subagent \
bash examples/blackbox_recipes/claude_code/run_infer.sh
```

For ordinary inference where Claude decides naturally whether to delegate, use
`ENABLE_SUBAGENTS=1 REQUIRE_SUBAGENT=0`. Leave both at `0` to preserve the
original single-agent behavior.

### 3.3 Inspect the Saved Trajectories

The output directory contains:

- `trajectories.jsonl`: one record per finalized chain, including token IDs,
  response mask, rollout log probabilities, reward, turn count, and final-chain
  marker. With the default `SAVE_TRAJECTORY_MESSAGES=1`, each record also has
  an optional top-level `messages` field containing that chain's exact,
  normalized OpenAI-style message history. This backward-compatible extension
  keeps `schema_version: 1`.
- `summary.json`: run configuration, resolve statistics, per-session trajectory
  counts, and structural validation results.

Check the high-level acceptance result:

```bash
jq '{multiple_chains_sessions, validation, per_session}' \
  /tmp/uni-agent-claude-subagent/summary.json
```

Inspect every session directly from the JSONL:

```bash
jq -s '
  group_by([.uid, .session_index])
  | map({
      uid: .[0].uid,
      session_index: .[0].session_index,
      trajectory_count: length,
      trajectory_indexes: map(.trajectory_index),
      final_count: ([.[] | select(.is_final_trajectory)] | length),
      reward_scores: (map(.reward_score) | unique),
      lengths_aligned: all(.[];
        ((.response_ids | length) == (.response_mask | length)) and
        ((.response_ids | length) == (.response_logprobs | length)))
    })
' /tmp/uni-agent-claude-subagent/trajectories.jsonl
```

Decode the saved token IDs with the exact model/tokenizer used for inference:

```bash
python examples/blackbox_recipes/claude_code/decode_trajectories.py \
  --model-path /path/to/model \
  --input /tmp/uni-agent-claude-subagent/trajectories.jsonl \
  --output /tmp/uni-agent-claude-subagent/trajectories.decoded.jsonl
```

The decoder keeps an existing exact `messages` field unchanged and adds
`decoded_messages`, `decoded_message_parser`, and
`decoded_message_warnings`. `decoded_messages` is a best-effort reconstruction
from `prompt_ids + response_ids` using Qwen3.5 ChatML boundaries; it recognizes
thinking content, function tool calls, and tool responses. The
[Qwen3.5 chat template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/ef3d031a90d340a92d71f83ec17d054e100ce713/tokenizer_config.json)
does not encode tool call IDs, so inferred tool calls omit `id`, inferred tool
messages omit `tool_call_id`, and the decoder emits a warning instead of
inventing a value. `--skip-special-tokens` only changes the human-readable
`decoded_prompt`, `decoded_response`, and segment text; structural parsing
always uses a second decode with special tokens retained.

`messages` is the exact Gateway-side history after the provider adapter has
normalized the request into OpenAI-style messages. It is not the original
Anthropic content-block payload. In particular, real tool-call IDs are present
in this field and can be paired with `tool_call_id`; they cannot be recovered
reliably from a token-only trajectory.

Message capture is intended for inference/debugging artifacts. It can persist
the complete prompt, tool output, credentials copied into messages, and base64
multimodal data. Keep the output directory access-controlled and set
`SAVE_TRAJECTORY_MESSAGES=0` when this data must not be retained. The generic
training configuration remains opt-in through
`actor_rollout_ref.rollout.custom.agent_framework.capture_messages=false`.

The run passes when `multiple_chains_sessions >= 1` and
`validation.passed == true`. The validator also requires contiguous trajectory
indexes, exactly one final trajectory per session, aligned response fields,
identical broadcast rewards, and both context (`0`) and generated-output (`1`)
mask values in the resumed final main chain. SWE task resolution is reported but
is not required for this feature-level validation. Even when validation fails,
the JSONL and summary are written before the process exits nonzero.

## 4. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_TURNS` | `100` | `claude --max-turns` (the agent's turn budget); read by the runner from the `AGENT_MAX_TURNS` env var |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout (seconds) |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Max wall time for the claude process in the sandbox |
| `CLAUDE_CODE_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest` | Sidecar tool image |
| `CONDA_ENV` | `testbed` | Conda env activated inside the sandbox before running claude |
| `ENABLE_SUBAGENTS` | `0` | Allow Claude Code `Agent`/`Task` tools and set `CLAUDE_CODE_FORK_SUBAGENT=1` |
| `REQUIRE_SUBAGENT` | `0` | Force a subagent in the validation prompt and fail if no multiple-chain session is captured |
| `OUTPUT_DIR` | `outputs/claude_code_infer` | Inference JSONL and summary output directory |
| `SAVE_TRAJECTORY_MESSAGES` | `1` | Add each chain's exact normalized Gateway message history to `trajectories.jsonl`; set to `0` for token-only artifacts |

`AGENT_MAX_TURNS` is the only knob that bounds the agent. The trainer's
`multi_turn.max_assistant_turns` is not enforced on the blackbox rollout path
(`AgentFrameworkRolloutAdapter`) — claude runs to its own `--max-turns` inside
the sandbox and the gateway counts the turns afterward — so it is not exposed as
a separate knob. A value of `1` would cripple the agent, hence the default `100`.
