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

**This recipe is self-contained.** It builds the sandbox via the shared
[`uni_agent.sandbox`](../../uni_agent/sandbox) abstraction (`build_sandbox` + `SandboxConfig`),
selecting the provider through `SANDBOX_PROVIDER` (default `openyuanrong`);
everything else (`dataset.py`, `reward.py`, `build_tool.sh`, `run_train.sh`,
config) lives in this directory and does not depend on `mini_swe_agent/`.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `claude_code` | Claude Code sidecar runner |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `openyuanrong_sandbox_sdk.Mount` and `sandbox.commands.run()` |

## Architecture

```text
[Rollouter Host: claude_code_runner]
  |
  |-- build_sandbox(SandboxConfig(provider=SANDBOX_PROVIDER, image, mounts=[{target="/opt/claude-code", image_url=sidecar}], upstream=...))
  |     `-- provider Sandbox (e.g. openYuanrong: Sandbox(mounts=[Mount(target="/opt/claude-code", ...)]))
  |
  |-- sandbox.exec_shell("<env> /opt/claude-code/bin/claude -p <task> ...")
  |     `-- [Inside Sandbox]
  |           claude binary, ANTHROPIC_BASE_URL -> 127.0.0.1:<proxy_port>
  |           commands run inside the SWE-bench sandbox /testbed
  |
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Prerequisites

1. **OpenYuanrong** - set `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`.
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
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
CLAUDE_CODE_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/blackbox_recipes/claude_code/run_train.sh
```

The training YAML keeps `claude_code` as the only runner:

```yaml
agent_runner_fqn: examples.blackbox_recipes.claude_code.claude_code_runner.claude_code_runner
```

## 3. Standalone SWE-bench Inference

`run_infer.sh` runs rollout and reward evaluation without starting the trainer.
`MAX_SAMPLES=N` selects the first `N` Parquet rows in their original order;
`MAX_SAMPLES=-1` selects all rows. Zero and values below `-1` are rejected
before Ray or the model server is initialized.

```bash
OPENYUANRONG_SERVER_ADDRESS="<address>" \
OPENYUANRONG_TOKEN="<token>" \
MODEL_PATH=~/models/Qwen3.5-9B \
DATA_PATH=~/data/swe_agent/swe_bench_verified.parquet \
MAX_SAMPLES=10 \
VLLM_LANGUAGE_MODEL_ONLY=1 \
MAX_NUM_BATCHED_TOKENS=8192 \
OUTPUT_DIR="$PWD/outputs/claude_code_infer/swebench-first-10" \
bash examples/blackbox_recipes/claude_code/run_infer.sh
```

SWE-bench tasks are text-only, so the wrapper enables vLLM's language-model-only
mode by default. This prevents multimodal modules and dummy image profiling from
consuming GPU memory. Set `VLLM_LANGUAGE_MODEL_ONLY=0` only when running a dataset
that contains multimodal inputs. `MAX_NUM_BATCHED_TOKENS` controls the engine's
per-batch scheduling budget independently of the model context length.

If `OUTPUT_DIR` is omitted, the wrapper creates a distinct timestamped run path
under `outputs/claude_code_infer/`. Each completed session is persisted by the
Framework as it finishes; the top-level `summary.json` is atomically published
only after generation, reporting, and gateway shutdown complete.

```text
outputs/claude_code_infer/<run-id>/
|-- summary.json
|-- session-sample-0-rollout-0-<uuid>/
|   |-- framework.log
|   |-- task.log
|   |-- trajectory.json
|   `-- trajectory.npz
`-- session-sample-1-rollout-0-<uuid>/
    |-- framework.log
    |-- task.log
    |-- trajectory.json
    `-- trajectory.npz
```

`trajectory.json` contains readable reward/length metadata and, by default,
the exact normalized message history. `trajectory.npz` contains compressed
token IDs, response masks, and log probabilities. If a run is interrupted,
already completed session directories remain available even though the final
`summary.json` is intentionally absent.

Captured messages can contain SWE-bench prompts, repository code, tool inputs,
and tool outputs. Treat the output directory as sensitive. To retain only the
token trajectory and compact metadata, disable readable message capture:

```bash
SAVE_TRAJECTORY_MESSAGES=0 \
MAX_SAMPLES=10 \
bash examples/blackbox_recipes/claude_code/run_infer.sh
```

Claude Code's built-in subagent decision remains on its official defaults; the
recipe does not require or force delegation.

## 4. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_SAMPLES` | `-1` | Dataset prefix length; positive values run rows `[0, N)`, while `-1` runs all rows |
| `OUTPUT_DIR` | timestamped path under `outputs/claude_code_infer/` | Per-run root for session artifacts and the completed `summary.json` |
| `SAVE_TRAJECTORY_MESSAGES` | `1` | Include exact normalized messages in `trajectory.json`; set `0` for token-only capture |
| `VLLM_LANGUAGE_MODEL_ONLY` | `1` | Disable vLLM multimodal inputs and modules for text-only SWE-bench; accepts `0` or `1` |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Maximum tokens scheduled in one engine batch; must be a positive integer |
| `AGENT_MAX_TURNS` | `100` | `claude --max-turns` (the agent's turn budget); read by the runner from the `AGENT_MAX_TURNS` env var |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout (seconds) |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Max wall time for the claude process in the sandbox |
| `CLAUDE_CODE_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest` | Sidecar tool image |
| `CLAUDE_CODE_PROXY_PORT` | `38197` | Sandbox-local reverse tunnel port shared by OpenYuanrong and `ANTHROPIC_BASE_URL` |
| `CONDA_ENV` | `testbed` | Conda env activated inside the sandbox before running claude |

`AGENT_MAX_TURNS` is the only knob that bounds the agent. The trainer's
`multi_turn.max_assistant_turns` is not enforced on the blackbox rollout path
(`AgentFrameworkRolloutAdapter`) — claude runs to its own `--max-turns` inside
the sandbox and the gateway counts the turns afterward — so it is not exposed as
a separate knob. A value of `1` would cripple the agent, hence the default `100`.
