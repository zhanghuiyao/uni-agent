# claude-code 轨迹分析

## TL;DR

结论：默认cc配置下会视情况拉起subagent，不过在当前最新代码下(7.31)不会导致链的快速异常分裂，基本上每个subagent只会分裂一条链。

## 主要结果

> 运行记录：155:/tmp/uni-agent-cc-subagent-exp1/outputs

修改 uni-agent 中 subagent 相关配置为 cc 默认配置，在swe-bench verified选取前 50 个任务用默认的方式运行，50 个任务中有 3 个任务拉起了 subagent，共生成 66 条 chain:

- 47 个任务：1 条链，无 subagent
- 2  个任务（sample 22、44）：各 2 条链，分别拉起 1 个 subagent
- 1  个任务（sample 42）：15 条链，拉起 14 个 subagent

## 主要改动

> 基于 7.31 main 分支代码 commit_id: b49d017c

- 在 `_CC_QUIET_ENV` 中删除 `CLAUDE_CODE_FORK_SUBAGENT`/`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`
- 在 `disallowed_tools` 中删除 `Agent` / `Task`
- 指定 `CLAUDE_CODE_SUBAGENT_MODEL` 为默认模型

## 详细轨迹分析

| Sample | Instance | 链数 | Agentic turns（各链） | Subagent | Score |
|---:|---|---:|---|---:|---:|
| 0 | astropy__astropy-12907 | 1 | 28 | 否 | 1 |
| 1 | astropy__astropy-13033 | 1 | 51 | 否 | 0 |
| 2 | astropy__astropy-13236 | 1 | 29 | 否 | 0 |
| 3 | astropy__astropy-13398 | 1 | 100 | 否 | 0 |
| 4 | astropy__astropy-13453 | 1 | 100 | 否 | 0 |
| 5 | astropy__astropy-13579 | 1 | 56 | 否 | 0 |
| 6 | astropy__astropy-13977 | 1 | 44 | 否 | 0 |
| 7 | astropy__astropy-14096 | 1 | 100 | 否 | 0 |
| 8 | astropy__astropy-14182 | 1 | 38 | 否 | 0 |
| 9 | astropy__astropy-14309 | 1 | 25 | 否 | 1 |
| 10 | astropy__astropy-14365 | 1 | 20 | 否 | 0 |
| 11 | astropy__astropy-14369 | 1 | 63 | 否 | 0 |
| 12 | astropy__astropy-14508 | 1 | 46 | 否 | 0 |
| 13 | astropy__astropy-14539 | 1 | 84 | 否 | 0 |
| 14 | astropy__astropy-14598 | 1 | 100 | 否 | 0 |
| 15 | astropy__astropy-14995 | 1 | 22 | 否 | 0 |
| 16 | astropy__astropy-7166 | 1 | 8 | 否 | 0 |
| 17 | astropy__astropy-7336 | 1 | 22 | 否 | 0 |
| 18 | astropy__astropy-7606 | 1 | 6 | 否 | 0 |
| 19 | astropy__astropy-7671 | 1 | 11 | 否 | 0 |
| 20 | astropy__astropy-8707 | 1 | 63 | 否 | 0 |
| 21 | astropy__astropy-8872 | 1 | 31 | 否 | 0 |
| 22 | django__django-10097 | **2** | **3 / 100** | **是，1 个** | 0 |
| 23 | django__django-10554 | 1 | 100 | 否 | 0 |
| 24 | django__django-10880 | 1 | 87 | 否 | 0 |
| 25 | django__django-10914 | 1 | 46 | 否 | 0 |
| 26 | django__django-10973 | 1 | 100 | 否 | 0 |
| 27 | django__django-10999 | 1 | 7 | 否 | 0 |
| 28 | django__django-11066 | 1 | 65 | 否 | 1 |
| 29 | django__django-11087 | 1 | 100 | 否 | 0 |
| 30 | django__django-11095 | 1 | 66 | 否 | 0 |
| 31 | django__django-11099 | 1 | 25 | 否 | 0 |
| 32 | django__django-11119 | 1 | 17 | 否 | 0 |
| 33 | django__django-11133 | 1 | 30 | 否 | 0 |
| 34 | django__django-11138 | 1 | 100 | 否 | 0 |
| 35 | django__django-11141 | 1 | 90 | 否 | 0 |
| 36 | django__django-11149 | 1 | 100 | 否 | 0 |
| 37 | django__django-11163 | 1 | 100 | 否 | 0 |
| 38 | django__django-11179 | 1 | 36 | 否 | 0 |
| 39 | django__django-11206 | 1 | 78 | 否 | 0 |
| 40 | django__django-11211 | 1 | 100 | 否 | 0 |
| 41 | django__django-11239 | 1 | 43 | 否 | 0 |
| 42 | django__django-11265 | **15** | **6 / 5 / 26 / 11 / 14 / 12 / 4 / 11 / 11 / 13 / 73 / 32 / 5 / 13 / 54** | **是，14 个** | 0 |
| 43 | django__django-11276 | 1 | 46 | 否 | 1 |
| 44 | django__django-11292 | **2** | **13 / 69** | **是，1 个** | 0 |
| 45 | django__django-11299 | 1 | 100 | 否 | 0 |
| 46 | django__django-11333 | 1 | 92 | 否 | 0 |
| 47 | django__django-11400 | 1 | 77 | 否 | 0 |
| 48 | django__django-11433 | 1 | 68 | 否 | 0 |
| 49 | django__django-11451 | 1 | 54 | 否 | 0 |

- 多链任务：
  - sample 22：1 条 100-turn 主链 + 1 条 3-turn subagent 链；总计 103 generations。
  - sample 42：1 条 54-turn 主链  + 14 条 subagent 链；总计 290 generations。
  - sample 44：1 条 69-turn 主链  + 1 条 13-turn subagent 链；总计 82 generations。

- 分析：
  - Qwen3.5-9 在 Claude-Code 下的 agent orchestration 能力不足，导致 sample 42 拉起 14 个 subagents。

- subagent 判定有四重证据互相对应：
  - 主链存在明确的 Agent tool call；
  - 调用包含 subagent_type、description 和 prompt；
  - 有实际 Agent 返回结果；
  - 同时生成一条带 file-search-specialist system prompt 的独立 trajectory。

## 运行命令

```bash
nohup env CUDA_VISIBLE_DEVICES=2,3,4,5 N_GPUS_PER_NODE=4 TP=2 ROLLOUT_GPU_MEM_UTIL=0.85 MAX_SAMPLES=50 MAX_CONCURRENT_SESSIONS=2 VLLM_LANGUAGE_MODEL_ONLY=1 MAX_NUM_BATCHED_TOKENS=8192 SAVE_TRAJECTORY_MESSAGES=1 bash examples/blackbox_recipes/claude_code/run_infer.sh > log_qwen35_9b.subagent.50.txt 2>&1 < /dev/null &
```
