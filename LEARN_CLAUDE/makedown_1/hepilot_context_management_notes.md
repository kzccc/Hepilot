# Hepilot Context Management Notes

## 一句话概括

Hepilot 的上下文管理不是简单地把历史对话拼进 prompt，而是围绕“模型下一步决策需要看到什么”做的一整套状态收集、分层、筛选、压缩、校验和恢复机制。

通俗说，上下文就是 agent 能看到的环境信息和状态。更准确地说，上下文是 agent 为了做出下一步决策而维护、筛选、压缩、恢复的一整套有效状态。

## 上下文包含哪些层面

| 序号 | 层面 | 含义 |
| --- | --- | --- |
| 1 | 规则上下文 | agent 的身份、行为规则、工具协议、权限边界 |
| 2 | 任务上下文 | 当前目标、当前阶段、已完成什么、接下来做什么、卡点是什么 |
| 3 | 工作区上下文 | 当前仓库状态、分支、文件结构、最近改动、项目文档、依赖环境 |
| 4 | 交互上下文 | 用户说过什么、agent 回过什么、工具调用过什么、结果是什么 |
| 5 | 工作记忆上下文 | 当前任务摘要、最近文件、文件摘要、短期过程笔记 |
| 6 | 长期记忆上下文 | 跨会话保留的偏好、项目约定、稳定决策、依赖事实 |
| 7 | 可信度上下文 | 哪些内容来自用户、工具、模型推断、外部文档，哪些是新鲜的，哪些可能过期 |
| 8 | 预算上下文 | 每个 section 分到多少预算、哪些信息被压缩过、哪些信息被舍弃了 |
| 9 | 恢复上下文 | checkpoint、resume 状态、关键文件、stale paths、恢复现场摘要 |

Hepilot 当前实现里，最核心的上下文管理入口是 `hepilot/context_manager.py`；checkpoint、resume、运行状态更新则在 `hepilot/runtime.py` 里完成。

## 上下文管理主要处理哪些问题

可以分成三大类。

第一类是 prompt 构建问题：

- 上下文收集：哪些上下文信息会作为 prompt 候选。
- 上下文分层：不同种类的信息应该放在哪一层。
- 上下文挑选：每一层如何有选择地进入 prompt。
- 上下文压缩：当 prompt 超预算时，按什么优先级压缩，压到什么程度。

第二类是状态更新和恢复问题：

- 内容层更新：模型读写文件、得到关键结论、尝试失败、用户确认后，哪些信息要写入 history / memory / durable memory。
- 控制层更新：尝试次数、工具步数、停止原因、运行环境、checkpoint 等推进状态要持续更新。
- 恢复表达：如果 agent 支持中断恢复和跨轮延续，需要决定哪些状态能被恢复，哪些状态已经 stale。

第三类是安全边界问题：

- 机密性安全：默认不信任原始输入，敏感信息和环境变量要在 trace/report 等上下文里脱敏。
- 完整一致性安全：上下文中最危险的不只是信息缺失，更是污染、过期或失真的信息，例如旧文件摘要误导模型。

## Prompt 候选内容

在每轮 prompt 构建的初始阶段，可能进入 prompt 的候选内容包括：

- agent 工作设定和固定规则
- 工具协议、工具列表和合法响应示例
- 工作区快照
- checkpoint 摘要
- 短期 memory 面板
- relevant memory 召回结果
- session history
- 当前用户问题
- 工具执行结果
- durable memory 主题信息

这些候选不会原样全量塞进 prompt，而是被分层、预算控制和压缩后再进入最终 prompt。

## Prompt 分层

Hepilot 当前把 prompt 分成 5 个固定 section：

```text
prefix
memory
relevant_memory
history
current_request
```

固定拼接顺序定义在 `SECTION_ORDER`：

```python
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
```

这个顺序的设计意图是：稳定规则和现场信息放前面，当前用户请求放最后，确保模型最后看到的是本轮最重要的输入。

## Prefix 层

prefix 是稳定前缀层，来自 `Hepilot.build_prefix()`。

它主要包含：

- agent 身份说明：告诉模型它是谁、处在什么环境里、应该以什么角色行动。
- 固定规则：约束模型这一轮只能按规定协议和行为边界做事。
- 工具列表：告诉模型当前有哪些工具可用、各自怎么调用、风险级别是什么。
- 合法响应示例：用样例固定输出格式，减少模型返回非法结构的概率。
- durable memory 写法说明：告诉模型如果用户要求“记住”，最终答案里应使用什么固定 label。
- 工作区快照：给模型当前仓库和运行环境的基础事实视图。

checkpoint 摘要也会被追加到 prefix 尾部：

```python
checkpoint_text = self.agent.render_checkpoint_text()
section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text
```

prefix 不是绝对不可压缩。它有初始预算和 floor，只有在 `relevant_memory`、`history`、`memory` 都压不够时，才会被最后压缩。

## Memory 层

memory 层来自 `agent.memory_text()`，最终由 `memory.render_memory_text(...)` 生成。

它是一个紧凑面板，不展开所有笔记正文。主要包含：

| 项 | 含义 |
| --- | --- |
| `task` | 当前任务摘要 |
| `recent_files` | 最近接触过的文件路径，最多 8 个 |
| `file_summaries` | 最近文件对应的短摘要，最多展示 6 个 |
| `episodic_notes` | 当前短期笔记数量，不直接展开正文 |
| `durable_topics` | 当前长期记忆主题名，不直接展开正文 |

`file_summaries` 的渲染逻辑比较关键：

1. 先遍历 `working.recent_files` 的前 6 个文件。
2. 用文件路径去 `file_summaries` 里查摘要。
3. 重新计算当前文件 freshness。
4. 只有摘要存在，并且摘要保存的 freshness 和当前文件 freshness 一致，才渲染进 `Memory:`。

这意味着旧文件摘要即使还在 state 里，只要 freshness 不匹配，也不会展示给模型。

memory section 自身也会被预算裁剪。普通 section 的渲染方式是 `_tail_clip(raw, budget)`，也就是按 section 当前预算从尾部截断。

## Relevant Memory 层

`relevant_memory` 是“按当前用户问题召回的相关记忆”。

它包含两类候选：

- 短期记忆里的 `episodic_notes`
- 长期记忆里的 durable notes

召回入口：

```python
selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=3)
```

### 召回依据

当前实现先对用户问题分词，得到 query tokens。然后每条 note 也会根据正文和 tags 生成 tokens。

排序分数是一个四元组，按优先级从高到低比较：

| 排序项 | 含义 |
| --- | --- |
| 精确标签匹配 | query token 是否命中 note tags |
| 关键词重叠度 | query token 与 note 正文/tags token 的重叠数量 |
| 新鲜时效性 | note 创建时间越新越靠前 |
| 来源优先级 | episodic note 的 tie breaker 高于 durable note |

这是一套轻量规则召回，不是语义检索。

### 渲染和预算

`relevant_memory` 最多召回 3 条 note。

渲染时会：

1. 先组成：

```text
Relevant memory:
- note 1
- note 2
```

2. 根据 section 预算和 note 数量计算每条 note 的平均预算。
3. 对每条 note 做 `_tail_clip(...)`。
4. 如果逐条裁剪后整段仍然超预算，就兜底裁剪整段。

metadata 会记录 selected notes 和实际 rendered notes，因此后续 trace/report 能解释“召回了什么”和“最终展示了什么”。

## History 层

history 层来自 `session["history"]`，由 `ContextManager._render_history_section(...)` 压缩后进入 prompt。

处理过程分三步。

### 1. 原始文本构建

先把 session history 转成完整 raw transcript：

- tool 事件渲染成工具名、参数、可选 `parent_tool_use_id` 和完整 content。
- user / assistant 消息渲染成 `[role] content`。

这份 raw 用于统计和兜底，但最终不一定全量进入 prompt。

### 2. 区分近期历史和旧历史

当前实现把最近 6 条 history 视为近期历史：

```python
recent_window = 6
recent_start = max(0, len(history) - recent_window)
```

近期历史：

- 尽量保留完整信息。
- 每条按 900 字符限制渲染。

旧历史：

- 旧 `read_file`：如果同一路径已经处理过，折叠重复读取。
- 旧 `read_file`：如果 memory 中有可复用 file summary，则渲染为 `path -> summary`。
- 其他旧 tool：压成单行摘要。
- 旧 `run_shell`：保留 command 和输出前 3 行非空内容。
- 旧 user / assistant：按 60 字符裁剪成一行。

### 3. 从近到远装入预算

压缩后的 history entries 会从最近往最远尝试加入 prompt。

每加入一条就检查是否超过 history section 的预算：

- 如果没超，保留。
- 如果超了，近期历史会根据剩余空间再次裁剪，尽量保留。
- 旧历史会更激进，每行压到 20 字符后再试。
- 仍然放不下就丢弃这条。

最后如果整段仍然超过预算，会用 raw transcript 做兜底裁剪。

## Current Request 层

`current_request` 是当前用户请求：

```text
Current user request:
<user_message>
```

它不参与普通 section 预算裁剪：

```python
budget_chars: None
```

原因很直接：这是本轮最重要的输入，不能为了保旧上下文而裁掉当前用户问题。

## SectionRender 数据结构

每个 prompt section 在组装阶段都会封装成 `SectionRender`：

```python
@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `raw` | 这个 section 的原始完整文本 |
| `budget` | 这个 section 当前分配到的字符预算 |
| `rendered` | 最终实际进入 prompt 的文本，可能已经被裁剪 |
| `details` | 附加元数据，不同 section 放不同内容 |

这个结构让上下文管理不只是“拼字符串”，而是能记录每层原始长度、预算、实际渲染长度和压缩细节。

## 预算控制

默认总预算：

```python
DEFAULT_TOTAL_BUDGET = 12000
```

默认 section 初始预算：

| section | 初始预算 |
| --- | --- |
| `prefix` | 3600 |
| `memory` | 1600 |
| `relevant_memory` | 1200 |
| `history` | 5200 |

默认 section floor：

| section | 最小预算 |
| --- | --- |
| `prefix` | 1200 |
| `memory` | 400 |
| `relevant_memory` | 300 |
| `history` | 1500 |

压缩优先级：

```python
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
```

也就是说，超预算时优先牺牲：

1. `relevant_memory`
2. `history`
3. `memory`
4. `prefix`

`current_request` 不在压缩列表里。

## 外层循环压缩过程

prompt 初次渲染后，如果 `len(prompt) > total_budget`，就进入循环。

每轮流程：

1. 计算当前超额：

```python
overflow = len(prompt) - self.total_budget
```

2. 按 `reduction_order` 找第一个还能压缩的 section。
3. 把这个 section 的预算从 `current_budget` 下调到：

```python
new_budget = max(floor, current_budget - overflow)
```

4. 记录一条 `budget_reductions` 日志。
5. 用新预算重新渲染所有 section。
6. 重新拼 prompt。
7. 如果还超预算，继续下一轮。

注意：每次成功压缩一个 section 后，下一轮循环会重新从 `relevant_memory` 开始尝试。因此同一个 section 可能被压缩多次，直到压到 floor 后才轮到后面的 section。

## Prompt Metadata

每轮 prompt 构建后会返回 metadata，后续进入 trace/report。

顶层指标：

| 路径 | 含义 |
| --- | --- |
| `prompt_chars` | 最终 prompt 字符数 |
| `prompt_budget_chars` | prompt 总预算 |
| `prompt_over_budget` | 最终是否仍然超预算 |
| `section_order` | section 固定拼接顺序 |
| `section_budgets` | 每个 section 最终预算 |
| `sections` | 各 section 渲染统计 |
| `budget_reductions` | 本轮压缩过程日志 |
| `reduction_order` | 实际采用的压缩优先级 |

每个 section 的统计：

| 路径 | 含义 |
| --- | --- |
| `sections.<section>.raw_chars` | 原始文本字符数 |
| `sections.<section>.budget_chars` | 实际采用预算 |
| `sections.<section>.rendered_chars` | 最终渲染字符数 |

`relevant_memory` 专项统计：

| 路径 | 含义 |
| --- | --- |
| `relevant_memory.limit` | 召回上限 |
| `relevant_memory.selected_count` | 实际选中的 note 数 |
| `relevant_memory.selected_notes` | 被选中的 note 正文 |
| `relevant_memory.selected_sources` | note 来源 |
| `relevant_memory.selected_kinds` | note 类型，如 episodic / durable |
| `relevant_memory.selected_durable_count` | durable note 数量 |
| `relevant_memory.raw_chars` | 原始字符数 |
| `relevant_memory.rendered_chars` | 最终字符数 |
| `relevant_memory.rendered_notes` | 实际渲染进 prompt 的 note |
| `relevant_memory.rendered_count` | 实际渲染 note 数 |

`history` 专项统计：

| 路径 | 含义 |
| --- | --- |
| `history.raw_chars` | history 原始字符数 |
| `history.rendered_chars` | history 最终字符数 |
| `history.older_entries_count` | 被作为旧历史处理的条目数 |
| `history.collapsed_duplicate_reads` | 被折叠的重复 read_file 数 |
| `history.reused_file_summary_count` | 旧 history 复用 file summary 次数 |
| `history.summarized_tool_count` | 被压成摘要的旧工具记录数 |

`current_request` 专项统计：

| 路径 | 含义 |
| --- | --- |
| `current_request.text` | 用户本轮原始输入 |
| `current_request.raw_chars` | 原始输入字符数 |
| `current_request.rendered_chars` | 当前请求正文字符数 |
| `current_request.section_chars` | 整个 current request section 字符数 |

runtime 还会额外补充一组 prompt metadata，例如：

- `prefix_chars`
- `workspace_chars`
- `memory_chars`
- `history_chars`
- `request_chars`
- `tool_count`
- `workspace_docs`
- `recent_commits`
- `prefix_hash`
- `prompt_cache_key`
- `workspace_fingerprint`
- `tool_signature`
- `workspace_changed`
- `prefix_changed`
- `prompt_cache_supported`
- `resume_status`
- `stale_summary_invalidations`
- `stale_paths`
- `runtime_identity_mismatch_fields`
- 自动识别到的敏感环境变量摘要

## Agent 推进中需要更新的上下文状态

可以分成内容层和控制层。

### 内容层更新

内容层回答的是：模型现在知道了什么。

主要包括：

- `history`：用户消息、assistant 工具意图、工具结果、重试提示、最终回答、失败收口。
- working memory：`task_summary`、`recent_files`、`file_summaries`。
- episodic notes：读取文件摘要、工具异常或拒绝形成的 process note。
- durable memory：用户明确要求记住时，从最终回答中抽取稳定事实。

工具执行后，`update_memory_after_tool(...)` 会更新短期记忆：

- `read_file` / `write_file` / `patch_file` 更新 `recent_files`。
- `read_file` 生成 `file_summary` 和一条 episodic note。
- `write_file` / `patch_file` 让旧 file summary 失效。

### 控制层更新

控制层回答的是：agent 推进到哪一步，能不能恢复。

主要包括：

- 当前目标
- 当前阶段或 checkpoint 节点
- 尝试次数 `attempts`
- 工具步数 `tool_steps`
- 是否完成
- 停止原因
- 下一步行动建议
- 关键文件列表
- 运行时身份配置
- resume 状态

这些控制层状态和 checkpoint 高度绑定。很多“任务状态的可恢复表达”不是靠模型自己记住，而是 runtime 生成 checkpoint 后，在下一轮 prompt 里通过 `render_checkpoint_text()` 展示给模型。

## 为什么需要 Checkpoint

模型本身没有可靠的持续状态管理能力。它可以看到 history，但 history 只说明“发生过什么”，不一定稳定表达“当前目标、卡点、下一步是什么”。

checkpoint 的作用是把任务推进状态结构化保存下来。

它带来三个实际收益：

1. 让模型知道当前阶段：当前目标、已完成、卡点、下一步都被提炼出来。
2. 支持中断恢复：进程退出或长任务中断后，可以从 checkpoint 摘要继续。
3. 防止上下文漂移：长 history 噪声越来越多，checkpoint 重新定锚关键状态。

## Checkpoint 包含哪些信息

当前 checkpoint 对象结构：

```python
checkpoint = {
    "checkpoint_id": checkpoint_id,
    "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
    "schema_version": CHECKPOINT_SCHEMA_VERSION,
    "created_at": now(),
    "current_goal": str(user_message),
    "completed": [task_state.final_answer] if task_state.final_answer else [],
    "excluded": [],
    "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
    "next_step": self.infer_next_step(task_state),
    "key_files": key_files,
    "freshness": freshness,
    "summary": f"{trigger}: {clip(str(user_message), 120)}",
    "runtime_identity": self.current_runtime_identity(),
}
```

字段解释：

| 字段 | 含义 |
| --- | --- |
| `checkpoint_id` | 当前 checkpoint 唯一 id |
| `parent_checkpoint_id` | 上一个 checkpoint id，用于串成演进链 |
| `schema_version` | checkpoint 结构版本 |
| `created_at` | 创建时间 |
| `current_goal` | 当前用户请求 |
| `summary` | 触发原因 + 用户请求摘要 |
| `completed` | 已完成结果；当前通常为空或 `[final_answer]` |
| `current_blocker` | 当前阻塞原因，本质来自 `task_state.stop_reason` |
| `next_step` | 系统推断的下一步 |
| `excluded` | 已排除方向；当前默认空列表 |
| `key_files` | 关键文件列表，每项包含 path 和 freshness |
| `freshness` | 以 path 为 key 的 freshness 映射 |
| `runtime_identity` | 创建 checkpoint 时的运行时身份 |

`key_files` 每项结构：

| 字段 | 含义 |
| --- | --- |
| `path` | 关键文件路径 |
| `freshness` | checkpoint 创建时该文件内容指纹 |

`next_step` 当前由 `infer_next_step()` 推断，只有几种固定形态：

- `No next step recorded.`
- `Resume from the latest checkpoint and continue the task.`
- `Decide the next action after <last_tool>.`
- `Continue the task from the latest checkpoint.`

## 创建 Checkpoint 的过程

`create_checkpoint(task_state, user_message, trigger)` 的参数：

| 参数 | 含义 |
| --- | --- |
| `task_state` | 当前这轮运行的任务状态 |
| `user_message` | 当前用户请求原文 |
| `trigger` | 这次创建 checkpoint 的触发原因 |

创建过程：

1. 从 session 里拿出 checkpoint 容器和当前活跃 checkpoint。
2. 生成新的 `checkpoint_id`。
3. 从 `memory.working.recent_files` 取最近关键文件。
4. 重新计算每个关键文件的 freshness。
5. 组装 checkpoint 对象。
6. 写入 `session["checkpoints"]["items"][checkpoint_id]`。
7. 更新 `session["checkpoints"]["current_id"]`。
8. 同步 `task_state.checkpoint_id`。
9. 同步 `session["runtime_identity"]`。
10. 保存整份 session。

## 创建 Checkpoint 的时机

当前实际 trigger 有这些：

| trigger | 时机 |
| --- | --- |
| `freshness_mismatch` | prompt 构建前发现 resume 状态是 `partial-stale` |
| `workspace_mismatch` | prompt 构建前发现 runtime identity 不匹配 |
| `context_reduction` | 本轮 prompt 触发了预算压缩 |
| `tool_executed` | 每个工具真实执行完成后 |
| `run_finished` | 本轮任务正常完成并返回 final answer |
| `task_state.stop_reason or "run_stopped"` | 主循环因为停止条件退出 |

不同创建时机下的典型状态：

| checkpoint 时机 | `completed` | `current_blocker` | `next_step` |
| --- | --- | --- | --- |
| `freshness_mismatch` | `[]` | `""` | 有 last_tool 则 `Decide the next action after <last_tool>.`，否则 `Continue the task from the latest checkpoint.` |
| `workspace_mismatch` | `[]` | `""` | 有 last_tool 则 `Decide...`，否则 `Continue...` |
| `context_reduction` | `[]` | `""` | 有 last_tool 则 `Decide...`，否则 `Continue...` |
| `tool_executed` | `[]` | `""` | `Decide the next action after <last_tool>.` |
| `run_finished` | `[final_answer]` | `""` | `No next step recorded.` |
| stopped | `[final]` | `stop_reason` | step limit 时 `Resume...`；否则按 last_tool 决定 `Decide...` 或 `Continue...` |

## Resume State

每次创建/恢复实例时，以及每轮组 prompt 前，runtime 都会执行 `evaluate_resume_state()`。

即使是新 session，也会执行；只是没有 checkpoint 时通常得到：

```python
status = "no-checkpoint"
stale_paths = []
runtime_identity_mismatch_fields = []
```

resume state 主要检查两类东西。

第一类是关键文件 freshness：

- 先调用 `invalidate_stale_memory()` 清理过期 file summaries。
- 再检查当前 checkpoint 的 `key_files`。
- 对每个关键文件重新计算 freshness。
- 如果和 checkpoint 记录的不一致，就加入 `stale_paths`。

第二类是运行时身份：

会比较当前运行环境和 checkpoint 里的 `runtime_identity`，包括：

- `cwd`
- `model`
- `model_client`
- `approval_policy`
- `read_only`
- `max_steps`
- `max_new_tokens`
- `feature_flags`
- `shell_env_allowlist`
- `workspace_fingerprint`
- `tool_signature`

最终状态：

| 状态 | 含义 |
| --- | --- |
| `no-checkpoint` | 没有 checkpoint |
| `schema-mismatch` | checkpoint 结构版本不兼容 |
| `full-valid` | 关键文件和运行环境都匹配 |
| `partial-stale` | 有关键文件或摘要过期 |
| `workspace-mismatch` | 运行环境身份不匹配 |

这个结果会写入：

```python
session["resume_state"] = resume_state
session["runtime_identity"] = self.current_runtime_identity()
```

prompt 构建时，checkpoint 摘要会被追加到 prefix，让模型看到恢复状态、当前目标、卡点、下一步、关键文件、stale paths 和 summary。

## 上下文安全边界

### 机密性安全

Hepilot 对 trace/report 等结构化工件做脱敏：

- 根据环境变量名识别疑似敏感项，如 `API_KEY`、`TOKEN`、`SECRET`、`PASSWORD`。
- 对显式配置的 secret env 做脱敏。
- `redact_artifact(...)` 会递归处理 dict/list/tuple/string。

注意：这主要保护 trace/report 等落盘工件。prompt 构建层本身仍应避免主动把敏感内容写进上下文。

### 完整一致性安全

上下文污染比上下文缺失更危险。

当前系统主要靠这些机制降低污染：

- file summary 带 freshness，文件变化后会失效。
- checkpoint key files 也带 freshness，用于判断恢复现场是否 stale。
- runtime identity 比较，避免模型、工作目录、工具签名、审批策略变化后误用旧现场。
- history 压缩时优先保留近期历史，旧 read_file 尽量复用新鲜摘要。
- 工具结果带 parent tool use id，保留 assistant tool call 与 tool result 的因果链。

## 当前边界和后续方向

当前 ContextManager 是规则驱动的上下文管理器，不是语义上下文系统。

它已经解决了：

- prompt 分层
- section 预算
- 超预算压缩
- history 压缩
- relevant memory 轻量召回
- checkpoint resume 展示
- prompt metadata 观测

后续可以继续增强：

- 更强的语义召回，不只靠关键词和 tags。
- 更结构化的任务状态板，不只靠 checkpoint summary。
- 更精细的上下文可信度标注，例如用户事实、工具事实、模型推断分开展示。
- 多 agent / delegate 结果写回 episodic note。
- 更成熟的 history compact 协议，把 assistant intent、tool result、file summary 和 checkpoint 串成更稳定的长会话恢复链。

