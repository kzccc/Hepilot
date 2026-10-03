# Hepilot Memory Architecture Notes

## 一句话概括

Hepilot 的记忆不是一个单体模块，而是三层协作：

- `history`：完整事件流，保存 user / assistant / tool 的往来，负责“不丢”。
- `memory`：session 内短期工作记忆，保存下一轮最可能用到的浓缩信息，负责“不撑爆 prompt”。
- `durable memory`：跨 session 的长期记忆，落盘到 `.hepilot/memory`，负责“沉淀稳定事实”。

换句话说：`history` 保完整回放，working memory 保当前任务工作集，durable memory 保跨会话的稳定知识。

## Session 的角色

`session` 是一次 agent 会话的总状态容器。它不是单纯保存对话文本，而是把这次会话后续继续运行、恢复、复盘所需要的状态集中在一起。

当前 session 的关键字段包括：

| 字段 | 作用 |
| --- | --- |
| `id` / `created_at` | 标识这次会话是谁、何时创建 |
| `workspace_root` | 绑定当前会话对应的工作区 |
| `history` | 保存这次会话里的完整 transcript |
| `memory` | 保存 working / episodic / file summary 状态 |
| `checkpoints` | 保存可恢复的 checkpoint 集合 |
| `runtime_identity` | 记录模型、工作目录、工具签名、审批策略等运行身份 |
| `resume_state` | 记录 checkpoint / freshness / runtime identity 的恢复评估结果 |

短期记忆和 session 密不可分，因为短期记忆直接挂在 `session["memory"]` 上；完整事件流也挂在 `session["history"]` 上。没有 session，working memory、episodic memory 和 history 都没有稳定承载体，也没法跨轮续接。

## Session 落盘时机

`SessionStore` 负责把 session 写入 `.hepilot/sessions/<session_id>.json`。

主要落盘时机：

1. 新建 `Hepilot(...)` 或通过 `Hepilot.from_session(...)` 恢复旧会话时，`__init__()` 会补齐 session 结构、重建 memory、构建工具和 prefix、评估 `resume_state`，然后立刻保存一次 session。
2. 每次调用 `record(...)` 向 `history` 追加新事件时，都会同步保存 session。
3. 每次 `create_checkpoint(...)` 创建 checkpoint 时，会更新 `session["checkpoints"]` 和 `runtime_identity`，并保存 session。
4. 运行正常结束、异常收口或达到停止条件时，最终 assistant 记录、durable memory 晋升和 checkpoint 会跟随对应流程写回。
5. `resume_state`、memory 等状态变更通常不会单独立刻保存，而是跟随后续 `record(...)`、checkpoint 或初始化保存时落盘。

## Session 恢复

恢复流程大致是：

1. `SessionStore.load(session_id)` 读取旧 session JSON。
2. `Hepilot.from_session(...)` 把旧 session 传入 `Hepilot.__init__()`。
3. `_ensure_session_shape()` 补齐旧 session 缺失字段。
4. `LayeredMemory(...)` 用 `session["memory"]` 重建 memory 对象。
5. 重新构建工具、prefix、context manager。
6. `evaluate_resume_state()` 清理 stale file summary，并评估 checkpoint 是否仍然有效。
7. 保存一次规范化后的 session。

这样下一轮 `ask()` 可以基于旧的 `history + memory + checkpoints` 继续运行。

## 短期记忆

Hepilot 的短期记忆主要是 `session["memory"]` 里的 working memory、file summaries 和 episodic notes。

它不是完整回放历史，而是从会话过程中提炼出的轻量工作集：

| 组成 | 含义 |
| --- | --- |
| `working.task_summary` | 当前这轮用户任务的简短摘要 |
| `working.recent_files` | 最近实际读写或 patch 过、后续大概率继续关注的文件 |
| `file_summaries` | 按文件路径保存的短摘要缓存，带 freshness 指纹 |
| `episodic_notes` | 近期过程里值得提醒模型的短期笔记 |

短期记忆和 durable memory 的区别是：短期记忆跟着当前 session 走，服务“这几轮怎么继续做”；durable memory 跨 session 保存，服务“长期稳定事实”。

短期记忆和 history 的区别是：history 保存完整事件流；短期记忆保存“下一轮大概率会用到，但不需要带全文”的浓缩信息。

## `task_summary`

`task_summary` 的更新时间在每次 `ask(user_message)` 开始时。runtime 会先调用 `memory.set_task_summary(user_message)`，把当前用户请求写入 `working.task_summary`。

它的定位是“当前这轮正在做什么”，不是每个工具执行后自动改写，也不是模型在中途不断自我总结。

当前实现里会通过 `clip(..., 300)` 限制长度。

## `recent_files`

`recent_files` 只在工具真正执行后更新，不在用户发消息时更新，也不在模型刚提出工具计划时更新。

`update_memory_after_tool(...)` 会从工具参数里取 `path`，规范化成工作区内相对路径；只有工具名属于以下三类时才调用 `memory.remember_file(...)`：

- `read_file`
- `write_file`
- `patch_file`

因此，agent 实际读过、写过或 patch 过的文件才会进入 `recent_files`。

`recent_files` 还会参与 checkpoint：创建 checkpoint 时，runtime 会优先从 `memory.working.recent_files` 生成 `key_files`，并为这些文件记录 freshness 指纹。

## `file_summaries`

`file_summaries` 是“按文件组织的内容缓存”，回答的是：“这个文件上次读到的内容大概是什么？”

每条摘要包含：

| 字段 | 含义 |
| --- | --- |
| `summary` | 文件内容短摘要 |
| `created_at` | 摘要生成时间 |
| `freshness` | 摘要生成时文件内容的 sha256 指纹 |

写入新摘要的时机：

- `read_file` 工具实际执行完成后，runtime 调用 `summarize_read_result(...)` 生成短摘要，再写入 `file_summaries[path]`。

失效旧摘要的时机：

- `write_file` 或 `patch_file` 执行完成后，直接删除该文件旧摘要。
- 创建/恢复 Hepilot 实例时，`evaluate_resume_state()` 会调用 `invalidate_stale_memory()`，比较 freshness，不匹配则删除。
- 每轮正式组装 prompt 前，`_build_prompt_and_metadata()` 也会重新执行 `evaluate_resume_state()`，再次清理 stale summary。

`file_summaries` 还会帮助压缩旧 history：较旧的 `read_file` 结果如果已有可复用摘要，`ContextManager` 会把完整读取结果替换成 `path -> summary` 形式，并折叠同一路径的重复旧读取。

当前摘要生成方式比较轻量：`summarize_read_result()` 去掉空行，可跳过 Markdown 一级标题，只取前 3 行，用 ` | ` 拼接，再 clip 到 180 字符。

## `episodic_notes`

`episodic_notes` 是“按过程组织的短期笔记”，回答的是：“最近做过什么、发现过什么、踩过什么坑？”

主要写入时机：

1. `read_file` 执行完成后，runtime 会把读取结果压成短摘要，除了写入 `file_summaries`，还会调用 `memory.append_note(...)` 写一条 episodic note。
2. 工具执行出现异常态时，如果状态是 `partial_success`、`error` 或 `rejected`，runtime 会通过 `record_process_note_for_tool(...)` 写入一条 process note。

`file_summaries` 和 `episodic_notes` 的边界：

- `file_summaries` 强绑定文件路径，带 freshness，文件一变就应该失效，更像文件级缓存。
- `episodic_notes` 不一定绑定文件，可以绑定工具、异常、过程状态、读取结论，更像过程级笔记。

当前 note 追加时会 clip 到 500 字符，并保留最近 12 条。

未来可以扩展的 episodic 来源：

- 中间结论：已经验证出的阶段性认知、被证伪的假设、成立的技术判断。
- 工具发现：search 命中关键符号、shell 暴露关键报错、read_file 发现实现约束。
- 行动决策：下一步怎么推进、哪些方向已经排除。
- 会话级偏好与约束：只在当前 session 内有效、但会跨几轮影响行为的要求。
- 子任务摘要：delegate 或子 agent 返回的核心发现、未解决点、下一步建议。

## Relevant Memory 召回

`ContextManager` 每轮构建 prompt 时会调用 `memory.retrieval_candidates(user_message, limit=3)`。

召回候选来自两类：

- session 内的 `episodic_notes`
- `.hepilot/memory` 中的 durable notes

当前排序是轻量规则：

- tag 是否精确命中
- query token 和 note token 的重叠数量
- note 时间新旧
- episodic note 相对 durable note 有 tie breaker 优先级

它不是语义检索，因此对同义改写、隐含相关性和跨语言语义匹配的能力有限。

## 长期记忆

长期记忆是跨 session 的稳定事实存储，当前落盘位置是：

```text
.hepilot/memory/
├── MEMORY.md
└── topics/
    ├── project-conventions.md
    ├── key-decisions.md
    ├── dependency-facts.md
    └── user-preferences.md
```

`MEMORY.md` 是索引文件，只列出已经存在的 topic 元信息；具体笔记存在 `topics/*.md`。第一次 promote 某个 topic 时，只会初始化被写入的 topic，不会预先创建所有默认 topic。

默认 topic：

| topic | 含义 |
| --- | --- |
| `project-conventions` | 项目约定 |
| `key-decisions` | 关键决策 |
| `dependency-facts` | 依赖和环境事实 |
| `user-preferences` | 用户偏好 |

可扩展方向：

| topic | 含义 |
| --- | --- |
| `architecture-notes` | 架构笔记 |
| `onboarding` | 新人上手指南 |
| `known-bugs` | 已知问题和坑位 |
| `code-style` | 更细粒度代码风格 |
| `testing-rules` | 测试约定 |
| `deploy-facts` | 部署相关事实 |

## 长期记忆晋升流程

长期记忆不会从所有普通回答里无条件提取，而是在运行结束时尝试从本轮用户消息和最终回答里抽取。

当前入口是 `promote_durable_memory(user_message, final_answer)`：

1. `extract_durable_promotions(...)` 先判断用户消息是否包含显式长期记忆意图。
2. 如果没有“remember / save / persist / 记住 / 保存 / 记录 / 沉淀 / 长期记忆”等意图，直接跳过。
3. 如果有意图，则逐行扫描最终回答。
4. 只有命中预定义 label 的行才会被抽取。
5. 抽取出的正文进入 `reject_durable_reason(...)` 做质量过滤。
6. 通过过滤的候选进入 `memory.promote_durable(...)`，真正写入 `.hepilot/memory`。
7. 写入后重新 `self.session["memory"] = self.memory.to_dict()`，确保 session 里的 `durable_topics` 是最新状态。

可识别 label：

```text
Project convention: ...
Decision: ...
Dependency: ...
Preference: ...
项目约定：...
决策：...
依赖：...
偏好：...
```

质量过滤会拒绝：

- 空内容
- 已脱敏或疑似 secret 的文本
- checkpoint / current goal / next step 这类临时任务状态
- stdout / stderr / traceback / exit_code 等工具噪音
- 超过 220 字符的过长文本

去重策略：

- 完全相同的 note 跳过。
- 如果能提取出相同主语，则新 note 替换旧 note。
- 否则追加到对应 topic。

## History

`history` 是 session 的一部分，不是独立文件。session 落盘时，history 跟着一起写入 JSON；恢复会话时也会完整带回。

典型记录结构：

普通消息：

```json
{
  "role": "user",
  "content": "...",
  "created_at": "..."
}
```

工具结果：

```json
{
  "role": "tool",
  "name": "read_file",
  "args": {"path": "README.md"},
  "content": "...",
  "parent_tool_use_id": "...",
  "synthetic": false,
  "created_at": "..."
}
```

assistant 发起工具调用时，当前实现会先把模型原始输出写入 history：

```json
{
  "role": "assistant",
  "content": "<tool>...</tool>",
  "tool_uses": [
    {
      "id": "...",
      "name": "read_file",
      "args": {"path": "README.md"}
    }
  ],
  "created_at": "..."
}
```

这很重要：工具结果不再是孤立事件，下一轮 prompt 可以看到 `assistant tool call -> tool result` 的因果链。

## History 写入时机

主要写入点：

1. 用户开始一次新请求时，写入一条 `user` 消息。
2. 模型决定调用工具时，写入一条 `assistant` 消息，内容包含 raw 输出和 `tool_uses` 摘要。
3. 每个工具执行完成后，写入一条 `tool` 结果。
4. 批量工具里未执行的后续调用，会补写 `synthetic` tool result。
5. 模型输出格式不合法需要重试时，写入一条 `assistant` 重试提示。
6. 模型正常产出最终答案时，写入一条 `assistant` 最终回复。
7. 模型调用过程直接报错时，写入一条 `assistant` 失败收口说明。
8. 运行因步数上限或重试上限停止时，写入一条 `assistant` 停止说明。
9. 手动 reset 时，不是追加新记录，而是清空整份 history。

## 为什么 assistant 工具意图必须进 history

曾经的问题是：第一轮模型调用工具时，只保存了 tool result，没有保存 assistant 的工具调用意图。第二轮模型能看到一条孤立工具结果，却不知道上一轮为什么调用它，容易重复执行已完成工具，或者声称“没有相关历史信息”。

对多轮工具协作来说，assistant 的动作意图是一等事件。它保存的是推理因果链：

```text
用户请求 -> assistant 决定调用工具 -> tool result -> assistant 下一步判断
```

tradeoff 是上下文会更大；收益是多轮推理一致性、恢复可解释性和 trace/debug 能力更强。对 agent 场景来说，这是正确性基础，不只是可选优化。

## 当前边界

当前短期记忆更像轻量工作记忆，不是完整任务状态板。

主要边界：

1. 更擅长保留文件级和局部事实，例如文件摘要、读写过的文件、某次工具异常。
2. 召回逻辑是关键词/tag/时间等规则，不是语义检索。
3. assistant 普通自然语言回复不会系统性写回 memory，大量分析、判断和阶段性结论仍主要停留在 history。
4. 摘要生成整体偏轻量，很多地方是 clip 或简单提炼，不是模型级总结。

这也是当前版本合理的取舍：先把减少重复读取、跨轮复用、恢复正确性做稳，比一开始做成复杂黑盒更重要。

## 后续改进方向

可以按风险从低到高推进：

1. 扩展记忆单位：从单条文件摘要扩展到任务片段，记录“这个文件为什么和当前任务相关”“这次读取得出的结论是什么”。
2. 增加 memory write-back：只在明确结论、关键决策、用户确认、重要约束、文件判断出现时，把 assistant 普通回复抽取成结构化 note。
3. 做分层召回：保留当前关键词规则作为第一层，再增加可选语义召回作为第二层。
4. 引入子 agent 总结：用专门总结 agent 在 run 结束、checkpoint、delegate 返回时生成更高质量的 episodic note 或 durable 候选。
5. 强化 history compact：围绕 assistant tool intent、tool result、file summary 和 checkpoint 设计更明确的长会话压缩协议。

