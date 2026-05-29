# Claude Code 中，Hepilot 最该先学的设计

## 文档目标

这份总结只回答一个问题：

**如果 `Hepilot` 要向 Claude Code 学习，并继续做成更有校招含金量的项目，最应该优先学什么。**

结论基于三类材料：

- 本地 Claude Code 源码
- 本地 `Hepilot` 源码
- `learn-claude-code` skill 指向的辅助学习资料

其中以**本地源码事实**为准，外部资料只用于辅助理解阅读顺序与设计意图。

---

## 一、优先级最高的 5 个设计

### 1. 多 Agent 编排，不是只做单 Agent 循环

#### Claude Code 里是怎么做的

Claude Code 有明确的 `Coordinator` 设计。主 agent 不直接改代码，而是负责：

- 拆任务
- 派 worker
- 给 worker 追加指令
- 综合 worker 结果

它把复杂任务拆成四阶段：

- `Research`
- `Synthesis`
- `Implementation`
- `Verification`

这不是“能开子 agent”这么简单，而是已经形成了**角色分层 + 阶段分工 + 结果回传**的任务系统。

#### 在源码里的位置

- `Claude-Code/docs/04-coordinator.md`
- `Claude-Code/src/tasks/LocalAgentTask/LocalAgentTask.tsx`

#### 对 Hepilot 的启发

这是 `Hepilot` 最该先补的一层。

你现在的 `delegate` 已经有了子 agent 雏形，但本质上还是“只读调查助手”。如果想让项目含金量明显上升，下一步不是继续加工具，而是先把 `Hepilot` 提升为：

- 主 agent 负责拆解与综合
- 子 agent 负责研究、实现、验证
- 子 agent 返回结构化结果，而不是一段自由文本

这会让 `Hepilot` 从“会调用工具的 agent”升级成“会组织工作的 agent”。

---

### 2. 工具层是调度系统，不是函数调用列表

#### Claude Code 里是怎么做的

Claude Code 的工具执行不是“模型点一个工具，程序就跑一下”。

它额外处理了几件很关键的事：

- 工具是否允许并发
- 只读工具如何批量并发执行
- 写操作如何串行执行
- 工具执行中的进度更新
- 某个工具失败后如何取消同批兄弟任务

这说明 Claude Code 把工具层当成了一个**执行调度器**，不是简单的 handler 集合。

#### 在源码里的位置

- `Claude-Code/src/services/tools/toolOrchestration.ts`
- `Claude-Code/src/services/tools/StreamingToolExecutor.ts`
- `Claude-Code/src/Tool.ts`

#### 对 Hepilot 的启发

`Hepilot` 现在的工具边界已经不错：

- 白名单明确
- 参数校验明确
- 风险工具有区分

但它还缺“调度层”。

最值得先补的是：

- 给工具增加 `is_read_only`、`is_concurrency_safe` 这类执行属性
- 把工具执行从 `runtime` 中拆成独立 `executor`
- 增加基础进度事件：`queued / running / completed / failed`

这类改造很工程化，也很适合面试讲清楚。

---

### 3. 会话内核独立，控制循环不是一锅粥

#### Claude Code 里是怎么做的

Claude Code 有一个很清晰的 `QueryEngine`，负责一整段会话的生命周期。它统一承接：

- 消息状态
- prompt 组装
- tool use
- session 持久化
- 权限拒绝记录
- resume 相关逻辑

也就是说，它不是“发一次模型请求”的函数，而是**会话内核**。

#### 在源码里的位置

- `Claude-Code/src/QueryEngine.ts`
- `Claude-Code/src/utils/queryContext.ts`
- `Claude-Code/src/utils/sessionStorage.ts`

#### 对 Hepilot 的启发

`pico/runtime.py` 已经承担了很多核心职责，这说明方向是对的；但如果后面继续堆多 agent、任务状态、异步执行，`runtime.py` 会越来越难扩展。

所以 `Hepilot` 最该学的不是 QueryEngine 的表面写法，而是它背后的拆分原则：

- 会话循环单独成核
- prompt 组装独立
- tool execution 独立
- resume / persistence 独立

这会直接决定后面 `Hepilot` 能不能继续长成一个像样的系统。

---

### 4. Prompt 是分层组装的，不是一大段固定前缀

#### Claude Code 里是怎么做的

Claude Code 的 prompt 不是单块字符串，而是多层上下文拼装结果。源码里能看到至少这些来源：

- 默认 system prompt
- user context
- system context
- coordinator 额外上下文
- memory 相关补充
- append system prompt

它的重点不是“写很长的 prompt”，而是**上下文来源明确、职责明确**。

#### 在源码里的位置

- `Claude-Code/src/utils/queryContext.ts`
- `Claude-Code/src/QueryEngine.ts`

#### 对 Hepilot 的启发

`Hepilot` 现在已经有：

- `prefix`
- `context_manager`
- `memory`
- `workspace fingerprint`

这说明基础并不差。

但下一步最值得学的，不是继续写更复杂的 prefix，而是把上下文正式分层，例如：

- agent 身份层
- workspace 层
- task 层
- memory 层
- execution 层

这样 `Hepilot` 的上下文工程会从“控字数”提升到“控语义结构”。

---

### 5. 恢复系统关注长会话正确性，不只是能保存

#### Claude Code 里是怎么做的

Claude Code 对 session persistence 很重视，但重点不是“把数据写盘”，而是：

- transcript 如何组织
- 长会话如何 compact
- compact 后如何 resume
- 子 agent 的 transcript 如何管理

这说明它把恢复问题看成**长期运行正确性问题**，不是单纯的持久化问题。

#### 在源码里的位置

- `Claude-Code/src/utils/sessionStorage.ts`
- `Claude-Code/src/QueryEngine.ts`

#### 对 Hepilot 的启发

`Hepilot` 在这方面已经比一般练手项目强：

- 有 `SessionStore`
- 有 `RunStore`
- 有 `checkpoint`
- 有 `resume_state`

但它目前更像“短会话恢复”，还不是“长会话治理”。

如果只选一个重点补，建议优先补：

- 把 history 升级为更清晰的消息日志模型
- 为未来 compact 做边界设计
- 给子 agent 预留独立 transcript 能力

这会让 `Hepilot` 的系统感明显更强。

---

## 二、为什么是这 5 个，而不是别的

### 因为它们最能提高项目含金量

这 5 个设计的共同点是：它们都不是“小功能”，而是**系统级能力**。

它们体现的是：

- 任务编排能力
- 执行调度能力
- 会话架构能力
- 上下文工程能力
- 长程运行能力

这些比“多接几个模型后端”或“再加几个工具”更能体现工程水平。

### 因为 Hepilot 已经有不错的底子

`Hepilot` 不是从零开始。现在已经有：

- session
- memory
- checkpoint
- prompt budget
- approval
- trace / report
- 受限 delegate

所以你现在最需要的不是继续补“基础设施碎片”，而是往上补**编排层、调度层、内核层**。

---

## 三、当前不建议优先学的部分

### 1. Ultraplan / 远程传送

这些能力很强，但更偏平台扩展，不是 `Hepilot` 当前最缺的核心层。

### 2. Buddy、Bridge、远程通道

这些更偏产品外围能力，对校招项目第一阶段加分不如任务编排与运行时架构。

### 3. 大而全的 MCP 生态

MCP 以后值得做，但前提是你的本地 agent 内核先足够稳。

---

## 四、最建议的学习顺序

### 第一阶段

先学：

1. `Coordinator`
2. `tool orchestration`
3. `QueryEngine`

这是最值得先读透的三块。

### 第二阶段

再学：

1. `query context`
2. `session storage`
3. `compact / resume` 相关逻辑

### 第三阶段

最后再看：

1. 更外围的桥接能力
2. 远程能力
3. 扩展生态能力

---

## 最后的判断

如果只用一句话概括 Claude Code 最值得 `Hepilot` 先学的地方，那就是：

**它最强的不是“工具多”，而是把 agent 做成了一个可编排、可恢复、可扩展的工程系统。**

对 `Hepilot` 来说，当前最该先学的亮点设计依次是：

1. 多 Agent 编排
2. 工具执行调度
3. 会话内核拆分
4. 分层上下文组装
5. 长会话恢复体系

这 5 个点，正是最容易把 `Hepilot` 从“能跑的本地 agent”提升成“有明显架构含金量的校招项目”的地方。
