---
name: learn-claude-code
description: Use when the user asks to study Claude Code, Claude Code internals, design, workflows, source analysis, or how to adapt ideas from Claude Code into hepilot. Consult the local Claude-Code source checkout in LEARN_CLAUDE first, then the curated external sources in references/sources.md, and answer in a learning-oriented way focused on Hepilot improvements.
---

# Learn Claude Code

Use this skill when the user is learning Claude Code in order to improve `Hepilot`.

## Scope

Focus on these kinds of requests:
- Claude Code architecture, control loop, prompt system, tools, memory, safety, resume, or CLI behavior
- How Claude Code differs from `Hepilot`
- What parts of Claude Code are worth borrowing into `Hepilot`
- Source-guided explanations grounded in the local checkout and the curated references

## Source order

Use sources in this order:
1. The local checkout at `../Claude-Code/`
2. The curated references listed in `references/sources.md`
3. Only if needed, infer carefully from the local source and clearly label the inference

## Workflow

1. Read the user question and decide which Claude Code subsystem it maps to.
2. Search the local checkout in `../Claude-Code/` first. Prefer concrete files and code paths over summaries.
3. If local source is not enough, consult the relevant curated source from `references/sources.md`.
4. Explain the Claude Code behavior in Chinese unless the user asks otherwise.
5. When useful, map the finding back to `Hepilot` with a short section: `对 Hepilot 的启发`.
6. Distinguish clearly between:
   - Claude Code 当前实现里能直接看到的事实
   - 你基于源码结构做出的推断
   - 适合迁移到 `Hepilot` 的设计建议

## Search guidance

For local source work, start with targeted search instead of bulk reading.

Useful commands:
```bash
rg -n "keyword" ../Claude-Code
rg --files ../Claude-Code
```

Good starting areas to inspect:
- CLI / entry: look for top-level commands, bootstrap, config loading
- Agent loop: look for model call orchestration, tool execution, retries, stop conditions
- Prompting: look for system prompt, context assembly, truncation, cache hints
- Memory / state: look for session state, persistence, checkpoint, resume, summaries
- Safety: look for approvals, sandboxing, path validation, command gating

## References

Read `references/sources.md` for the curated external sources and when to use each one.

## Output style

Keep answers learning-oriented and grounded.

Preferred structure when the request is substantial:
- `Claude Code 里是怎么做的`
- `在源码里的位置`
- `对 Hepilot 的启发`

When the user asks a narrow code question, answer directly and skip the extra structure.
