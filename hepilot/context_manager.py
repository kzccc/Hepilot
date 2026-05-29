"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3600,
    "memory": 1600,
    "relevant_memory": 1200,
    "history": 5200,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "memory": 400,
    "relevant_memory": 300,
    "history": 1500,
}
# 当 prompt 超预算时，会优先压缩这些 section。
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 3


def _tail_clip(text, limit):
    """从尾部裁剪文本，保持 prompt section 不突破预算。"""
    text = str(text)  # 统一转成字符串，避免传入非字符串对象时无法计算长度
    if limit <= 0:
        # 没有可用预算时，直接返回空文本。
        return ""
    if len(text) <= limit:
        # 原文没有超预算时，保持原样。
        return text
    if limit <= 3:
        # 预算太小时连省略号都放不下，直接截取可容纳的前几个字符。
        return text[:limit]
    # 预算足够时，保留前 limit - 3 个字符，并用省略号标记尾部被截断。
    return text[: limit - 3] + "..."


@dataclass
class SectionRender:
    """单个 prompt section 的原文、预算和渲染结果。"""

    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    """按预算组装 prompt 的调度器。

    它把 prefix、memory、history 和当前请求压成一份受限大小的 prompt，
    同时保留 metadata，供 trace 与实验统计复盘。
    """

    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)  # 整个 prompt 允许的最大字符预算
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)  # 各 section 的初始字符预算
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()  # 各 section 被裁剪时必须保留的最低字符数
        
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)  # 超预算时依次尝试压缩的 section 顺序

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 工作记忆 +
        相关笔记 + 历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `Hepilot.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，`LayeredMemory`
        提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        #判断是否启用
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")

        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        
        #checkpoint_text是一些额外的文本信息，通常由 agent 的 render_checkpoint_text 方法提供，用于在 prompt 中显示一些当前状态或调试信息。
        #他是被当作perfix的一部分的,追加在perfix的尾部
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text

        #这里是处理召回的笔记
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)
        #这里是不处理预算的前提下，直接把所有文本渲染出来，主要用于调试和对比。开启后才会启用预算控制和压缩策略。
        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes=selected_notes)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata
        # 先拷贝一份各 section 的初始预算，后面裁剪时会在这份副本上调整
        budgets = dict(self.section_budgets)  
        # 按当前预算把各个 section 渲染成结构化的 SectionRender 结果
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)  
        # 将渲染后的各 section 按顺序拼接成最终 prompt 文本
        prompt = self._assemble_prompt(rendered)  
        # 记录后续每一步预算压缩的日志；初始为空，表示还没开始 reduction
        reduction_log = []  

        # 如果 prompt 超预算，就按固定顺序不断压缩。
        # 这里的顺序体现了平台偏好：
        # 先牺牲 relevant_memory，再牺牲 history，然后才动 memory 和 prefix。
        # 最新用户请求永远不裁剪，因为那是本轮最重要的输入。
        while len(prompt) > self.total_budget:
            #先计算出溢出的budget数量，也就是当前prompt的长度减去总预算的长度
            overflow = len(prompt) - self.total_budget
            #本轮是否成功压缩过 section 的标记位
            reduced = False
            #每一轮按照顺序压缩一个section，直到压缩成功或者所有section都尝试过了
            for section in self.reduction_order:
                #拿出当前这个section的最低预算
                floor = int(self.section_floors.get(section, 0))
                #拿出这个section的当前预算
                current_budget = int(budgets.get(section, 0))
                #如果当前预算比最低预算还要小,就跳过
                if current_budget <= floor:
                    continue
                #这一条的意思是当前先试着把预算调整到刚好能满足多出来的overflow的程度，也就是在当前预算的基础上减去overflow,但是不能低于floor,所以取max(floor, current_budget - overflow)
                #同时这里也就代表着如果没法一次就压缩满足多出来的overflow,那么调整到floor后,就代表还要继续尝试压缩下一个section
                new_budget = max(floor, current_budget - overflow)
                #防御性编程
                if new_budget >= current_budget:
                    continue
                #记录压缩的日志,包括被压缩的section,压缩前的预算,压缩后的预算,以及这次压缩节省的预算数量
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow,
                    }
                )
                #改变指定section的budget
                budgets[section] = new_budget
                #用新的budget重新渲染整个文本，
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    def _render_sections_without_reduction(self, section_texts, selected_notes=None):
        """关闭压缩策略时，直接输出各 section 的原始文本。"""
        #处理selected_notes的显示,如果没有selected_notes,则显示none
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)

        #处理history的显示,直接把所有历史记录原样输出出来
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._raw_history_text(history)
        
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=len(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=len(section_texts["memory"]), rendered=section_texts["memory"], details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw,
                budget=len(relevant_raw),
                rendered=relevant_raw,
                details={
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes),
                    "rendered_count": len(selected_notes),
                    "note_budget": 0,
                },
            ),
            "history": SectionRender(raw=history_raw, budget=len(history_raw), rendered=history_raw, details={"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _compute_section_floors(self):
        """为每个 section 计算最小可压缩下限。"""
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None):
        """按当前预算渲染每个 section。"""
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0))
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        """把相关记忆渲染成独立 section，并对每条 note 平均分配预算。"""
        header = "Relevant memory:"
        #将note中的text全部提出来,如果text为空或者全是空白字符,则替换成空字符串,最后过滤掉空字符串的note
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        #加上header构建初步的raw文本
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        #构建正式的raw文本,用\n连接起来形成一个字符串,如果没有note_texts,则显示"- none"
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        #如果没有note_texts,则返回raw，并且rendered和raw一样，details里selected_notes和rendered_notes都是空列表，selected_count和rendered_count都是0，note_budget也是0
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )
        #计算每条 note 的预算
        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉,得出来的rendered_notes每一条的长度都要小于等于per_note_budget
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            #重新组合成raw文本，每条note前面加上"- "，并且加上header
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            #判断是否已经满足预算要求,但是如果 per_note_budget 已经到了 1，说明：这已经是极限了，再缩也没有什么意义。
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            # 如果逐条裁剪后仍然超预算，就兜底按整段 raw 文本强制截断。
            rendered = _tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        """估算单条 note 的可用字符预算。"""
        #判断笔记的数量
        if note_count <= 0:
            return 0
        #除了正文之外,还要算上 header 和每条 note 前面的 "- " 这两个字符的占位，所以总的占位是 header 的长度加上每条 note 的占位乘以 note 的数量
        overhead = len(header) + 3 * note_count
        #usable 是 budget 减去 header 和每条 note 前面的 "- " 这两个字符的占位,可以真正分给正文的预算
        usable = max(0, budget - overhead)
        #计算出每条笔记的分配预算
        return max(1, usable // note_count)

    def _render_history_section(self, budget):
        """压缩历史记录，优先保留最近几轮和高价值摘要。"""
        history = list(getattr(self.agent, "session", {}).get("history", []))
        #返回一个处理好的多行字符串
        raw = self._raw_history_text(history)
        #如果没有历史记录,就可以直接返回一个空的SectionRender了
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_file_summary_count": 0,
                    "summarized_tool_count": 0,
                },
            )

        # 优先保留最近的历史，因为下一步决策通常最依赖刚刚发生的工具结果。
        recent_window = 6
        recent_start = max(0, len(history) - recent_window)
        #传入历史记录的str和recent_start的index,返回一个压缩过的历史记录列表和一些统计信息
        history_entries, history_details = self._compressed_history_entries(history, recent_start)
        rendered_entries = []
        for entry in reversed(history_entries):
            # 将刚才取回来的处理好 recent 窗口的历史记录先反转，让最近的在前面。
            # 然后逐条尝试加入 rendered_entries，同时检查是否超过预算。
            recent = bool(entry.get("recent", False))
            candidate_lines = list(entry.get("lines", []))
            # 这里采用的是每一条累加，每累加一次就转为字符串检查是否超过预算。
            # 如果超过预算了，就根据 recent 的不同采取不同策略压缩当前这条记录，而不是直接丢掉。
            # 因为前面已经反转，所以现在是从最近的历史记录开始往回加，保证预算有限时优先保留最近历史。
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            # 检查加入当前记录后是否超过预算。
            if len(candidate_rendered) <= budget:
                # 没超过的话就更新 rendered_entries，然后继续尝试加入下一条记录。
                rendered_entries = candidate_entries
                continue
            # 到了这一步，说明超出预算了；但这里不直接丢掉，而是先按记录新旧尝试压缩当前记录。
            if recent:
                # 这里是针对最近历史记录的策略。
                # 预算减去 "Transcript:" 这部分固定占位，再减去已经 rendered_entries 的长度占位，
                # 剩下的就是当前这条记录可以使用的预算。
                available = budget - len("Transcript:")
                if rendered_entries:
                    # 如果当前 rendered_entries 不为空，则需要计算它们已经占用的长度。
                    # 加一是因为每个 line 拼接后还会有一个换行符占位。
                    available -= sum(len(line) + 1 for line in rendered_entries)
                # 即使预算很紧，也尽量给 recent history 至少 20 个字符，让它还有一点可读性。
                # 里面 available - 1 是在给换行符/拼接开销留 1 个字符的余量。
                available = max(20, available - 1)
                # 算出当前还能使用的预算后，用这个预算裁剪当前这条历史记录的每一行。
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if len(candidate_rendered) <= budget:
                    # 将裁剪后的记录更新到正式的 rendered_entries。
                    # 如果这个判断还是没通过，就不会更新 rendered_entries，当前 history 会被放弃，循环继续处理下一条。
                    rendered_entries = candidate_entries
            else:
                # 这里是针对非最近历史记录的策略。
                # 旧记录价值较低，所以直接把每行裁剪到 20 个字符，再检查是否还能放进预算。
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if len(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
        # 最后返回形成完整的 rendered 字符串。
        rendered = "\n".join(["Transcript:", *rendered_entries])

        # 整个文本的兜底裁剪；理论上前面已经尽量控制了，但这里防止极端情况下仍然超预算。
        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "recent_window": recent_window,
                "recent_start": recent_start,
                "rendered_entries": rendered_entries,
                **history_details,
            },
        )

    def _compressed_history_entries(self, history, recent_start):
        """把旧历史压成摘要，把近期历史保留为更完整的记录。"""
        entries = []  # 收集最终保留下来、准备渲染进 history section 的历史条目
        seen_older_reads = set()  # 记录已经处理过的旧 read_file 路径，用来折叠重复读取
        details = {
            "older_entries_count": 0,  # 被归类为较早历史条目的数量
            "collapsed_duplicate_reads": 0,  # 被折叠掉的重复 read_file 结果数量
            "reused_file_summary_count": 0,  # 复用 file_summaries 里的文件摘要次数
            "summarized_tool_count": 0,  # 被压缩成摘要形式展示的工具结果数量
        }

        for index, item in enumerate(history):
            #要index>= recent_start才有效,
            recent = index >= recent_start
            #如果这是一条有效的历史记录
            if recent:
                #这行的限制为900
                line_limit = 900
                entries.append(
                    {
                        "recent": True,
                        "lines": self._render_history_item(item, line_limit),#返回的是截断的单条记录,且为列表形式
                    }
                )
                #这条加入了entries后,到这里就结束
                continue
            #如果现在这条不是最近的记录并且是工具类的read_file的记录
            if item["role"] == "tool" and item["name"] == "read_file":
                #将read_file工具调用的路径作为标识,
                path = str(item["args"].get("path", "")).strip()
                #如果之前已经处理过同样路径的read_file调用了,就认为这是重复读取,直接跳过不加入entries了
                if path in seen_older_reads:
                    details["collapsed_duplicate_reads"] += 1
                    continue
                seen_older_reads.add(path)
                #这个函数会去state将path对应的文件摘要拿出来
                summary = self._reusable_file_summary(path)
                if summary:
                    #如果成功取到了,就加入entries,格式是"path -> summary",并且更新统计信息
                    entries.append({"recent": False, "lines": [f"{path} -> {summary}"]})
                    details["older_entries_count"] += 1
                    details["reused_file_summary_count"] += 1
                    continue
            #处理其他的工具记录,不区分是否recent,都直接压成一行摘要,格式是"工具调用 -> 摘要内容",并且更新统计信息
            if item["role"] == "tool":
                summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line]})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue
            #其他的就是旧的re历史记录
            entries.append({"recent": False, "lines": self._render_history_item(item, 60)})

        return entries, details

    def _reusable_file_summary(self, path):
        """复用 memory 里已有的文件摘要，减少旧 read_file 记录占位。"""
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""
        snapshot = memory.to_dict()
        summary = snapshot.get("file_summaries", {}).get(str(path), {})
        if not summary:
            return ""
        return str(summary.get("summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        """为较旧的工具调用生成单行摘要。"""
        if item["name"] == "run_shell":
            # run_shell 的 content 往往是多行 stdout/stderr，直接塞回 prompt 会很占空间。
            # 这里先保留原始命令，后面再摘取前几行非空输出，压成一行 "command -> summary"。
            command = str(item["args"].get("command", "")).strip() or "shell"
            # splitlines() 按行拆输出；strip() 去掉空白；if line.strip() 过滤空行。
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            # 旧 shell 输出最多保留前三行，用 " | " 串起来；完全没有输出时标记为 "(empty)"。
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        # 非 shell 工具复用通用 history 渲染；旧历史只要一行摘要，所以取返回列表的第一行。
        return self._render_history_item(item, 60)[0]

    def _raw_history_text(self, history):
        """生成未经压缩的 history 文本。"""
        if not history:
            # 没有任何历史时，也返回一个稳定的 Transcript 占位文本。
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                # 工具历史保留工具名、参数和完整 content，作为未压缩 raw 文本。
                # tool history 在新协议里可能带 parent_tool_use_id；
                # 这里按“旧前缀 + 可选 id 后缀”渲染，确保老测试和新配对信息都能兼容。
                lines.append(self._render_tool_header(item))
                lines.append(str(item["content"]))
            else:
                # 普通 user/assistant 历史用 "[role] content" 的单行格式保留。
                lines.append(f"[{item['role']}] {item['content']}")
        # 最终统一加上 Transcript 标题，形成完整的原始 history 文本块。
        return "\n".join(["Transcript:", *lines])

    def _render_history_item(self, item, line_limit):
        """把单条 history item 渲染成 prompt 行。"""
        if item["role"] == "tool":
            # 工具调用分两行渲染：第一行是工具名和参数，方便知道当时调用了什么。
            prefix = self._render_tool_header(item)
            # 第二行是工具输出；输出可能很长，所以按 line_limit 做尾部裁剪，但至少保留 20 字符。
            content = _tail_clip(item["content"], max(20, line_limit))
            return [prefix, content]
        # 普通 user/assistant 消息不拆行，直接压成一行并按 line_limit 做尾部裁剪。
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]

    def _render_tool_header(self, item):
        """渲染 tool history 头部，并在存在时附加 parent_tool_use_id。"""
        prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
        parent_tool_use_id = str(item.get("parent_tool_use_id", "")).strip()
        if parent_tool_use_id:
            prefix += f" [parent_tool_use_id:{parent_tool_use_id}]"
        return prefix

    def _assemble_prompt(self, rendered):
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts):
        """汇总 prompt 组装过程的观测指标。"""
        section_metadata = {}  # 按 section 收集各自的渲染统计信息
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {  # 记录 prefix/memory/relevant_memory/history 的渲染统计
                "raw_chars": rendered[section].raw_chars,  # 该 section 原始文本的字符数
                "budget_chars": int(budgets.get(section, 0)),  # 该 section 本轮分到的预算字符数
                "rendered_chars": rendered[section].rendered_chars,  # 该 section 最终渲染后的字符数
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {  # 单独记录当前用户请求，不参与普通 section 预算体系
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),  # 当前请求原始文本字符数
            "budget_chars": None,  # current request 不设置独立预算
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),  # 当前请求最终渲染字符数
        }
        return {
            "prompt_chars": len(prompt),  # 最终 prompt 的总字符数
            "prompt_budget_chars": self.total_budget,  # 整个 prompt 允许的总字符预算
            "prompt_over_budget": len(prompt) > self.total_budget,  # 本轮 prompt 是否超出总预算
            "section_order": list(SECTION_ORDER),  # prompt 中各 section 的固定顺序
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))  # 每个 section 的最终预算
                for section in SECTION_ORDER
            },
            "sections": section_metadata,  # 各 section 的详细统计
            "budget_reductions": reduction_log,  # 预算压缩过程的记录
            "reduction_order": list(self.reduction_order),  # 实际采用的 section 压缩顺序
            "relevant_memory": {  # relevant_memory 召回与渲染的统计
                "limit": RELEVANT_MEMORY_LIMIT,  # 召回上限
                "selected_count": len(selected_notes),  # 实际选中的 note 数量
                "selected_notes": [note["text"] for note in selected_notes],  # 被选中的 note 文本
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],  # 被选中 note 的来源
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],  # 被选中 note 的类型
                "selected_durable_count": sum(  # 其中 durable note 的数量
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["relevant_memory"].raw_chars,  # relevant_memory 原始字符数
                "rendered_chars": rendered["relevant_memory"].rendered_chars,  # relevant_memory 最终字符数
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),  # 最终实际渲染出来的 note 文本
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),  # 最终渲染的 note 数量
            },
            "history": {  # history section 的压缩与渲染统计
                "raw_chars": rendered["history"].raw_chars,  # history 原始字符数
                "rendered_chars": rendered["history"].rendered_chars,  # history 最终字符数
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),  # 被压缩成旧摘要的条目数
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),  # 被折叠的重复 read_file 数
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),  # 复用 file_summary 的次数
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),  # 被压成摘要的工具调用数
            },
            "current_request": {  # 当前用户请求的统计信息
                "text": user_message,  # 用户原始输入文本
                "raw_chars": len(user_message),  # 原始输入字符数
                "rendered_chars": len(user_message),  # 当前请求最终渲染字符数
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),  # current_request section 里实际显示的字符数
            },
        }
