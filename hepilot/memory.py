"""多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
from datetime import datetime
import re
from pathlib import Path

from .workspace import clip, now

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


def default_memory_state():
    """返回一份空白但结构完整的 memory state。"""
    # 用一个小而结构化的状态，而不是一大段自由文本摘要。
    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }


def _score_retrieval_note(query_tokens, note, extra_tokens=None, tie_breaker=0):
    """统一计算召回 note 的排序分数。"""
    # 计算 note_tags 的 token,是用于分数第一项精准匹配的
    note_tags = {tag.lower() for tag in note.get("tags", [])}
    #待匹配的note_token合计来自note_text和note_tags
    note_tokens = _tokenize(note.get("text", "")) | note_tags
    #如果有额外 token，就把它们并入 note_tokens 集合，作为这条 note 的额外匹配信号。
    if extra_tokens:
        note_tokens |= set(extra_tokens)
    #然后开始计算4个分数
    exact_tag_match = int(bool(query_tokens & note_tags))
    keyword_overlap = len(query_tokens & note_tokens)
    recency = _parse_timestamp(note.get("created_at"))
    #如果是来自episodic note,则优先级更高,tie_breaker会传入1；如果是durable note,则优先级更低tie_breaker会传入0
    return (exact_tag_match, keyword_overlap, recency, tie_breaker)


class DurableMemoryStore:
    """落盘保存跨 session 的 durable memory 主题与笔记。"""

    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"

    def topic_slugs(self):
        """返回 durable memory 里所有 topic slug。"""
        #可能就是长这样["project-conventions", "key-decisions", "user-preferences"]
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self):
        """从 `MEMORY.md` 读取 durable topic 索引。"""
        if not self.index_path.exists():
            return []
        #这里就是将MEMORY.md的每一行拆成一个列表
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            #用正则去判断 line 是否符合 MEMORY.md 里 topic 索引那种格式。
            #具体类似- [project-conventions](topics/project-conventions.md): Project Conventions 这样的会被选中
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                #正则表达式种()一个括号算一个group
                #那么match.group(1)就是上面例子里的project-conventions,match.group(2)就是Project Conventions
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            #将summary填入
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            #将tag填入
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
            '''
            最后 topics 的形式可能是这样：
            [
                {
                    "topic": "project-conventions",
                    "title": "Project Conventions",
                    "summary": "Stable repository conventions.",
                    "tags": ["convention"],
                },
                {
                    "topic": "key-decisions",
                    "title": "Key Decisions",
                    "summary": "Long-lived decisions and rationale anchors.",
                    "tags": ["decision"],
                },
            ]
            '''
        return topics

    def load_topic_notes(self, topic):
        """读取某个 durable topic 下的所有 note。"""
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        #先将文件切分成多行，然后读成列表
        lines = path.read_text(encoding="utf-8").splitlines()
        
        notes = []  # 用来收集当前 topic 文件里解析出来的所有 note
        capture = False  # 是否已经进入 "## Notes" 区域；只有进入后才开始真正收集 note
        updated_at = ""  # 记录这个 topic 文件里的更新时间；后面会作为 note 的 created_at 使用
        tags = []  # 记录这个 topic 的标签列表；后面每条 durable note 都会沿用这组 tags

        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

    @staticmethod
    def _subject_key(text):
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
        return None

    def ranked_retrieval_candidates(self, query, limit=3):
        """返回带统一排序分数的 durable note 候选，供外层直接复用。"""
        query_tokens = _tokenize(query)
        ranked = []
        for topic in self.load_index():
            notes = self.load_topic_notes(topic["topic"])
            topic_tokens = _tokenize(topic.get("title", ""))
            for note in notes:
                score = _score_retrieval_note(
                    query_tokens,
                    note,
                    extra_tokens=topic_tokens,
                    tie_breaker=0,
                )
                if score[0] == 0 and score[1] == 0:
                    continue
                ranked.append((score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:limit]

    def _write_index(self, topics):
        """写回 durable memory 索引文件。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        """写回某个 topic 的 markdown 文件。"""
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def promote(self, promotions):
        if not promotions:
            # 没有要写入的内容，直接返回空列表
            return [], []
        # 从 MEMORY.md 读出已有的 topic 索引，按 topic 名做 key
        topics = {topic["topic"]: topic for topic in self.load_index()}
        # 从 topics/*.md 读出每个 topic 下的已有笔记文本列表
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []
        # 逐条处理本次提取到的 promotion
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            # 如果此 topic 是第一次被写入，从默认值里取标题、摘要、tag 来初始化
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            # 确保该 topic 在 topic_notes 里也有条目（新 topic 初始化为空列表）
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                # 完全相同的文本已存在，跳过
                continue
            # 提取新笔记的主语，用于和旧笔记做"主题级别的去重"
            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text) == new_subject:
                        # 主语相同 → 替换旧笔记，不追加重复的
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                # 没有旧笔记被覆盖，作为新笔记追加
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")
        # 重建 MEMORY.md 索引文件
        self._write_index([topics[slug] for slug in sorted(topics)])
        # 重建每个 topic 文件（只写有笔记的 topic）
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        return results, superseded


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    """把相对路径解析到工作区内，并阻止路径逃逸。"""
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    """返回工作区内使用的规范路径字符串。"""
    # 例子：
    # - workspace_root="/repo"，raw_path="hepilot/runtime.py" -> "hepilot/runtime.py"
    # - workspace_root="/repo"，raw_path="/repo/hepilot/runtime.py" -> "hepilot/runtime.py"
    # - 如果路径不在工作区内，就退回成原始路径的 posix 字符串形式。
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    """为文件内容计算 freshness 指纹，用于摘要失效判断。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _tokenize(text):
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def _parse_timestamp(value):
    #把 ISO 格式时间解析成可比较的 Unix 时间戳(float)；如果为空或解析失败，就退回 0.0
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _normalize_note(note, index):
    if isinstance(note, str):
        text = clip(note.strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    if not isinstance(note, dict):
        text = clip(str(note).strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    text = clip(str(note.get("text", "")).strip(), 500)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"
    return {
        "text": text,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }


def normalize_memory_state(state, workspace_root=None):
    """把旧格式或不完整的 memory state 规整成当前标准结构。"""
    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = Path(workspace_root) / ".hepilot" / "memory" if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    """更新当前任务摘要。"""
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    """把文件加入最近工作文件列表。"""
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    #写一份新的files,不包括那个path,相当于抛弃他
    files = [item for item in state["working"]["recent_files"] if item != path]
    #再重新加入,到最前面
    files.append(path)
    #截一下容量限制
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    """追加一条 episodic 或 durable note。"""
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state

    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] = note["note_index"] + 1

    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state
def set_file_summary(state, path, summary, workspace_root=None):
    """为某个文件保存短摘要与 freshness。"""
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    """显式删除某个文件的摘要。"""
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    """清理 freshness 不匹配的文件摘要。"""
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    #这个函数会遍历所有文件摘要，
    # 如果当前文件摘要的 freshness 与当前文件一致，则保留，否则删除。
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        #将字典中这一项取出并且删除
        state["file_summaries"].pop(path, None)
    return state, invalidated




def summarize_read_result(result, limit=180):
    """把 read_file 结果压成可写入 memory 的短摘要。"""
    # 我们不会把完整文件内容塞进记忆层，
    # 这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    #如果第一行是标题，则忽略标题行。
    if lines[0].startswith("# "):
        lines = lines[1:]
    #有可能去掉标题后，后面就没内容了
    if not lines:
        return "(empty)"
    #截取前 3 行，并合并成摘要
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(state, query, limit=3, workspace_root=None):
    """从 episodic memory 和 durable memory 里统一召回候选笔记。"""
    state = normalize_memory_state(state, workspace_root)
    query_tokens = _tokenize(query)
    ranked = []
    for note in state["episodic_notes"]:
        score = _score_retrieval_note(query_tokens, note, tie_breaker=1)
        if score[0] == 0 and score[1] == 0:
            continue
        ranked.append((score, note))

    if workspace_root is not None:
        durable_store = DurableMemoryStore(Path(workspace_root) / ".hepilot" / "memory")
        for score, note in durable_store.ranked_retrieval_candidates(query, limit=limit):
            ranked.append((score, note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None):
    """把召回结果渲染成给模型看的紧凑文本。"""
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
        return "\n".join(lines)
    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state, workspace_root=None):
    """把 memory 渲染成 prompt 里的概览面板。"""
    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    #这个line是一个列表,通过append/entend的方式添加元素
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    """判断 memory 是否已经接近空白状态。"""
    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    """对 memory state 的面向对象包装。

    runtime 更频繁调用的是这个类，而不是底层纯函数；它负责把当前工作区根、
    durable store 和 state 变更封装在一起。
    """

    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = DurableMemoryStore(Path(workspace_root) / ".hepilot" / "memory") if workspace_root is not None else None

    def to_dict(self):
        """返回规范化后的原始状态字典。"""
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        """把路径转换成 memory 内部统一使用的相对形式。"""
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        #这里的调用回来的结果会更新当前的state,并且将删除的不合法记忆文件的名单传回去
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3):
        return retrieval_candidates(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def retrieval_view(self, query, limit=3):
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self):
        return render_memory_text(self.state, self.workspace_root)

    def promote_durable(self, promotions):
        """将筛选后的长期事实持久化，并同步刷新内存态。"""
        if self.durable_store is None:
            return [], []
        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions)
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded
