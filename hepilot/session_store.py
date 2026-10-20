"""集中式会话存储。

一个会话一个目录，按工作区分组：

    <root>/
      --<cwd-slug>--/
        <session_id>/
          session.json      只存头，不随对话增长
          events.jsonl      只追加，一行一个事件
          snapshot.json     聚合态缓存（memory / checkpoints），可删可重建
          runs/<run_id>/    单次运行工件

设计取舍（有意为之）：
- `snapshot.json` 里的 memory 是**权威**，不是从事件重放出来的。
  代价是删掉它就等于丢记忆；换来的是不必维护一条永远对不齐的 reducer 链。
- 事件是对话与工具的唯一权威。`load()` 用 snapshot + 回放尾部事件还原。
- 写 header / snapshot 走原子写；事件走单行追加。
"""

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# 会话磁盘格式版本；字段只增不改，改结构才升版本。
SESSION_FORMAT_VERSION = 1
# 单个事件的信封版本，按类型自带。
EVENT_VERSION = 1

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now():
    """统一时间戳格式，和 trace/report 保持一致。"""
    return datetime.now(timezone.utc).isoformat()


def workspace_slug(workspace_root):
    """把工作区路径压成一个安全、可读、可排序的目录名。"""
    text = str(Path(workspace_root).resolve()).replace("\\", "/")
    # 保留盘符和层级分隔语义，其余字符一律压平。
    text = _SAFE_SLUG.sub("_", text).strip("/")
    if ":" in text:
        text = text.replace(":", "-")
    return f"--{text or 'no-cwd'}--"


def atomic_write_text(path, text):
    """先写临时文件再 replace，避免留下半截文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    Path(temp_name).replace(path)


def atomic_write_json(path, payload):
    """原子写 JSON，并保证结尾只有一个换行。"""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


class SessionStore:
    """集中式会话存储。

    对上层暴露两类接口：

    - 新接口：`create / stat / list / append_event / write_snapshot`
    - 兼容接口：`save / load / latest / path`，供既有 runtime 与测试继续使用
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 记住每个会话当前的 seq，避免每次追加都去读文件尾。
        self._seq = {}

    # ------------------------------------------------------------------ 路径

    def dir_for(self, workspace_root, session_id):
        """在会话落盘之前先算出目录，供 RunStore 等提前挂载。"""
        return self.root / workspace_slug(workspace_root) / str(session_id)

    def session_dir(self, session_id):
        """按 id 定位会话目录。id 本身是扁平的，目录由 header 里的工作区决定。"""
        session_id = str(session_id)
        found = self._find_dir(session_id)
        if found is not None:
            return found
        return self.root / "--no-cwd--" / session_id

    def _find_dir(self, session_id):
        direct = self.root / session_id
        if (direct / "session.json").exists():
            return direct
        for candidate in self.root.glob(f"*/{session_id}"):
            if (candidate / "session.json").exists():
                return candidate
        return None

    def path(self, session_id):
        """兼容旧调用：返回 `session.json` 的路径。"""
        return self.session_dir(session_id) / "session.json"

    def events_path(self, session_id):
        return self.session_dir(session_id) / "events.jsonl"

    def snapshot_path(self, session_id):
        return self.session_dir(session_id) / "snapshot.json"

    def runs_dir(self, session_id):
        return self.session_dir(session_id) / "runs"

    # --------------------------------------------------------------- 创建/读

    def create(self, header):
        """新建一个会话目录并写出 header。目录由工作区 slug 决定。"""
        session_id = str(header["id"])
        directory = self.root / workspace_slug(header.get("workspace_root", "")) / session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "runs").mkdir(exist_ok=True)
        payload = {
            "format_version": SESSION_FORMAT_VERSION,
            "id": session_id,
            "title": str(header.get("title", "") or "").strip(),
            "workspace_root": str(header.get("workspace_root", "")),
            "created_at": header.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "status": str(header.get("status", "idle")),
            "last_seq": 0,
            "last_run_id": "",
        }
        atomic_write_json(directory / "session.json", payload)
        self._seq[session_id] = 0
        return payload

    def read_header(self, session_id):
        """读 header；不存在返回 None。"""
        path = self.path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update_header(self, session_id, **changes):
        """局部更新 header 字段。"""
        header = self.read_header(session_id) or {}
        header.update(changes)
        header["updated_at"] = utc_now()
        atomic_write_json(self.path(session_id), header)
        return header

    def stat(self, session_id):
        """返回一个会话的元数据快照，不读事件体。"""
        header = self.read_header(session_id)
        if header is None:
            return None
        events = self.events_path(session_id)
        return {
            "id": header.get("id", session_id),
            "title": header.get("title", ""),
            "workspace_root": header.get("workspace_root", ""),
            "created_at": header.get("created_at", ""),
            "updated_at": header.get("updated_at", ""),
            "status": header.get("status", "idle"),
            "last_seq": int(header.get("last_seq", 0)),
            "last_run_id": header.get("last_run_id", ""),
            "event_count": self._count_events(events),
            "events_bytes": events.stat().st_size if events.exists() else 0,
        }

    def list(self):
        """列出全部会话，按最近活跃倒序。"""
        items = []
        for header_path in self.root.glob("*/*/session.json"):
            session_id = header_path.parent.name
            meta = self.stat(session_id)
            if meta:
                items.append(meta)
        items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return items

    def latest(self):
        """兼容旧调用：返回最近活跃的会话 id。"""
        items = self.list()
        return items[0]["id"] if items else None

    @staticmethod
    def _count_events(path):
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    # ------------------------------------------------------------- 事件/快照

    def append_event(self, session, event_type, data=None, run_id=""):
        """追加一个事件，返回补全信封后的完整事件。"""
        session_id = str(session["id"])
        seq = self._seq.get(session_id)
        if seq is None:
            seq = int((self.read_header(session_id) or {}).get("last_seq", 0))
        seq += 1
        self._seq[session_id] = seq

        payload = {
            "seq": seq,
            "ts": utc_now(),
            "run": str(run_id or ""),
            "type": str(event_type),
            "v": EVENT_VERSION,
            "data": data or {},
        }
        path = self.events_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            handle.write("\n")

        header = self.read_header(session_id)
        if header is not None:
            header["last_seq"] = seq
            if run_id:
                header["last_run_id"] = str(run_id)
            header["updated_at"] = payload["ts"]
            atomic_write_json(self.path(session_id), header)
        return payload

    def read_events(self, session_id, after_seq=0):
        """读取 seq 大于 after_seq 的事件。"""
        path = self.events_path(session_id)
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # 崩在写入中途可能留下半行；跳过而不是整体失败。
                    continue
                if int(event.get("seq", 0)) > after_seq:
                    events.append(event)
        return events

    def write_snapshot(self, session):
        """把聚合态写成快照。memory 是权威，不是缓存。"""
        session_id = str(session["id"])
        payload = {
            "format_version": SESSION_FORMAT_VERSION,
            "seq": int(self._seq.get(session_id, 0)),
            "updated_at": utc_now(),
            "memory": session.get("memory", {}),
            "checkpoints": session.get("checkpoints", {"current_id": "", "items": {}}),
            "runtime_identity": session.get("runtime_identity", {}),
        }
        atomic_write_json(self.snapshot_path(session_id), payload)
        return payload

    def _write_snapshot_payload(self, session_id, payload):
        atomic_write_json(self.snapshot_path(session_id), payload)

    def read_snapshot(self, session_id):
        path = self.snapshot_path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- 兼容旧 API

    def save(self, session):
        """兼容旧调用：确保目录存在、写 header 与快照。"""
        session_id = str(session["id"])
        header = self.read_header(session_id)
        if header is None:
            header = self.create(
                {
                    "id": session_id,
                    "title": session.get("title", ""),
                    "workspace_root": session.get("workspace_root", ""),
                    "created_at": session.get("created_at"),
                }
            )
        created = {
            "id": session_id,
            "title": session.get("title", ""),
            "workspace_root": session.get("workspace_root", ""),
            "created_at": session.get("created_at") or header.get("created_at"),
        }
        if not header.get("workspace_root") and created["workspace_root"]:
            header.update(created)
            atomic_write_json(self.path(session_id), header)
        payload = {
            "format_version": SESSION_FORMAT_VERSION,
            "seq": int(self._seq.get(session_id, 0)),
            "updated_at": utc_now(),
            "memory": session.get("memory", {}),
            "checkpoints": session.get("checkpoints", {"current_id": "", "items": {}}),
            "runtime_identity": session.get("runtime_identity", {}),
        }
        # 兼容桥：完全没有事件的会话（测试与旧数据）把 history 整份放进快照，
        # 这样 save()/load() 仍然能原样往返。一旦有事件，history 以事件为准。
        if self._count_events(self.events_path(session_id)) == 0:
            payload["history"] = session.get("history", [])
        self._write_snapshot_payload(session_id, payload)
        return self.path(session_id)

    def load(self, session_id):
        """还原会话字典：header + 快照 + 回放快照之后的事件。"""
        header = self.read_header(session_id)
        if header is None:
            raise FileNotFoundError(f"session not found: {session_id}")
        snapshot = self.read_snapshot(session_id) or {}
        session = {
            "id": header.get("id", session_id),
            "title": header.get("title", ""),
            "created_at": header.get("created_at", ""),
            "workspace_root": header.get("workspace_root", ""),
            "history": list(snapshot.get("history") or []),
            "memory": snapshot.get("memory") or {},
            "checkpoints": snapshot.get("checkpoints") or {"current_id": "", "items": {}},
            "runtime_identity": snapshot.get("runtime_identity") or {},
            "resume_state": {},
        }
        # 转录以事件为唯一权威：一律从 0 回放。
        # `session_reset` 是逻辑截断点——从最后一次 reset 之后开始回放，
        # 这样 append-only 的日志也能表达"这段历史被清掉了"。
        # 快照里的 history 只是给"完全没有事件的旧会话"留的兼容桥。
        events = self.read_events(session_id, after_seq=0)
        if not events:
            history = list(snapshot.get("history") or [])
        else:
            start = 0
            for index, event in enumerate(events):
                if event.get("type") == "session_reset":
                    start = index + 1
            history = []
            for event in events[start:]:
                item = event_to_history_item(event)
                if item is not None:
                    history.append(item)
        session["history"] = history
        self._seq[session_id] = int(header.get("last_seq", 0))
        return session


def event_to_history_item(event):
    """把转录类事件还原成 history 条目；非转录事件返回 None。"""
    kind = event.get("type")
    data = event.get("data") or {}
    created_at = event.get("ts", "")
    if kind == "user_message":
        return {"role": "user", "content": str(data.get("text", "")), "created_at": created_at}
    if kind == "assistant_message":
        item = {"role": "assistant", "content": str(data.get("text", "")), "created_at": created_at}
        if data.get("tool_uses"):
            item["tool_uses"] = data["tool_uses"]
        if data.get("kind"):
            item["kind"] = data["kind"]
        return item
    if kind == "tool_executed":
        item = {
            "role": "tool",
            "parent_tool_use_id": str(data.get("tool_use_id", "")),
            "name": str(data.get("name", "")),
            "args": data.get("args", {}) or {},
            "content": str(data.get("result", "")),
            "created_at": created_at,
        }
        if data.get("synthetic"):
            item["synthetic"] = True
        return item
    return None
