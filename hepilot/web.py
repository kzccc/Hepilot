"""HTTP + SSE 服务层。

把 Hepilot 暴露成浏览器可用的接口，同时保留"会话集中存储"的形状：

- 会话由 `SessionStore` 集中存放，按工作区分组
- 每轮 `ask()` 在后台线程里跑，前端通过 SSE 读 `events.jsonl` 增量
- 风险工具的审批走带外回调，不读 stdin

不引入第三方依赖：用标准库的 `ThreadingHTTPServer`。
"""

import json
import mimetypes
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import memory as memorylib
from .models import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .run_store import RunStore
from .runtime import Hepilot
from .session_store import SessionStore, utc_now, workspace_slug
from .workspace import WorkspaceContext

ASSETS_DIR = Path(__file__).parent / "web_assets"
APPROVAL_TIMEOUT_SECONDS = 300
GLOBAL_CONFIG_PATH = Path.home() / ".hepilot" / "config.json"


def _provider_config(provider):
    """读取 ~/.hepilot/config.json 里某个 provider 的默认项。

    和 CLI 保持同一套回落顺序：命令行 > 环境变量 > 全局配置 > 代码默认值。
    """
    try:
        payload = json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    providers = dict(payload.get("providers", {}) or {})
    return dict(providers.get(str(provider), {}) or {})


# --------------------------------------------------------------------- 配置


@dataclass
class AgentConfig:
    """一次服务进程使用的模型与会话配置。"""

    provider: str = "openai"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_style: str = ""
    temperature: float = 0.2
    max_steps: int = 20
    max_new_tokens: int = 2048
    approval_policy: str = "ask"
    secret_env_names: tuple = ()


def build_model_client(config):
    """按 provider 构造模型 client，回落顺序与 CLI 一致。"""
    provider = (config.provider or "openai").lower()
    provider_config = _provider_config(provider)
    if provider == "ollama":
        return OllamaModelClient(
            model=config.model or "qwen3.5:4b",
            host=config.base_url or "http://127.0.0.1:11434",
            temperature=config.temperature,
            top_p=0.9,
            timeout=300,
        )
    if provider == "anthropic":
        return AnthropicCompatibleModelClient(
            model=config.model or "claude-sonnet-4-6",
            base_url=config.base_url or str(provider_config.get("base_url", "")).strip() or "https://www.right.codes/claude/v1",
            api_key=config.api_key
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or str(provider_config.get("api_key", "")).strip(),
            temperature=config.temperature,
            timeout=300,
        )
    return OpenAICompatibleModelClient(
        model=config.model or "gpt-5.4",
        base_url=config.base_url
        or os.environ.get("OPENAI_API_BASE", "")
        or str(provider_config.get("base_url", "")).strip()
        or "https://www.right.codes/codex/v1",
        api_key=config.api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or str(provider_config.get("api_key", "")).strip(),
        temperature=config.temperature,
        timeout=300,
        api_style=config.api_style or None,
    )


# --------------------------------------------------------------------- 会话


class PendingApproval:
    """一次等待前端裁决的风险工具调用。"""

    def __init__(self, approval_id, name, args):
        self.id = approval_id
        self.name = name
        self.args = args
        self.event = threading.Event()
        self.approved = False
        self.reason = ""


class SessionHandle:
    """一个会话对应的运行时对象与并发门禁。"""

    def __init__(self, manager, agent):
        self.manager = manager
        self.agent = agent
        self.lock = threading.Lock()
        self.running = False
        self.pending = {}
        self.agent.approval_handler = self._approval_handler
        # 流式输出：模型每吐一块文本，就落一条 assistant_delta 事件。
        self.agent.stream_sink = self._on_delta
        self.agent.stream_flush = self._flush_delta
        # 取消令牌：前端 POST /cancel 置位，runtime 在循环边界和流式读取处检查。
        self.cancel_event = threading.Event()
        self.agent.cancel_token = self.cancel_event
        self._delta_buf = []
        self._delta_text = ""
        self._delta_started = False
        self._delta_flushed_at = 0.0

    # ---- 流式 ----

    def _run_id(self):
        task_state = getattr(self.agent, "current_task_state", None)
        return getattr(task_state, "run_id", "") or ""

    @staticmethod
    def _stream_kind(text):
        """从已累积的文本判断这一轮模型在产出什么。"""
        tool_at = text.find("<tool")
        final_at = text.find("<final>")
        if tool_at != -1 and (final_at == -1 or tool_at < final_at):
            return "tool"
        if final_at != -1:
            return "final"
        return "text"

    def _reset_delta(self):
        self._delta_buf = []
        self._delta_text = ""
        self._delta_started = False

    def _on_delta(self, chunk):
        """模型层的增量回调。chunk 为 None 表示“新一轮调用开始”。"""
        if chunk is None:
            self._reset_delta()
            self._emit("assistant_start", {})
            self._delta_started = True
            return
        if not chunk:
            return
        self._delta_buf.append(chunk)
        self._delta_text += chunk
        now = time.time()
        # 合并小块：太频繁的落盘会让 events.jsonl 爆炸。
        if now - self._delta_flushed_at >= 0.05 or len("".join(self._delta_buf)) >= 160 or "\n" in chunk:
            self._flush_delta()

    def _flush_delta(self):
        if not self._delta_buf:
            return
        text = "".join(self._delta_buf)
        self._delta_buf = []
        self._delta_flushed_at = time.time()
        self._emit("assistant_delta", {"text": text, "kind": self._stream_kind(self._delta_text)})

    def _emit(self, event_type, data):
        try:
            self.agent.session_store.append_event(self.agent.session, event_type, data, run_id=self._run_id())
        except Exception:
            # 流式落盘失败不该拖垮这一轮。
            pass

    # 审批：写一条事件告诉前端，然后阻塞等待裁决
    def _approval_handler(self, name, args):
        approval_id = f"ap_{len(self.pending) + 1}_{int(time.time() * 1000)}"
        pending = PendingApproval(approval_id, name, args)
        self.pending[approval_id] = pending
        self.agent.session_store.append_event(
            self.agent.session,
            "approval_requested",
            {"approval_id": approval_id, "name": name, "args": args},
            run_id=getattr(getattr(self.agent, "current_task_state", None), "run_id", "") or "",
        )
        try:
            got = pending.event.wait(APPROVAL_TIMEOUT_SECONDS)
            approved = bool(got and pending.approved)
            self.agent.session_store.append_event(
                self.agent.session,
                "approval_resolved",
                {
                    "approval_id": approval_id,
                    "name": name,
                    "approved": approved,
                    "reason": pending.reason or ("" if got else "timeout"),
                },
                run_id=getattr(getattr(self.agent, "current_task_state", None), "run_id", "") or "",
            )
            return approved
        finally:
            self.pending.pop(approval_id, None)

    def resolve(self, approval_id, approved, reason=""):
        pending = self.pending.get(str(approval_id))
        if pending is None:
            return False
        pending.approved = bool(approved)
        pending.reason = str(reason or "")
        pending.event.set()
        return True


class SessionManager:
    """集中管理会话：创建、加载、跑任务。"""

    def __init__(self, store, config):
        self.store = SessionStore(store)
        self.config = config
        self.handles = {}
        self.guard = threading.Lock()
        self.reconcile_orphans()

    def reconcile_orphans(self):
        """把上次进程没跑完的 running 收回来。

        进程被 kill（重启、崩溃、Ctrl+C）时，磁盘上会留下 status=running
        但永远不会有 run_finished 的会话。启动时统一收口，否则它会一直
        显示"运行中"，而且新消息会被 busy 挡住。
        """
        repaired = []
        for meta in self.store.list():
            if meta.get("status") != "running":
                continue
            session_id = meta["id"]
            try:
                header = self.store.read_header(session_id) or {}
                self.store.append_event(
                    {"id": session_id, "workspace_root": header.get("workspace_root", "")},
                    "run_interrupted",
                    {"reason": "process_restarted", "last_run_id": header.get("last_run_id", "")},
                )
                self.store.update_header(session_id, status="interrupted")
                repaired.append(session_id)
            except Exception:
                continue
        return repaired

    # ---- 创建与获取 ----

    def create(self, workspace_root, title="", name=""):
        root = Path(str(workspace_root)).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace does not exist or is not a directory: {root}")
        if root.parent == root:
            # 盘符根/文件系统根：list_files 会列出整块盘，
            # 高风险工具的工作区快照要递归 hash 整块盘，能把 run 卡死。
            raise ValueError(
                f"refusing to use a filesystem root as workspace: {root}. 请选一个具体的项目目录。"
            )
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()
        session = {
            "id": session_id,
            "name": str(name or "").strip(),
            "title": str(title or "").strip() or f"会话 {session_id[:15]}",
            "created_at": utc_now(),
            "workspace_root": str(root),
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        agent = self._build_agent(session)
        agent.session_store = self.store
        agent.session_store.create(
            {
                "id": session_id,
                "name": session["name"],
                "title": session["title"],
                "workspace_root": str(root),
                "created_at": session["created_at"],
            }
        )
        self.store.append_event(session, "session_created", {"workspace_root": str(root)})
        agent.session_path = self.store.save(agent.session)
        return self.handle(session_id)

    def _build_agent(self, session):
        workspace = WorkspaceContext.build(session["workspace_root"], repo_root_override=session["workspace_root"])
        agent = Hepilot(
            model_client=build_model_client(self.config),
            workspace=workspace,
            session_store=self.store,
            session=session,
            run_store=RunStore(self.store.dir_for(session["workspace_root"], session["id"]) / "runs"),
            approval_policy=self.config.approval_policy,
            max_steps=self.config.max_steps,
            max_new_tokens=self.config.max_new_tokens,
            secret_env_names=tuple(self.config.secret_env_names or ()),
            user_memory_root=memorylib.default_user_memory_root(),
        )
        return agent

    def handle(self, session_id):
        """按 id 取句柄；不存在就从磁盘还原一个。"""
        with self.guard:
            handle = self.handles.get(session_id)
            if handle is not None:
                return handle
            header = self.store.read_header(session_id)
            if header is None:
                raise KeyError(f"session not found: {session_id}")
            session = self.store.load(session_id)
            session["title"] = header.get("title", "")
            agent = self._build_agent(session)
            handle = SessionHandle(self, agent)
            self.handles[session_id] = handle
            return handle

    # ---- 运行 ----

    def submit(self, session_id, text):
        handle = self.handle(session_id)
        if handle.running:
            raise RuntimeError("session is busy")
        handle.running = True
        handle._reset_delta()
        handle.cancel_event.clear()
        self.store.update_header(session_id, status="running")

        def worker():
            try:
                handle.agent.ask(str(text))
            except Exception as exc:  # 兜底：任何未收口异常也要落成事件
                self.store.append_event(
                    handle.agent.session,
                    "run_failed",
                    {"error_type": exc.__class__.__name__, "error_message": str(exc)},
                )
            finally:
                try:
                    handle._flush_delta()
                except Exception:
                    pass
                try:
                    self.store.write_snapshot(handle.agent.session)
                finally:
                    self.store.update_header(session_id, status="idle")
                    handle.running = False

        thread = threading.Thread(target=worker, name=f"ask-{session_id}", daemon=True)
        thread.start()
        return handle


    def cancel(self, session_id):
        """请求取消当前运行。真正的收口由 runtime 在检查点完成。"""
        handle = self.handle(session_id)
        if not handle.running:
            return False
        handle.cancel_event.set()
        self.store.append_event(handle.agent.session, "cancel_requested", {"reason": "user_cancelled"})
        return True


# --------------------------------------------------------------------- 路由


def _read_trace(store, session_id, run_id):
    """读某次 run 的 trace.jsonl；缺失就返回空列表。"""
    if not run_id:
        return []
    path = store.runs_dir(session_id) / str(run_id) / "trace.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def build_trajectory(store, session_id):
    """把会话折叠成「每轮一条」的轨迹，供轨迹页展示。

    思路和 DSH 的 turnOutline 一样：以用户消息为一轮的边界，
    每轮挂上它的时间线。差别是我们把 run trace 里的细节也带上
    （模型耗时、工具耗时、状态），因为这里同时当调试视图用。
    """
    header = store.read_header(session_id)
    if header is None:
        raise KeyError(f"session not found: {session_id}")
    events = store.read_events(session_id)

    turns = []
    current = None
    for event in events:
        kind = event.get("type")
        data = event.get("data") or {}
        if kind == "user_message":
            if current is not None:
                turns.append(current)
            current = {
                "index": len(turns) + 1,
                "seq": event.get("seq"),
                "ts": event.get("ts", ""),
                # 用户消息是这一轮的开始，此时 TaskState 还没建，
                # 它自带的 run 可能是上一轮的，所以这里一律留空，
                # 由紧随其后的 run_started 补上。
                "run": "",
                "prompt": str(data.get("text", "")),
                "answer": "",
                "status": "",
                "stopReason": "",
                "durationMs": None,
                "steps": [],
            }
            continue
        if current is None:
            continue
        # user_message 是在 TaskState 创建之前写的，run 字段为空；
        # 用同一轮里后面第一条带 run 的事件把它补齐，才能找到对应的 trace。
        if not current["run"] and event.get("run"):
            current["run"] = str(event["run"])
        if kind == "assistant_message":
            if str(data.get("kind", "")) in {"final", "stopped"} and data.get("text"):
                current["answer"] = str(data["text"])
        elif kind == "run_finished":
            current["status"] = str(data.get("status", ""))
            current["stopReason"] = str(data.get("stop_reason", ""))
            current["durationMs"] = data.get("run_duration_ms")
    if current is not None:
        turns.append(current)

    for turn in turns:
        trace = _read_trace(store, session_id, turn["run"])
        turn["steps"] = [
            {
                "event": str(item.get("event", "")),
                "ts": item.get("created_at", ""),
                "durationMs": item.get("duration_ms"),
                "tool": item.get("name", ""),
                "args": item.get("args") or {},
                "status": item.get("tool_status", ""),
                "stopReason": item.get("stop_reason", ""),
                "trigger": item.get("trigger", ""),
                "attempts": item.get("attempts"),
                "toolSteps": item.get("tool_steps"),
            }
            for item in trace
            if item.get("event") not in {"run_started", "run_finished"}
        ]
    return {"session": store.stat(session_id), "turns": turns, "eventCount": len(events)}


def transcript_from_events(events):
    """把事件流压成前端直接能画的条目列表。"""
    items = []
    for event in events:
        kind = event.get("type")
        data = event.get("data") or {}
        seq = event.get("seq")
        if kind == "user_message":
            items.append({"kind": "user", "seq": seq, "text": data.get("text", "")})
        elif kind == "assistant_message":
            items.append(
                {
                    "kind": "assistant",
                    "seq": seq,
                    "text": data.get("text", ""),
                    "toolUses": data.get("tool_uses") or [],
                    "msgKind": data.get("kind", ""),
                }
            )
        elif kind == "assistant_delta":
            items.append({"kind": "delta", "seq": seq, "text": data.get("text", ""), "streamKind": data.get("kind", "text")})
        elif kind == "assistant_start":
            items.append({"kind": "assistant_start", "seq": seq})
        elif kind == "tool_executed":
            items.append(
                {
                    "kind": "tool",
                    "seq": seq,
                    "name": data.get("name", ""),
                    "args": data.get("args") or {},
                    "result": data.get("result", ""),
                    "synthetic": bool(data.get("synthetic")),
                }
            )
        elif kind == "run_started":
            items.append({"kind": "run_start", "seq": seq, "runId": event.get("run", "")})
        elif kind == "run_finished":
            items.append(
                {
                    "kind": "run_end",
                    "seq": seq,
                    "status": data.get("status", ""),
                    "stopReason": data.get("stop_reason", ""),
                    "final": data.get("final_answer", ""),
                }
            )
        elif kind == "checkpoint_created":
            items.append({"kind": "checkpoint", "seq": seq, "trigger": data.get("trigger", "")})
        elif kind == "model_failed":
            items.append({"kind": "error", "seq": seq, "message": data.get("error_message", "")})
        elif kind == "approval_requested":
            items.append(
                {
                    "kind": "approval",
                    "seq": seq,
                    "approvalId": data.get("approval_id", ""),
                    "name": data.get("name", ""),
                    "args": data.get("args") or {},
                }
            )
        elif kind == "approval_resolved":
            items.append(
                {
                    "kind": "approval_resolved",
                    "seq": seq,
                    "approvalId": data.get("approval_id", ""),
                    "approved": bool(data.get("approved")),
                    "reason": data.get("reason", ""),
                }
            )
    return items


_PICKER_LOCK = threading.Lock()


def pick_folder_native(initial=""):
    """弹出操作系统的原生目录选择框。

    只在本地场景可用（服务跑在用户自己的桌面上）。用系统的文件夹对话框
    而不是自己画一个，是因为用户要的就是"像选本地文件夹那样选"。
    """
    if os.name != "nt":
        raise RuntimeError("native folder picker is only implemented on Windows")
    if not _PICKER_LOCK.acquire(blocking=False):
        raise RuntimeError("folder picker is already open")
    try:
        # 初始路径通过环境变量传入，而不是拼进脚本字符串：
        # 路径里可能带引号，盘符根还会以反斜杠结尾，拼字符串会破语法。
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.Description = 'Select workspace folder';"
            "$d.ShowNewFolderButton = $true;"
            "if ($env:HEPILOT_PICK_INITIAL) { $d.SelectedPath = $env:HEPILOT_PICK_INITIAL };"
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }"
        )
        env = dict(os.environ)
        env["HEPILOT_PICK_INITIAL"] = str(initial or os.path.expanduser("~"))
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=env,
        )
        chosen = (result.stdout or "").strip().splitlines()
        path = chosen[-1].strip() if chosen else ""
        if not path:
            raise RuntimeError("no folder selected")
        return path
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("folder picker timed out") from exc
    finally:
        _PICKER_LOCK.release()


def list_directory(path):
    """给前端的工作区选择器列目录。"""
    target = Path(str(path)).expanduser().resolve()
    if not target.exists() or not target.is_dir():
        raise ValueError(f"not a directory: {target}")
    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except PermissionError:
        children = []
    for child in children[:300]:
        if child.name.startswith(".git") and child.name != ".gitignore":
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "isDir": child.is_dir(),
                "isRepo": (child / ".git").exists(),
            }
        )
    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else "",
        "isRepo": (target / ".git").exists(),
        "entries": entries,
    }


class RequestHandler(BaseHTTPRequestHandler):
    """极简 REST + SSE 路由。"""

    server_version = "hepilot-web/1.0"
    manager = None  # 由 serve() 注入

    # 关掉默认的 noisy 日志
    def log_message(self, fmt, *args):
        return

    # ---- 基础工具 ----

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": str(message)}, status=status)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_asset(self, name):
        path = (ASSETS_DIR / name).resolve()
        if not str(path).startswith(str(ASSETS_DIR.resolve())) or not path.is_file():
            self._send_error_json("not found", 404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix in {".html", ".css"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- 路由 ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                return self._send_asset("index.html")
            if path.startswith("/static/"):
                return self._send_asset(path[len("/static/"):])
            if path == "/api/sessions":
                return self._send_json({"sessions": self.manager.store.list()})
            if path == "/api/fs":
                target = (query.get("path") or [""])[0]
                return self._send_json(list_directory(target or Path.home()))
            if path.startswith("/api/sessions/"):
                rest = path[len("/api/sessions/"):]
                if rest.endswith("/events"):
                    session_id = unquote(rest[: -len("/events")])
                    after = int((query.get("after") or ["0"])[0] or 0)
                    return self._stream_events(session_id, after)
                if rest.endswith("/trajectory"):
                    session_id = unquote(rest[: -len("/trajectory")])
                    return self._send_json(build_trajectory(self.manager.store, session_id))
                session_id = unquote(rest)
                header = self.manager.store.read_header(session_id)
                if header is None:
                    return self._send_error_json("session not found", 404)
                events = self.manager.store.read_events(session_id)
                return self._send_json(
                    {
                        "session": self.manager.store.stat(session_id),
                        "transcript": transcript_from_events(events),
                        "lastSeq": events[-1]["seq"] if events else 0,
                    }
                )
            return self._send_error_json("not found", 404)
        except KeyError as exc:
            return self._send_error_json(str(exc), 404)
        except Exception as exc:  # 兜底，避免 handler 线程静默死掉
            return self._send_error_json(f"{exc.__class__.__name__}: {exc}", 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json()
        try:
            if path == "/api/pick-folder":
                return self._send_json({"path": pick_folder_native(payload.get("initial", ""))})
            if path == "/api/sessions":
                handle = self.manager.create(
                    payload.get("workspace_root", ""),
                    payload.get("title", ""),
                    payload.get("name", ""),
                )
                return self._send_json({"session": self.manager.store.stat(handle.agent.session["id"])})
            if path.startswith("/api/sessions/"):
                rest = path[len("/api/sessions/"):]
                if "/approvals/" in rest:
                    session_id, _, approval_id = rest.partition("/approvals/")
                    handle = self.manager.handle(unquote(session_id))
                    ok = handle.resolve(unquote(approval_id), payload.get("approved", False), payload.get("reason", ""))
                    if not ok:
                        return self._send_error_json("approval not pending", 404)
                    return self._send_json({"ok": True})
                if rest.endswith("/messages"):
                    session_id = unquote(rest[: -len("/messages")])
                    text = str(payload.get("text", "")).strip()
                    if not text:
                        return self._send_error_json("text must not be empty")
                    self.manager.submit(session_id, text)
                    return self._send_json({"ok": True}, 202)
                if rest.endswith("/cancel"):
                    session_id = unquote(rest[: -len("/cancel")])
                    ok = self.manager.cancel(session_id)
                    return self._send_json({"ok": ok})
            return self._send_error_json("not found", 404)
        except KeyError as exc:
            return self._send_error_json(str(exc), 404)
        except ValueError as exc:
            # 工作区不合法（不存在 / 不是目录 / 是盘符根）属于用户输入问题。
            return self._send_error_json(str(exc), 400)
        except RuntimeError as exc:
            return self._send_error_json(str(exc), 409)
        except Exception as exc:
            return self._send_error_json(f"{exc.__class__.__name__}: {exc}", 500)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json()
        try:
            if path.startswith("/api/sessions/"):
                session_id = unquote(path[len("/api/sessions/"):])
                if self.manager.store.read_header(session_id) is None:
                    return self._send_error_json("session not found", 404)
                changes = {}
                if "name" in payload:
                    # name 只是展示名；session id 仍然是磁盘上和接口里的唯一标识。
                    changes["name"] = str(payload.get("name") or "").strip()[:80]
                if changes:
                    self.manager.store.update_header(session_id, **changes)
                return self._send_json({"session": self.manager.store.stat(session_id)})
            return self._send_error_json("not found", 404)
        except Exception as exc:
            return self._send_error_json(f"{exc.__class__.__name__}: {exc}", 500)

    # ---- SSE ----

    def _stream_events(self, session_id, after_seq):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = int(after_seq)
        idle = 0
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                events = self.manager.store.read_events(session_id, after_seq=cursor)
                if events:
                    idle = 0
                    for event in events:
                        cursor = max(cursor, int(event.get("seq", 0)))
                        chunk = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(chunk)
                    self.wfile.flush()
                else:
                    idle += 1
                    if idle % 20 == 0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                # 会话空闲时也不必高频轮询
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def serve(store_root, config, host="127.0.0.1", port=8765):
    """启动 HTTP 服务，返回 server 对象（调用方负责 serve_forever）。"""
    manager = SessionManager(store_root, config)
    handler = type("BoundRequestHandler", (RequestHandler,), {"manager": manager})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.manager = manager
    return server


def main(argv=None):
    """`hepilot web` 的入口。"""
    import argparse

    parser = argparse.ArgumentParser(prog="hepilot web", description="Serve Hepilot as a browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sessions-root", default=str(Path.home() / ".hepilot" / "sessions"))
    parser.add_argument("--provider", default="openai", choices=("ollama", "openai", "anthropic"))
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-style", default="", choices=("", "responses", "chat"), help="openai 兼容接口风格；留空按 base_url 自动判断")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--approval", default="ask", choices=("ask", "auto", "never"))
    args = parser.parse_args(argv)

    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_style=args.api_style,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        approval_policy=args.approval,
    )
    server = serve(args.sessions_root, config, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"hepilot web listening on {url}")
    print(f"sessions root: {Path(args.sessions_root).resolve()}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
