"""Agent 运行时核心逻辑。

Hepilot 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import os
import re
import textwrap
import uuid
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import memory as memorylib
from .context_manager import ContextManager
from .run_store import RunStore
from .task_state import TaskState
from . import tools as toolkit
from .workspace import IGNORED_PATH_NAMES, MAX_HISTORY, WorkspaceContext, clip, now

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
#功能的开关配置
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "context_reduction": True,
    "prompt_cache": True,
}
# checkpoint 的格式版本
CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
#一般作为默认状态使用
CHECKPOINT_NONE_STATUS = "no-checkpoint"
#三种checkpoint的恢复状态
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
#判定格式为旧版本后,checkpoint的状态应该设置为下面这个
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"

DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")


@dataclass
class PromptPrefix:
    """可缓存的稳定 prompt 前缀及其校验元数据。"""

    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


class SessionStore:
    """管理可恢复 session 的读写。"""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        """返回 session JSON 文件路径。"""
        return self.root / f"{session_id}.json"

    def save(self, session):
        """落盘保存当前 session。"""
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        """读取指定 session。"""
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        """返回最近一次修改的 session id。"""
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


class Hepilot:
    """本地 coding agent 的主运行时。

    它把 workspace 快照、prompt 组装、模型调用、工具执行、checkpoint、
    memory 和 trace/report 写盘串成一个完整控制循环。
    """

    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".hepilot" / "runs")
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self.build_tools()
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        """从已保存 session 恢复一个 Hepilot 实例。"""
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self):
        """补齐旧 session 缺失的关键字段，保持向后兼容。"""
        #history
        self.session.setdefault("history", [])
        #memory
        self.session.setdefault("memory", memorylib.default_memory_state())
        #checkpoints
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        #runtime_identity
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        #resume_state
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}

    def current_runtime_identity(self):
        """返回影响 resume 有效性的运行时身份信息。"""
        return {
            "session_id": self.session.get("id", ""),  # 当前会话ID
            "cwd": str(self.root),  # 工作目录路径
            "model": str(getattr(self.model_client, "model", "")),  # 使用的模型名称
            "model_client": self.model_client.__class__.__name__,  # 模型客户端实现类名
            "approval_policy": self.approval_policy,  # 审批策略
            "read_only": bool(self.read_only),  # 是否只读模式
            "max_steps": int(self.max_steps),  # 最大迭代步数
            "max_new_tokens": int(self.max_new_tokens),  # 单次生成 token 上限
            "feature_flags": dict(self.feature_flags),  # 特性开关集合
            "shell_env_allowlist": list(self.shell_env_allowlist),  # 环境变量白名单
            "workspace_fingerprint": getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", self.workspace.fingerprint()),  # 工作区文件结构指纹
            "tool_signature": self.tool_signature(),  # 工具集签名
        }

    def checkpoint_state(self):
        """返回 session 内的 checkpoint 容器。"""
        #这个规范化方法是将避免 session 恢复时字段缺失导致异常。
        self._ensure_session_shape()
        return self.session["checkpoints"]

    def current_checkpoint(self):
        """返回当前活跃 checkpoint，没有则为 `None`。"""
        state = self.checkpoint_state()
        # current_checkpoint_id从state里面拿出来,如果没有就用空字符串,然后strip一下,如果strip之后是空字符串就返回None,否则就继续往下走拿checkpoint item
        checkpoint_id = str(state.get("current_id", "")).strip()
        if not checkpoint_id:
            return None
        return state.get("items", {}).get(checkpoint_id)
    #
    def invalidate_stale_memory(self):
        """清理已经被工作区变更污染的文件摘要。"""
        invalidated = self.memory.invalidate_stale_file_summaries()
        #刚才在上面那个方法将state里面删除了一些文件摘要,所以在这里需要将state重新写入session中
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        """根据 checkpoint、freshness 和 runtime identity 评估恢复状态。"""
        #首先去session中寻找resume_state的历史状态,看看能不能拿出来直接转换为dist,如果能拿出来说明之前已经评估过了,直接用之前的结果就行了,如果拿不出来说明之前没有评估过,或者评估过但是结果不合法,这时候才需要重新评估一次. 这样做的好处是避免每次evaluate_resume_state都要重新评估一遍,提高效率.
        previous_resume_state = dict(self.session.get("resume_state", {}) or {})
        # 检查内存是否被工作区变更污染了，如果有污染的文件摘要，就认为 checkpoint 受到了污染，需要重新评估 resume 状态。
        #这个返回的东西就是被删除的文件摘要的名单,并且state和session已经做相对应的更新
        invalidated = self.invalidate_stale_memory()
        #将checkpoint拿出来
        checkpoint = self.current_checkpoint()
        #防御性编程,设置为默认值,如果checkpoint不存在,就认为这个checkpoint是空的,没有内容,所以status设置为CHECKPOINT_NONE_STATUS
        status = CHECKPOINT_NONE_STATUS
        stale_paths = list(invalidated)
        #用来记录当前运行环境与 checkpoint 保存的运行时身份信息之间发生了哪些不匹配的字段。
        mismatch_fields = []
        if checkpoint:
            #先判断schema_version是不是匹配的,如果不匹配就直接认为这个checkpoint是无效的,状态设置为CHECKPOINT_SCHEMA_MISMATCH_STATUS
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
            else: 
                #版本号匹配后,遍历每一个key_files,然后判断文件摘要是否一致,不一致就加入stale_paths中,如果一致就继续遍历下一个key_files
                for item in checkpoint.get("key_files", []):
                    path = str(item.get("path", "")).strip()
                    if not path:
                        continue
                    expected = item.get("freshness")
                    current = memorylib.file_freshness(path, self.root)
                    if expected != current and path not in stale_paths:
                        stale_paths.append(path)
                #遍历完key_files后,判断runtime_identity是否一致,不一致就加入mismatch_fields中,如果一致就继续遍历下一个key
                saved_identity = dict(checkpoint.get("runtime_identity", {}) or self.session.get("runtime_identity", {}) or {})
                #这里会返回本次运行时候的运行时身份信息
                current_identity = self.current_runtime_identity()
                identity_keys = (
                    "cwd",
                    "model",
                    "model_client",
                    "approval_policy",
                    "read_only",
                    "max_steps",
                    "max_new_tokens",
                    "feature_flags",
                    "shell_env_allowlist",
                    "workspace_fingerprint",
                    "tool_signature",
                )
                for key in identity_keys:
                    if key not in saved_identity:
                        continue
                    if saved_identity.get(key) != current_identity.get(key):
                        mismatch_fields.append(key)
                #确保 mismatch_fields 的结果是确定性的
                mismatch_fields.sort()
                if stale_paths:
                    # 文件摘要不一致（文件变更），checkpoint 部分过期
                    status = CHECKPOINT_PARTIAL_STALE_STATUS
                elif mismatch_fields:
                    # 文件没变但运行环境变了（模型、目录、策略等），不能直接复用
                    status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
                else:
                    # 文件和环境都没变，checkpoint 完全有效
                    status = CHECKPOINT_FULL_VALID_STATUS
        #判断好当前的state后,开始评估resume_state
        resume_state = {
            "status": status,
            "stale_paths": stale_paths,
            "runtime_identity_mismatch_fields": mismatch_fields,
            #累计失效的文件摘要总数
            "stale_summary_invalidations": max(
                len(invalidated),
                int(previous_resume_state.get("stale_summary_invalidations", 0))
                if status == CHECKPOINT_PARTIAL_STALE_STATUS
                else 0,
            ),
        }
        self.session["resume_state"] = resume_state
        self.session["runtime_identity"] = self.current_runtime_identity()
        return resume_state

    def render_checkpoint_text(self):
        """把 checkpoint 摘要渲染进 prompt prefix。"""
        #先拿当前活跃的 checkpoint；如果当前 session 里没有 checkpoint，就返回空字符串，不往 prompt 里加任何 checkpoint 信息
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return ""
        #先准备 checkpoint 摘要的基础几行：恢复状态、当前目标、当前阻塞、下一步建议
        lines = [
            "Task checkpoint:",
            f"- Resume status: {self.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
            f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
            f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
            f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
        ]
        #从 checkpoint 的 key_files 里只提取出非空路径，后面渲染成一行“关键文件”提示
        key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
        lines.append(f"- Key files: {', '.join(key_files) or '-'}")
        #如果 checkpoint 里记录了已完成事项，就把它们压成一行，帮助模型快速知道已经做过什么
        if checkpoint.get("completed"):
            lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
        #如果 checkpoint 里记录了排除项，也一并展示出来，避免恢复后重复走被排除的方向
        if checkpoint.get("excluded"):
            lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
        #如果 resume_state 里有 stale_paths，说明有关键文件已经变旧或不一致，把这些路径提示给模型
        if self.resume_state.get("stale_paths"):
            lines.append("- Stale paths: " + ", ".join(self.resume_state["stale_paths"]))
        #summary 是这次 checkpoint 的简短摘要；非空时追加到最后，方便模型快速建立恢复上下文
        summary = str(checkpoint.get("summary", "")).strip()
        if summary:
            lines.append(f"- Summary: {summary}")
        #最后把每一行用换行拼起来，返回给上层 prefix 组装逻辑
        return "\n".join(lines)

    def build_tools(self):
        """构造当前 agent 暴露给模型的工具注册表。"""
        return toolkit.build_tool_registry(self)

    def tool_signature(self):
        """为当前工具集合生成稳定签名。"""
        payload = []
        for name in sorted(self.tools):
            tool = self.tools[name]
            payload.append(
                {
                    "name": name,
                    "schema": tool["schema"],
                    "risky": tool["risky"],
                    "description": tool["description"],
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def build_prefix(self):
        """构造模型每轮都会看到的稳定工作手册前缀。"""
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        examples = "\n".join(
            [
                '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
                '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
                '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
                '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
                "<final>Done.</final>",
            ]
        )
        # prefix 可以理解成 agent 的“工作手册”：
        # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
        text = textwrap.dedent(
            f"""\
            You are Hepilot, a local coding agent working inside a local repository.

            Rules:
            - Use tools instead of guessing about the workspace.
            - Return exactly one <tool>...</tool> or one <final>...</final>.
            - Tool calls must look like:
              <tool>{{"name":"tool_name","args":{{...}}}}</tool>
            - For write_file and patch_file with multi-line text, prefer XML style:
              <tool name="write_file" path="file.py"><content>...</content></tool>
            - Final answers must look like:
              <final>your answer</final>
            - Never invent tool results.
            - Keep answers concise and concrete.
            - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
            - Before writing tests for existing code, read the implementation first.
            - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
            - New files should be complete and runnable, including obvious imports.
            - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
            - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.

            Tools:
            {tool_text}

            Valid response examples:
            {examples}

            {self.workspace.text()}
            """
        ).strip()
        return PromptPrefix(
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            workspace_fingerprint=self.workspace.fingerprint(),
            tool_signature=self.tool_signature(),
            built_at=now(),
        )

    def _apply_prefix_state(self, prefix_state):
        """替换当前前缀对象与纯文本缓存。"""
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        """在工作区或工具集变化时刷新 prompt 前缀。"""
        #读取当前实例中的前缀状态的文本哈希和工作区哈希
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)
        #读取当前实例中的前缀状态里保存的旧工具签名,后面用来判断工具集合是否变了
        previous_tool_signature = getattr(getattr(self, "prefix_state", None), "tool_signature", None)


        #这里取当前的工作区快照
        refreshed_workspace = WorkspaceContext.build(self.root)
        #将当前工作区的快照指纹取出来
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        #这里取当前这一次工具集合的签名,后面和旧工具签名比较
        current_tool_signature = self.tool_signature()


        #如果工作区有变化或者有强制刷新限制,则更新实例中工作区的快照
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        #如果当前工具签名和旧工具签名不一致,说明工具集合发生了变化
        tools_changed = current_tool_signature != previous_tool_signature

        
        if workspace_changed:
            self.workspace = refreshed_workspace
        #然后如果工作区的快照都因为工作区有变化或者force刷新限制，则prefixd状态需要重建
        #这里额外把 tools_changed 也算进去,这样工具集合变化时也会重建 prefix 状态
        prefix_state = self.build_prefix() if workspace_changed or tools_changed or force or previous_hash is None else self.prefix_state
        #这里是判断重建后的prefix状态哈希是否和一开始取出来的previous_hash有不一致,代表了工作区以及其他地方是否被改变了
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            #有的话直接将重建的perfix状态赋给prefixd状态
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        #返回的是一个字典,包含工作区和整体prefix是否被改变的bool值
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        """返回给模型看的 memory 面板文本。"""
        return self.memory.render_memory_text()

    def history_text(self):
        """把 session history 渲染成人类可读摘要。"""
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name):
        """读取 feature flag，未知开关默认关闭。"""
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        """兼容旧调用方，只返回 prompt 文本本身。"""
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        """向 session history 追加一条事件并控制总长度。"""
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)


    '''下面是和敏感环境变量相关的几个函数,主要是为了在trace和report中对敏感环境变量进行脱敏处理,以及提供一个接口让用户显式配置哪些环境变量是敏感的'''
    @staticmethod
    def looks_sensitive_env_name(name):
        """基于名称模式判断环境变量是否像敏感项。"""
        upper = str(name).upper()
        #实际上就是三种模糊匹配方式
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        """判断环境变量名是否应被视为敏感。"""
        #这段代码的逻辑就是对传入的name(key)进行判断敏感性与否
        upper = str(name).upper()
        #判断的依据是依次转换为大写后,通过提前设定好的敏感变量名列表和模糊匹配模式
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        """返回显式配置的敏感环境变量及其当前值。"""
        items = [
            (name, value)
            for name, value in os.environ.items()
            if str(name).upper() in self.secret_env_names and value
        ]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        """扫描环境变量名，找出名称上疑似敏感的变量。"""
        # 扫描当前进程全部环境变量，挑出名称像敏感变量 + 值非空的
        items = [
            (name, value)
            for name, value in os.environ.items()
            if self.is_secret_env_name(name) and value  # 名称命中（显式配置或模式匹配）且值不为空
        ]
        # 按变量名排序，保证返回值确定，方便上层比较和测试
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        """返回显式配置敏感环境变量的简表。"""
        names = [name for name, _ in self.configured_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def detected_secret_env_summary(self):
        """返回自动识别出的敏感环境变量简表。"""
        names = [name for name, _ in self.detected_secret_env_items()]
        return {
            "secret_env_count": len(names),
            "secret_env_names": names,
        }

    def redact_text(self, text):
        """对文本中的显式 secret 值做统一脱敏。"""
        text = str(text)
        #self.detected_secret_env_items()这个函数返回的是一个列表，列表的元素是敏感环境变量的名称和值
        #按照列表中每个value值的长度进行排序的，从长到短
        #然后依次遍历这些value值,在文本中将它们替换为REDACTED_VALUE,这样就可以保证无论这个敏感环境变量的值是什么样子的,只要它出现在文本中,都会被替换为REDACTED_VALUE,从而达到脱敏的目的
        #那从长到短的排序是为了防止某些敏感环境变量的值是另一些敏感环境变量值的子串的情况,如果不排序,就可能先把短的那个子串替换了,导致长的那个敏感环境变量值无法被正确替换掉
        for _, value in sorted(self.detected_secret_env_items(), key=lambda item: len(item[1]), reverse=True):
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        """递归脱敏报告、trace 等结构化工件。"""
        #首先解释一下这个key是什么东西,这个是检查dict或者其他各种类型的key，如果key是敏感的，就返回REDACTED_VALUE
        #但是调用redact_artifact的时候，第一层一般是emit_trace和report的时候调用
        #传入的value顶层结构是 trace 事件的字段名（payload, event、task_id、tool_steps 等），不可能是敏感 key，所以不需要判定。
        #但是顶层字段下面的字段名就需要判断,于是每一个字段都再次递归调用redact_artifact判断,如果字段名是敏感的就返回REDACTED_VALUE,不需要继续递归了
        #如果不是的话,这个字段下面还有结构,就继续递归调用,如果不是,就到最后按照原样返回value就可以了
        if key and self.is_secret_env_name(key):
            #如果这里if key判断通过就说明是递归进来的dict或者有下层结构的类型了,这里是对他的某一个下层字段在判断是否敏感,需要对key进行敏感判断,如果key是敏感的就直接返回REDACTED_VALUE,不需要继续递归了
            #所以其实这里就是redact_artifact最本质的逻辑了,如果key是敏感的就直接返回REDACTED_VALUE,不需要继续递归了
            #其他的代码段只是为了继续递归下去
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self.redact_artifact(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            #这里的key沿用的是上面那个key
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [self.redact_artifact(item, key=key) for item in value]
        if isinstance(value, str):
            redacted = self.redact_text(value)
            return redacted
        return value

    def shell_env(self):
        """只暴露白名单环境变量给 shell 工具。"""
        env = {
            name: os.environ[name]
            for name in self.shell_env_allowlist
            if name in os.environ
        }
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        #返回的是一个字典,对用命令:系统执行路径
        return env

    def prompt_metadata(self, user_message, prompt):
        """保留旧接口，返回 prompt 对应的 metadata。"""
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        """统一生成 prompt，并附带缓存与 resume 元数据。"""
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_manager.build(user_message)
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),  # prefix 这部分文本的字符数
                "workspace_chars": len(self.workspace.text()),  # 当前工作区快照文本的字符数
                "memory_chars": len(self.memory_text()),  # 当前 memory 面板文本的字符数
                "history_chars": len(self.history_text()),  # 当前 session history 摘要的字符数
                "request_chars": len(user_message),  # 当前用户输入的字符数
                "tool_count": len(self.tools),  # 当前可用工具数量
                "workspace_docs": len(self.workspace.project_docs),  # workspace 文档摘要条目数
                "recent_commits": len(self.workspace.recent_commits),  # 最近提交数量
                "prefix_hash": self.prefix_state.hash,  # 当前 prefix 的稳定哈希
                "prompt_cache_key": self.prefix_state.hash,  # prompt cache 使用的 key
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,  # 工作区指纹
                "tool_signature": self.prefix_state.tool_signature,  # 当前工具集签名
                "workspace_changed": refresh["workspace_changed"],  # 这轮工作区是否变化
                "prefix_changed": refresh["prefix_changed"],  # 这轮 prefix 是否变化
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),  # 模型客户端是否支持 prompt cache
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),  # 当前恢复状态
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),  # 被判定失效的旧摘要数量
                "stale_paths": list(self.resume_state.get("stale_paths", [])),  # 被标记为过期的文件路径
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),  # runtime 身份不一致的字段
            }
        )
        metadata.update(self.detected_secret_env_summary())#记录自动识别出的敏感环境变量的简表到metadata中
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        """向 trace.jsonl 追加一条结构化事件。"""
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        """抓取工作区文件内容摘要，用于工具前后 diff。"""
        snapshot = {}
        #从工作区根目录 self.root 开始，递归遍历所有文件和目录。
        #rglob("*") 的意思差不多就是“把整个工作区里的所有路径都扫一遍”。
        for path in self.root.rglob("*"):
            #把一个绝对路径先变成相对工作区根目录的路径，
            #再拆成“每一级目录名/文件名”的元组，返回类型是 tuple[str, ...]
            #如果不在root下面,会抛出错误直接下一个文件/文件夹
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            #如果在IGNORED_PATH_NAMES中，则忽略
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            #文件夹也直接跳过
            if not path.is_file():
                continue
            # 读取文件字节并计算 sha256，把“相对路径 -> 内容哈希”记进快照；单个文件读取失败时直接跳过，不影响整体扫描。
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        """比较工具执行前后的工作区变化。"""
        changed_paths = []
        summaries = []
        #取一个并集
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            #运行到这行说明哈希发生了变化
            changed_paths.append(path)
            #判断哪些是被创建了,哪些被删除了
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                #这条分支代表双方都有,但是被改了内容
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        """为当前运行状态创建可恢复 checkpoint。"""
        #将当前session中的checkpoint拿出来
        state = self.checkpoint_state()
        #拿出当前的current_id对应的checkpoint快照本体
        current = self.current_checkpoint()
        #生成随机checkpoint_id
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        #将当前memory中的working中的recent_files拿出来,算出每个的哈希值,以字典的形式更新进新的checkpoint快照
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        #以下字段添加进新的checkpoint快照,都是当场现取的状态
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
        #更新刚才写的checkpoint进入session
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        #状态机也要更新
        task_state.checkpoint_id = checkpoint_id
        #顺便更新session的runtime_identity
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        #写入磁盘
        self.session_path = self.session_store.save(self.session)
        return checkpoint

    def infer_next_step(self, task_state):
        """基于最近 history 猜测 checkpoint 里的下一步描述。"""
        if task_state.status == "completed":
            return "No next step recorded."
        if task_state.stop_reason == "step_limit_reached":
            return "Resume from the latest checkpoint and continue the task."
        if task_state.last_tool:
            return f"Decide the next action after {task_state.last_tool}."
        return "Continue the task from the latest checkpoint."

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return
        # 把绝对路径转化为工作区内的相对路径
        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；
        # write/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            #三种都需要做state中rencent_files的更新
            self.memory.remember_file(canonical_path)
        #如果是读文件,则需要生成摘要,更新state中file_summaries和episodic笔记
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            #显示的删除state中的指定path的file_summary
            self.memory.invalidate_file_summary(canonical_path)

    def record_process_note_for_tool(self, name, metadata):
        """把工具执行异常态写成 process note。"""
        status = str(metadata.get("tool_status", "")).strip()
        #只有status为partial_success, error, rejected时才记录process note,往下推进,不然返回错误
        if status not in {"partial_success", "error", "rejected"}:
            return
        #判断哪些文件被修改,形成一个列表
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        #拼成一个多行字符串
        path_text = ", ".join(affected_paths) or "workspace"
        #根据不同的状态生成不同的提示
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        # 把这次工具执行的异常态/拒绝态沉淀成一条 process note，供后续 relevant memory 和恢复上下文参考。
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        """判断某条 durable memory 候选为何应被拒绝。"""
        # 先把候选文本规范成去掉首尾空白的字符串，避免后面判断被 None 或额外空格干扰。
        text = str(note_text or "").strip()
        # 再准备一份全小写版本，方便后面做前缀匹配时忽略大小写差异。
        lowered = text.lower()
        # 完全空白的候选没有沉淀价值，直接标记为空内容。
        if not text:
            return "empty"
        # 如果文本里已经出现脱敏占位符，或者长得像敏感信息形状，就不允许进入 durable memory。
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        # 这些前缀更像 checkpoint / 运行现场里的临时状态，不适合作为长期记忆保存。
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        # 如果候选文本就是以这些“临时任务状态”前缀开头，就拒绝它，避免把运行中间态写进 durable memory。
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        # 如果文本看起来像 stdout/stderr/traceback 这类噪声输出，或者长度过长，也认为不适合进入长期记忆。
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        """从最终回答中抽取应晋升为 durable memory 的事实。"""
        user_text = str(user_message or "")
        # 先看用户这一轮有没有显式表达“记住/保存/沉淀”之类的意图；如果没有，就直接跳过 durable 提升流程。
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        # promotions 用来收集本轮最终准备写入 durable memory 的候选条目。
        promotions = []
        # rejections 用来收集本轮被判定不适合进入 durable memory 的候选及其拒绝原因。
        rejections = []
        # 把最终回答按行拆开，逐行检查有没有符合 durable memory 模式的句子。
        for line in str(final_answer or "").splitlines():
            # 先去掉这一行首尾空白，便于后面统一做模式匹配。
            text = line.strip()
            # 空行直接跳过；如果这一行里已经含有脱敏占位符，也不允许进入 durable memory。
            if not text or REDACTED_VALUE in text:
                continue
            # 依次尝试每一种 durable memory 模式，看看这行更像“项目约定 / 决策 / 依赖 / 偏好”中的哪一类。
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                # 用当前模式去匹配这一行；只有整行命中对应前缀格式时，才继续往下抽取正文。
                match = pattern.match(text)
                # 当前模式没命中，就继续尝试下一个 topic 的模式。
                if not match:
                    continue
                # group(1) 是模式里冒号后面真正要沉淀的正文内容，再做一次 strip 去掉首尾空白。
                note_text = match.group(1).strip()
                # 只有正文非空时，才值得进入后续的 durable 候选筛选逻辑。
                if note_text:
                    # 检查这条候选是否应该被拒绝，例如内容太噪、像临时状态、像敏感信息等。
                    reason = self.reject_durable_reason(note_text)
                    # 如果有拒绝原因，就把“topic:reason”记进 rejections，并停止尝试这一行的其他 topic。
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    # 通过筛选的候选，按 (topic, note_text) 的形式加入 promotions，等待后续真正写入 durable store。
                    promotions.append((topic, note_text))
                # 这一行一旦命中了某个 durable 模式，无论最后是晋升还是拒绝，都不再继续尝试其他模式。
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        """执行 durable memory 晋升，并记录接受/拒绝结果。"""
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        # 把前面筛出来的 (topic, note_text) 候选真正写入 durable memory；
        # 返回的 promoted 是本次成功新增/更新的条目，superseded 是因此被替换掉的旧条目。
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def ask(self, user_message):
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。

        为什么存在：
        `ask()` 是整个 runtime 的总调度器。它把“用户提一个请求”扩展成一条
        可持续推进的控制循环：记录会话、组 prompt、调用模型、执行工具、
        写 trace/report、更新状态，直到模型给出最终答案或系统主动停下。

        输入 / 输出：
        - 输入：`user_message`，即用户这一次的任务描述
        - 输出：字符串形式的最终回答；如果中途达到步数上限或重试上限，
          返回的是一条停止原因说明

        在 agent 链路里的位置：
        它是 CLI 和底层工具/模型之间的核心桥梁。CLI 收到用户输入后基本只做
        一件事：调用 `agent.ask()`。而 `ask()` 内部再去驱动 `ContextManager`
        组 prompt、`model_client.complete()` 调模型、`run_tool()` 执行动作。
        如果新人想理解 Hepilot 是怎么“从一句话跑成一个 agent 流程”的，
        这里就是最关键的入口。
        """
        # 记录这次运行的起始时刻，后面用来统计本轮处理耗时。
        run_started_at = time.monotonic()
        # 把当前用户请求写入 memory 里的 task_summary，供后续记忆和摘要更新使用。
        self.memory.set_task_summary(user_message)
        # 把这条用户消息正式写进 session history。
        self.record({"role": "user", "content": user_message, "created_at": now()})

        # 为这次任务创建一个新的 TaskState，带上 run_id、task_id 和用户请求。
        task_state = TaskState.create(run_id=self.new_run_id(), task_id=self.new_task_id(), user_request=user_message)
        # 从当前 session 的 resume_state 里取出恢复状态，写回 task_state，方便后续知道是否是续跑。
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        # 保存当前正在执行的任务状态，供后续流程继续引用。
        self.current_task_state = task_state
        # 启动一个 run 目录，用来落盘这次任务的审计工件和 trace。
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0     # 真正执行了工具的次数
        attempts = 0       # 模型被调用的轮次（一轮可能因为输出非法而重试，不一定消耗 tool_step）
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)  # 调用次数上限，比 max_steps 宽松，防止模型反复输出非法内容导致死循环

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            #单次状态机的调用模型请求次数+1,并且落盘记录
            task_state.record_attempt()
            self.run_store.write_task_state(task_state)
            #这里开始构建提示词并记时间
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = self._build_prompt_and_metadata(user_message)
            #记录下提示词构建完毕时的状态和细节
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            '''接下来需要分情况讨论关于checkpoint'''
            #如果是CHECKPOINT_PARTIAL_STALE_STATUS
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                #这里要将状态机写磁盘的原因是刚才在创建checkpoint的过程中,task_state.checkpoint_id = checkpoint_id,状态机的状态也改变了
                self.run_store.write_task_state(task_state)
                #做checkpoint_created trace追踪
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",#创建原因
                    },
                )
            #如果是CHECKPOINT_WORKSPACE_MISMATCH_STATUS
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                #先做trace追踪,告诉监控器,是在哪些运行环境出现不匹配的错误
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                #创建新的checkpoint快照
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                self.run_store.write_task_state(task_state)
                #做checkpoint_created trace追踪
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            #如果构建提示词的过程中出现了压缩日志
            #说明信息已丢失 —— 模型接下来看到的不是完整上下文，而是经过压缩的版本。如果后续运行产生重要结果后崩溃，恢复时需要知道当时上下文已经被裁剪过。
            if prompt_metadata.get("budget_reductions"):
                #创建一个checkpoint
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="context_reduction")
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            #在模型调用前记录trace,记录下在调用模型 API 前记录一条观测埋点，标记当前是第几次请求、已执行了多少工具步数，以及 prompt 缓存 key。
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            #默认不使用缓存
            prompt_cache_key = None  # prompt 稳定前缀的 sha256，后端用它命中 KV cache
            prompt_cache_retention = None  # 缓存保留策略，None 表示本次请求不使用缓存
            if getattr(self.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            #开始对模型的回复记录时间点
            model_started_at = time.monotonic()
            #实际发送请求的接口调用
            raw = self.model_client.complete(
                prompt,  # 完整的 prompt 文本
                self.max_new_tokens,  # 模型最大输出 token 数
                prompt_cache_key=prompt_cache_key,  # 稳定前缀的 sha256，用于缓存命中
                prompt_cache_retention=prompt_cache_retention,  # 缓存保留策略
            )
            #记录complete的结果的元数据记录进Hepilot实例的last_completion_metadata属性里,以供后续使用,方便统一写入 report 和 trace。
            #同时也将prompt_metadata更新,赋值给Hepilot实例的self.last_prompt_metadata
            completion_metadata = dict(getattr(self.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata

            #kind：模型输出被解析后的控制类型，可能是 "tool"、"final"、"retry"。
            #payload：和这个类型对应的数据；工具调用时是工具 payload，最终回答时是文本，重试时是格式纠错提示。
            kind, payload = self.parse(raw)
            # 记录这次模型输出被解析成了哪种动作，以及模型调用耗时和 completion 元数据。
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )
            #如果判断本次调用为工具调用
            if kind == "tool":
                #将工具的调用次数+1
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                # 先把模型原始 tool call 回写到 history，保证下一轮能看到完整因果链。
                self.record({"role": "assistant", "content": raw, "created_at": now()})
                #记录状态机中工具调用的相关信息,以供后续流程使用
                task_state.record_tool(name)
                #记录工具调用开始的时间戳
                tool_started_at = time.monotonic()
                #工具执行的核心函数,传入工具名称和参数,返回工具执行结果
                result = self.run_tool(name, args)
                # 把这次工具调用的名称、参数和结果写进 session history，后续 history 压缩和恢复上下文都会用到。
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                #模型执行完了之后将刚才task_state.record_tool(name)的执行结果落盘存到task_state中
                self.run_store.write_task_state(task_state)
                #再追加一条trace,代表tool执行完毕,记录工具执行的相关信息和耗时,以及工具执行结果的元数据
                self.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(self._last_tool_result_metadata or {}),
                    },
                )
                #创建一个checkpoint,记录工具执行完毕的状态,以供后续恢复使用
                checkpoint = self.create_checkpoint(task_state, user_message, trigger="tool_executed")
                #落盘状态机,这一次的落盘是为了刚才create checkpoint过程中task_state.checkpoint_id = checkpoint_id这个状态的改变
                self.run_store.write_task_state(task_state)
                #再记录一条trace,代表checkpoint的创建,记录checkpoint的相关信息和创建原因
                self.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                #如果是重试,记录进history,落盘状态机,并且继续下一轮循环
                self.record({"role": "assistant", "content": payload, "created_at": now()})
                continue

            #如果走到这条分支,就说明模型已经给出了最终的答案
            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            #更新状态机为最新的状态
            task_state.finish_success(final)
            # 尝试把这轮问答里长期有价值、值得跨会话保留的信息提升到 durable memory。
            self.promote_durable_memory(user_message, final)
            # 创建一个检查点，用于保存当前对话的状态，以便后续恢复使用。触发原因是“run_finished”，表示这是在整个运行结束时创建的检查点。
            checkpoint = self.create_checkpoint(task_state, user_message, trigger="run_finished")
            self.run_store.write_task_state(task_state)
            self.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            #追踪一条trace,算是结束的一条记录,记录这次运行的状态、停止原因、最终答案和总耗时
            self.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            #将最后的report生成和落盘
            self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
            return final

        # 走到这里说明主循环是因为达到某个上限而退出；如果是 attempts 先耗尽，表示模型多次输出格式不合规。
        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            # 否则就是 tool_steps 先打满，表示虽然可能执行了工具，但始终没能在步数上限内收敛出最终答案。
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        #记录history
        self.record({"role": "assistant", "content": final, "created_at": now()})
        #记录持久化记忆
        self.promote_durable_memory(user_message, final)
        #创建一个checkpoint,记录运行被迫停止的状态,以供后续恢复使用
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        #落盘状态机,这一次的落盘既包含 stop_reason/final_answer 的变化，也包含 create_checkpoint 里更新的 checkpoint_id。
        self.run_store.write_task_state(task_state)
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        #做trace,记录这次运行的状态、停止原因、最终答案和总耗时
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        #生成报告并落盘,返回结果
        self.run_store.write_report(task_state, self.redact_artifact(self.build_report(task_state)))
        return final

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        # 工具执行不是“直接调函数”，而是一条带护栏的流水线：
        # 工具是否存在 -> 参数是否合法 -> 是否重复调用 -> 是否通过审批
        # -> 真正执行 -> 更新记忆。

        #首先通过name获取工具的使用说明包,如果工具不存在,直接返回错误信息
        #现在这个tool就是这个工具的配置信息
        tool = self.tools.get(name)
        if tool is None:
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "unknown_tool",
                "security_event_type": "",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: unknown tool '{name}'"
        
        #然后这里尝试在真正执行前做同步参数校验与安全前置检查
        try:
            self.validate_tool(name, args)
        except Exception as exc:
            #将正确示例取出来
            example = self.tool_example(name)
            #拼凑错误信息
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            # 如果异常里包含“路径逃出工作区”的信号，就把这次拒绝额外标记成 path_escape 安全事件。
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            self._last_tool_result_metadata = {
                "tool_status": "rejected",  # 这次工具调用在执行前就被拒绝了
                "tool_error_code": "invalid_arguments",  # 拒绝原因是参数不合法
                "security_event_type": security_event_type,  # 如果命中了安全相关异常，就记录对应的安全事件类型
                "risk_level": "high" if tool["risky"] else "low",  # 当前工具按定义属于高风险还是低风险
                "read_only": not tool["risky"],  # 这个工具是否按只读工具看待
                "affected_paths": [],  # 这次拒绝发生在执行前，因此没有实际影响到任何路径
                "workspace_changed": False,  # 工作区没有发生变化
                "diff_summary": [],  # 没有文件改动，因此也没有 diff 摘要
            }
            return message
        #走到这里说明合法性校验通过了,现在检测是否重复调用
        if self.repeated_tool_call(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "repeated_identical_call",
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        # 如果这是高风险工具且审批没有通过，就在真正执行前拒绝它，并记录这次拒绝的审计元数据。
        if tool["risky"] and not self.approve(name, args):
            self._last_tool_result_metadata = {
                "tool_status": "rejected",
                "tool_error_code": "approval_denied",
                "security_event_type": "read_only_block" if self.read_only else "approval_denied",
                "risk_level": "high",
                "read_only": False,
                "affected_paths": [],
                "workspace_changed": False,
                "diff_summary": [],
            }
            return f"error: approval denied for {name}"
        #取一个当前工作区的快照,意思是一个字典,包含root下所有文件的哈希值
        before_snapshot = self.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            #真正的执行工具的一层,调用toolkit中定义好的函数,除了shell命令,失败都会抛出异常
            result = clip(tool["run"](args))
            #根据risky的值,判断是否需要更新快照
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            #判断哪些文件被修改了,哪些被创建,那些被删除
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            #根据快照变化判断的结果给workspace_changed赋值
            workspace_changed = bool(affected_paths)
            # 先把工具结果默认标记为成功；如果后面发现 shell 退出码非 0，再改成 error 或 partial_success。
            tool_status = "ok"
            # 默认没有错误码；只有后面识别到失败或部分成功时，才会填具体的错误分类。
            tool_error_code = ""
            #run_shell 和其他工具不一样，它即使成功返回了一段文本，
            #也不代表 shell 命令真的执行成功，所以还要额外从结果里解析 exit_code。
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", result)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            #工具执行结束之后需要更新短期记忆相关文件和信息
            self.update_memory_after_tool(name, args, result)
            #构建工具元数据
            self._last_tool_result_metadata = {
                "tool_status": tool_status,
                "tool_error_code": tool_error_code,
                "security_event_type": "",
                "risk_level": "high" if tool["risky"] else "low",
                "read_only": not tool["risky"],
                "affected_paths": affected_paths,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint": self.workspace.fingerprint(),
                "diff_summary": diff_summary,
            }
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            return result
        except Exception as exc:
            # 工具真正执行时抛异常后，先重新抓一次工作区快照；这样即使执行失败，也能判断它有没有留下部分副作用。
            after_snapshot = self.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            # 比较执行前后的快照，得出哪些路径被影响，以及可读的 diff 摘要。
            affected_paths, diff_summary = self.diff_workspace_snapshots(before_snapshot, after_snapshot)
            # 只要有受影响路径，就认为这次失败不是“纯失败”，而是伴随了工作区变化。
            workspace_changed = bool(affected_paths)
            # 如果异常里带有路径逃逸信号，就把它单独标记成安全事件，方便 trace/report 后续识别。
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            # 这里构造异常路径下的工具结果元数据：有副作用的失败记为 partial_success，没有副作用的失败记为 error。
            self._last_tool_result_metadata = {
                "tool_status": "partial_success" if workspace_changed else "error",  # 失败但留下改动算 partial_success；否则算 error
                "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",  # 根据是否有副作用给出更细的错误分类
                "security_event_type": security_event_type,  # 记录是否命中了 path_escape 这类安全事件
                "risk_level": "high" if tool["risky"] else "low",  # 当前工具本身的风险等级
                "read_only": not tool["risky"],  # 这个工具是否属于只读类别
                "affected_paths": affected_paths,  # 这次异常执行实际影响到的文件路径
                "workspace_changed": workspace_changed,  # 工作区是否真的发生了变化
                "workspace_fingerprint": self.workspace.fingerprint(),  # 异常发生后当前工作区的整体指纹
                "diff_summary": diff_summary,  # 执行前后工作区变化的摘要
            }
            # 把这次失败/部分成功的过程态写成 process note，帮助后续轮次知道“出了什么问题”。
            self.record_process_note_for_tool(name, self._last_tool_result_metadata)
            # 把异常包装成统一的错误文本返回给模型，让下一轮可以基于这份反馈调整动作。
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        """检测最近是否重复发起同一个工具调用。"""
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        # 先从 session history 里筛出所有已经执行过的 tool 事件，忽略 user/assistant 等非工具记录。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        # 如果历史里的工具调用不到两次，就不可能构成“最近连续两次都重复”的模式，直接返回 False。
        if len(tool_events) < 2:
            return False
        # 只看最近两条工具记录，判断它们是不是和当前准备执行的 name/args 完全一致。
        recent = tool_events[-2:]
        # 只有最近两条工具调用都和这次调用完全相同，才认为进入了最简单的重复调用坏循环。
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        """生成 task 级别唯一标识。"""
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        """生成 run 级别唯一标识。"""
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        """汇总当前运行的核心结果，供 report.json 使用。"""
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,  # 本次运行的唯一标识
            "task_id": task_state.task_id,  # 本次任务的唯一标识
            "status": task_state.status,  # 任务结束时的状态，例如 running/completed/failed
            "stop_reason": task_state.stop_reason,  # 这次运行是因为什么原因停下的
            "final_answer": task_state.final_answer,  # 最终返回给用户的答案文本
            "tool_steps": task_state.tool_steps,  # 实际执行过的工具步数
            "attempts": task_state.attempts,  # 模型一共被调用了多少轮
            "checkpoint_id": task_state.checkpoint_id,  # 当前关联的最后一个 checkpoint id
            "resume_status": task_state.resume_status,  # 本轮运行开始时的恢复状态
            "task_state": task_state.to_dict(),  # 完整的 task_state 快照，便于 report 直接复用
            "prompt_metadata": self.last_prompt_metadata,  # 最后一轮 prompt 组装与模型调用的元数据
            "durable_promotions": list(self.last_durable_promotions),  # 本次运行里成功提升到 durable memory 的条目
            "durable_rejections": list(self.last_durable_rejections),  # 本次运行里未通过 durable 提升的候选条目
            "durable_superseded": list(self.last_durable_superseded),  # 本次运行里被更新或替换掉的旧 durable 条目
            "redacted_env": self.detected_secret_env_summary(),  # 经过脱敏后的敏感环境变量摘要
        }

    def tool_example(self, name):
        """返回某个工具的提示示例。"""
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self, name, args)
        if name == "delegate":
            #子任务的深度不能超过最大深度
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self, args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self, args)

    def tool_search(self, args):
        return toolkit.tool_search(self, args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self, args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self, args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self, args)

    def tool_delegate(self, args):
        return toolkit.tool_delegate(self, args)

    def approve(self, name, args):
        """按照 approval policy 判断风险工具是否允许运行。"""
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容

        #如果将这次kind定义为tool,首先要满足的第一个条件是，必须有 <tool> 标签
        #其次第二个条件是<tool> 比 <final> 更靠前，或者根本没有 <final>
        #那就把这段输出当成工具调用来解析
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            #将具体的<tool>标签内容取出来
            body = Hepilot.extract(raw, "tool")
            try:
                #尝试将json解析成字典
                payload = json.loads(body)
            except Exception:
                #解析json失败就定义为重试和提示:模型返回了格式错误的工具JSON
                return "retry", Hepilot.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                #如果解析出来但是不是dict就定义为重试和提示:模型返回了格式错误的工具JSON
                return "retry", Hepilot.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                #如果解析出来但是没有name字段就定义为重试和提示:模型返回了格式正确的工具JSON,但是缺少工具名称
                return "retry", Hepilot.retry_notice("tool payload is missing a tool name")
            #从解析出来的dict中解析出参数的dict
            args = payload.get("args", {})
            #如果参数字段不存在就定义为空字典,可能不需要参数
            if args is None:
                payload["args"] = {}
            #但是如果参数不是dict的格式,那就重试,提示为默认提示
            elif not isinstance(args, dict):
                return "retry", Hepilot.retry_notice()
            return "tool", payload
        
        #这一段是在处理第二种工具调用格式：不是 <tool>JSON</tool>，而是 XML 风格的 <tool ...>...</tool>
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = Hepilot.parse_xml_tool(raw)
            if payload is not None:
                #如果返回的值存在,那就直接返回payload
                return "tool", payload
            #前一行没返回代表模型输出格式有问题
            return "retry", Hepilot.retry_notice()
        
        #这一段是在处理最终输出
        if "<final>" in raw:
            final = Hepilot.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", Hepilot.retry_notice("model returned an empty <final> answer")
        #如果前面的 <tool> / <final> 解析都没命中，那就把模型原始输出当成普通最终答案处理；
        raw = raw.strip()
        if raw:
            return "final", raw
        #但如果连这个答案都是空的，就要求模型重试。
        return "retry", Hepilot.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        """生成统一的模型重试提示。"""
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            #"{prefix}。请回复一个有效的 <tool> 调用，或者一个非空的 <final> 答案。
            # 对于多行文件，建议使用 <tool name="write_file" path="file.py"><content>...</content></tool> 格式。"
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        """兼容 XML 风格的 write_file / patch_file 工具调用。"""
        #先用正则从整段 raw 里找一个 <tool ...>...</tool> 块。
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            #如果这个大块都找不到,就直接返回None
            return None
        # 先把 <tool ...> 开始标签里的属性字符串解析成字典，例如 name/path 这类属性。
        attrs = Hepilot.parse_attrs(match.group("attrs"))
        # 再从属性字典里取出工具名；pop 会一边取值一边把 name 从 attrs 里移除，剩下的属性继续作为普通 args 使用。
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None
        body = match.group("body")
        # 从 <tool ...> 标签里取出标签内容，即工具参数,因为刚才name已经被取走了
        #先把 <tool ...> 开始标签里的属性当作初始参数，再去 body 里面找子标签参数，把它们补进去或覆盖掉。
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = Hepilot.extract_raw(body, key)
        # 把 body 内容作为参数
        body_text = body.strip("\n")
        # write_file 允许模型直接把文件正文裸写在 <tool>...</tool> 里面；如果没有显式 <content> 标签，就把整个 body 当作 content。
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        # delegate 也支持把任务描述直接写在 tool body 里；如果没有 <task> 标签，就把整个 body 当作 task。
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        """解析简单 XML 风格标签上的属性。"""
        # 用来收集解析出来的属性键值对，例如 name="write_file" 会变成 {"name": "write_file"}。
        attrs = {}
        # 逐个匹配形如 key="value" 或 key='value' 的属性写法。
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            # group(1) 是属性名；group(2) 是双引号包裹的值；group(3) 是单引号包裹的值。
            # 因为两种引号只会命中一种，所以优先取 group(2)，否则取 group(3)。
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        """提取标签内容，并做首尾空白清理。"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            # 如果模型漏掉了结束标签，就把开始标签后面的剩余内容全部当作正文返回。
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        """提取标签原始内容，不做 strip, 避免把本来有意义的前后空白、换行删掉。"""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        """清空当前 session 的 history、memory 与 checkpoint。"""
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        """把工具传入路径解析为工作区内绝对路径。"""
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


MiniAgent = Hepilot
