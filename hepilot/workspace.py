"""工作区快照工具。

这个模块负责在 agent 按需读文件之前，先给它一份便宜的“仓库第一印象”。
这份快照刻意保持小而稳定：主要包含 Git 事实和少量白名单项目文档。
"""

import subprocess
import textwrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
IGNORED_PATH_NAMES = {".git", ".hepilot", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


def now():
    """返回统一格式的 UTC 时间戳，供 session/trace/report 共用。"""
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT):
    """裁剪过长文本，避免工具输出或快照内容失控膨胀。"""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    """保留首尾信息压缩长字符串，适合路径或状态行展示。"""
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


class WorkspaceContext:
    """工作区的轻量静态快照。

    它只收集 branch、status、最近提交和少量项目文档，作为 agent 启动时的
    基线上下文，而不是替代后续按需读文件。
    """

    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd, repo_root_override=None):
        """从当前目录和 Git 仓库状态构造一份工作区快照。"""
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )
        docs = {}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        """把快照渲染成 prompt prefix 可直接拼接的文本块。"""
        # 这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            """
        ).strip()

    def fingerprint(self):
        """生成快照指纹，用于判断 prefix 是否需要整体刷新。"""
        # 这个指纹用来判断仓库状态是否发生了足够大的变化，
        # 从而决定是否需要重建缓存中的 prompt prefix。
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
