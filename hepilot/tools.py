"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import os
import shutil
import signal
import subprocess
import tempfile
import textwrap
import time
from functools import partial

from .workspace import IGNORED_PATH_NAMES, clip

# 搜索的硬上限：防止把盘符根当工作区时把 run 拖死。
MAX_SEARCH_FILES = 2000
MAX_SEARCH_FILE_BYTES = 1 * 1024 * 1024
MAX_SEARCH_MATCHES = 200
SEARCH_TIME_BUDGET_SECONDS = 15.0

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}
#子代理(子agent)工具的规格说明
DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
}
#这个是当模型调用参数有问题时,就会到这里来针对性的取某个工具的用法示例,再喂给模型,帮助模型理解正确的调用方式和参数格式,从而纠正错误的调用方式
TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}


def build_tool_registry(agent):
    """基于当前 agent 能力边界构造工具注册表。"""
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        #**spec是字典解包语法
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], agent)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if agent.depth < agent.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, agent)}
    return tools


def tool_example(name):
    """返回工具调用示例，供 prompt prefix 展示。"""
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(agent, name, args):
    """在真正执行前做同步参数校验与安全前置检查。"""
    args = args or {}
    #列举文件需要检查路径是否是文件夹
    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return
    #读取文件需要检查路径是否是文件，以及行号范围是否合法
    if name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return
    #这里验证search功能的参数传入的不能是空字符串,以及传入的path不能是在越界范围
    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        agent.path(args.get("path", "."))
        return
    #如果工具是shell,不可以不传命令,同时超时设置要在范围内
    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return
    #写文件需要检查路径不能是文件夹,并且必须提供content参数(写的内容)
    if name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    # patch_file功能首先路径不能是文件夹,同时被替换文本和替换文本都不可以为空
    # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
    # 这样修改行为才是确定的，失败原因也更容易解释。
    if name == "patch_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return
    
    #子任务不可以为空
    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        return


def tool_list_files(agent, args):
    """列出目录项，输出保持小而稳定，适合模型快速浏览。"""
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(agent, args):
    """按行号范围读取 UTF-8 文本文件。"""
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(agent.root)}\n{body}"


def tool_search(agent, args):
    """优先用 `rg` 搜索，否则退回到纯 Python 扫描。

    两条路径都必须有上限：工作区一旦是盘符根（例如 D:\\），无上限的递归
    扫描会把整个 run 拖在这里几十分钟，而且没有任何进度可供外部判断。
    """
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        try:
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", str(MAX_SEARCH_MATCHES), pattern, str(path)],
                cwd=agent.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SEARCH_TIME_BUDGET_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"(search timed out after {int(SEARCH_TIME_BUDGET_SECONDS)}s; narrow the path or pattern)"
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    deadline = time.monotonic() + SEARCH_TIME_BUDGET_SECONDS
    matches = []
    scanned = 0
    truncated = False
    candidates = [path] if path.is_file() else path.rglob("*")
    for file_path in candidates:
        if time.monotonic() > deadline or scanned >= MAX_SEARCH_FILES:
            truncated = True
            break
        try:
            if not file_path.is_file():
                continue
            if any(part in IGNORED_PATH_NAMES for part in file_path.relative_to(agent.root).parts):
                continue
            # 大文件不扫：读它既慢又基本不会是我们要找的源码。
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
        except (OSError, ValueError):
            continue
        scanned += 1
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root)}:{number}:{line}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    truncated = True
                    break
        if truncated:
            break

    if matches:
        body = "\n".join(matches)
        if truncated:
            body += f"\n... (结果可能不全：已扫 {scanned} 个文件后停止)"
        return body
    note = f"(no matches; scanned {scanned} files"
    note += " — 搜索范围被上限截断)" if truncated else ")"
    return note


def _kill_tree(process):
    """杀掉整个进程树。

    `shell=True` 时子进程其实是 `cmd /c ...` 或 `sh -c ...` 包了一层，
    直接 kill 只杀掉这层壳，真正的命令会变成孤儿继续跑。
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        # 进程树杀不掉时退回单进程 kill，至少别让调用方崩。
        try:
            process.kill()
        except Exception:
            pass


def tool_run_shell(agent, args):
    """在受控环境变量白名单下执行 shell 命令，并支持中途取消。

    为什么不用 subprocess.run：
    - 它一旦进去就没办法从外部打断，用户点"停止"也停不下来。
    为什么不用 PIPE + communicate：
    - 输出超过管道缓冲区会把子进程堵死；而在 Windows 上
      `communicate(timeout)` 超时后再 `communicate()` 还可能永久阻塞。
    所以这里的做法是：输出重定向到临时文件，主循环只用 poll() 探活，
    取消/超时就 kill。这样既不堵管道，也不会卡在读取上。
    """
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    cancel = getattr(agent, "cancel_token", None)

    cancelled = False
    timed_out = False
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        process = subprocess.Popen(
            command,
            cwd=agent.root,
            shell=True,
            stdout=out_file,
            stderr=err_file,
            # POSIX 下让子进程自成一个进程组，取消时才能整组杀掉。
            start_new_session=(os.name != "nt"),
            # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
            # 目的是减少敏感信息被意外带进命令执行环境的风险。
            env=agent.shell_env(),
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if cancel is not None and cancel.is_set():
                _kill_tree(process)
                cancelled = True
                break
            if time.monotonic() >= deadline:
                _kill_tree(process)
                timed_out = True
                break
            time.sleep(0.1)

        if cancelled:
            # 取消时不需要命令输出，最多等 2 秒就走了。
            # 等太久会让前端"正在停止"卡住十几秒。
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError("cancelled by user")

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        out_file.seek(0)
        err_file.seek(0)
        stdout = out_file.read().decode("utf-8", "replace")
        stderr = err_file.read().decode("utf-8", "replace")

    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout)

    return textwrap.dedent(
        f"""\
        exit_code: {process.returncode}
        stdout:
        {stdout.strip() or "(empty)"}
        stderr:
        {stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(agent, args):
    """直接覆盖写入文本文件，并自动创建父目录。"""
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root)} ({len(content)} chars)"


def tool_patch_file(agent, args):
    """执行一次严格的精确文本替换。"""
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root)}"


def tool_delegate(agent, args):
    """创建只读子 agent 做受限调查，并把结论文本带回父 agent。"""
    if agent.depth >= agent.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")

    from .runtime import Hepilot

    child = Hepilot(
        model_client=agent.model_client,
        workspace=agent.workspace,
        session_store=agent.session_store,
        run_store=agent.run_store,
        approval_policy="never",
        max_steps=int(args.get("max_steps", 3)),
        max_new_tokens=agent.max_new_tokens,
        depth=agent.depth + 1,
        max_depth=agent.max_depth,
        read_only=True,
        secret_env_names=agent.secret_env_names,
        shell_env_allowlist=agent.shell_env_allowlist,
        user_memory_root=getattr(agent, "user_memory_root", None),
    )
    # 委派的目标是“调查”，不是“放权执行”。
    # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
    child.session["memory"]["task"] = task
    child.session["memory"]["notes"] = [clip(agent.history_text(), 300)]
    return "delegate_result:\n" + child.ask(task)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
