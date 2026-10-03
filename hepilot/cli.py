"""命令行入口。

这个模块负责把“用户怎么启动 Hepilot”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""
import argparse
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
import tomllib
 
from .models import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .runtime import Hepilot, SessionStore, record_resume_failure
from .workspace import WorkspaceContext, middle


class AgentStartupError(RuntimeError):
    """启动阶段失败，且这次失败已经落成受控运行工件。

    为什么存在：
    恢复 session 失败时还没有 agent 实例，失败信息只能通过异常上抛。
    用这个专用类型，调用方才能区分“已知的、已经收口好的启动失败”和
    “真正的意外异常”：前者给用户看一句干净的话，后者才该暴露 traceback。
    """


DEFAULT_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

WELCOME_ART = (
    "        /\\___/\\\\",
    "       (  o o  )",
    "       /   ^   \\\\",
    "      /|       |\\\\",
)
WELCOME_NAME = "hepilot"
WELCOME_SUBTITLE = "local coding agent"
WELCOME_STATUS = "calm shell, ready for work"
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
LEGACY_SECRET_ENV_NAMES_VAR = "MINI_CODING_AGENT_SECRET_ENV_NAMES"
SECRET_ENV_NAMES_VAR = "HEPILOT_SECRET_ENV_NAMES"
GLOBAL_HEPILOT_CONFIG_DIR = Path.home() / ".hepilot"
GLOBAL_HEPILOT_CONFIG_PATH = GLOBAL_HEPILOT_CONFIG_DIR / "config.json"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


def _bootstrap_global_hepilot_config():
    """在首次启动时，用本机 codex 配置生成 Hepilot 的全局默认配置。"""
    payload = {"providers": {"openai": {}}}
    try:
        if CODEX_CONFIG_PATH.is_file():
            codex_config = tomllib.loads(CODEX_CONFIG_PATH.read_text(encoding="utf-8"))
            openai_provider = dict(codex_config.get("model_providers", {}).get("OpenAI", {}) or {})
            base_url = str(openai_provider.get("base_url", "")).strip()
            if base_url:
                payload["providers"]["openai"]["base_url"] = base_url
    except Exception:
        pass
    try:
        if CODEX_AUTH_PATH.is_file():
            codex_auth = json.loads(CODEX_AUTH_PATH.read_text(encoding="utf-8"))
            api_key = str(codex_auth.get("OPENAI_API_KEY", "")).strip()
            if api_key:
                payload["providers"]["openai"]["api_key"] = api_key
    except Exception:
        pass
    return payload


def _load_global_hepilot_config():
    """读取 `~/.hepilot/config.json`，首次缺失时自动按 codex 配置引导生成。"""
    if not GLOBAL_HEPILOT_CONFIG_PATH.exists():
        GLOBAL_HEPILOT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_HEPILOT_CONFIG_PATH.write_text(
            json.dumps(_bootstrap_global_hepilot_config(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    try:
        return json.loads(GLOBAL_HEPILOT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _provider_config(provider):
    """返回某个 provider 在全局 Hepilot 配置里的默认项。"""
    config = _load_global_hepilot_config()
    providers = dict(config.get("providers", {}) or {})
    return dict(providers.get(str(provider), {}) or {})


def _effective_model(args, provider):
    """解析模型名优先级：CLI 参数 > 环境变量 > 代码默认值。"""
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. provider 对应的环境变量
    # 3. 代码里的默认值
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL")
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL")
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_OLLAMA_MODEL


def _first_env(*names):
    """从多个环境变量名里返回第一个非空值。"""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _configured_secret_names(args):
    """合并默认、CLI 和环境变量里声明的敏感环境变量名。"""
    # 第一来源：内置默认名单（OPENAI_API_KEY、ANTHROPIC_API_KEY、GITHUB_PAT 等）
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    # 第二来源：CLI 参数 --secret-env-names，用户显式指定的敏感变量名
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    # 第三来源：环境变量 SECRET_ENV_NAMES_VAR （逗号分隔的变量名列表）
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if not extra_names.strip():
        # 向后兼容旧名称 LEGARY_SECRET_ENV_NAMES_VAR
        extra_names = os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    # 排序保证结果确定性，方便测试和日志输出
    return sorted(configured_secret_names)


def _build_model_client(args):
    provider = getattr(args, "provider", "openai")
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        provider_config = _provider_config("openai")
        model = _effective_model(args, provider)
        base_url = (
            getattr(args, "base_url", None)
            or os.environ.get("OPENAI_API_BASE")
            or str(provider_config.get("base_url", "")).strip()
            or DEFAULT_OPENAI_BASE_URL
        )
        api_key = os.environ.get("OPENAI_API_KEY") or str(provider_config.get("api_key", "")).strip()
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "anthropic":
        provider_config = _provider_config("anthropic")
        model = _effective_model(args, provider)
        base_url = (
            getattr(args, "base_url", None)
            or os.environ.get("ANTHROPIC_API_BASE")
            or str(provider_config.get("base_url", "")).strip()
            or DEFAULT_ANTHROPIC_BASE_URL
        )
        api_key = (
            _first_env("ANTHROPIC_API_KEY", "RIGHT_CODES_API_KEY", "OPENAI_API_KEY")
            or str(provider_config.get("api_key", "")).strip()
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    model = _effective_model(args, provider)
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)
    return OllamaModelClient(
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )


def build_welcome(agent, model, host):
    """生成交互式启动时展示的终端欢迎面板。"""
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    """根据 CLI 参数装配出一个可运行的 Hepilot 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `Hepilot`，或一个从旧 session 恢复出来的 `Hepilot`

    失败路径：
    如果用户要求恢复 session 但恢复不了（session 不存在、文件损坏，或者
    `--resume latest` 却一个 session 都没有），这里不会让原始异常直接抛穿：
    先把失败写成 task_state / trace / report 工件，再抛 `AgentStartupError`。

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先整理 secret 名单，再采集工作区快照，随后决定是恢复旧 session
    # 还是创建一个新的 Hepilot 实例。
    configured_secret_names = _configured_secret_names(args)
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(workspace.repo_root + "/.hepilot/sessions")
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
        if not session_id:
            # 用户明确要求“恢复最近的会话”，却一个 session 都没有。
            # 这里不能静默开一个新会话：那会让用户以为在续跑，实际是全新的，
            # 而且没有任何提示。所以按受控失败处理，并留下可复盘的工件。
            run_dir, message = record_resume_failure(
                workspace,
                "latest",
                FileNotFoundError(f"no saved sessions under {store.root}"),
                secret_env_names=configured_secret_names,
            )
            raise AgentStartupError(f"{message}\nFailure artifacts: {run_dir}")
    if session_id:
        try:
            return Hepilot.from_session(
                model_client=model,
                workspace=workspace,
                session_store=store,
                session_id=session_id,
                approval_policy=args.approval,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                secret_env_names=configured_secret_names,
            )
        except Exception as exc:
            # 恢复失败同样要按“受控失败”收口：写状态机、trace、report，
            # 然后抛一个专用错误类型，由 main() 用干净的消息结束进程，
            # 而不是让 traceback 崩到终端、磁盘上什么都不留。
            run_dir, message = record_resume_failure(
                workspace,
                session_id,
                exc,
                secret_env_names=configured_secret_names,
            )
            raise AgentStartupError(f"{message}\nFailure artifacts: {run_dir}") from exc
    return Hepilot(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )


def build_arg_parser():
    """声明 Hepilot CLI 的全部启动参数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama, OpenAI-compatible, or Anthropic-compatible models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic"), default="openai", help="Model backend to use.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama, OPENAI_MODEL for openai, and ANTHROPIC_MODEL for anthropic when set.",
    )
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for openai or anthropic.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser


def main(argv=None):
    """CLI 总入口，负责 one-shot 与 REPL 两种运行模式切换。"""
    args = build_arg_parser().parse_args(argv)
    try:
        agent = build_agent(args)
    except AgentStartupError as exc:
        # 启动失败已经写成受控运行工件，这里只负责用一句干净的话收口，
        # 不再把 traceback 抛给用户。
        print(str(exc), file=sys.stderr)
        return 1

    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", DEFAULT_OLLAMA_HOST)))
    print(build_welcome(agent, model=model, host=host))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input("\nhepilot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
