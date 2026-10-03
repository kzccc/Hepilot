"""`Hepilot` 对外暴露的稳定 API 入口。

这个文件只负责重导出最常用的 CLI、runtime、workspace 与模型适配对象，
这样测试和外部调用方可以从 `Hepilot` 顶层统一导入。
"""

from .cli import build_agent, build_arg_parser, build_welcome, main
from .models import AnthropicCompatibleModelClient, FakeModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .runtime import MiniAgent, Hepilot, SessionStore
from .workspace import WorkspaceContext

__all__ = [
    "AnthropicCompatibleModelClient",
    "FakeModelClient",
    "Hepilot",
    "build_agent",
    "build_arg_parser",
    "build_welcome",
    "main",
    "MiniAgent",
    "OllamaModelClient",
    "OpenAICompatibleModelClient",
    "SessionStore",
    "WorkspaceContext",
]
