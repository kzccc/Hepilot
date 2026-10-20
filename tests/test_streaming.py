"""流式输出与"当前模型在产出什么"的类型判定。"""

import json

from hepilot.models import _detect_api_style
from hepilot.runtime import Hepilot
from hepilot.session_store import SessionStore
from hepilot.web import SessionHandle
from hepilot.workspace import WorkspaceContext


class StreamingFake:
    """按固定块大小回吐，并调用 on_delta。"""

    def __init__(self, text, chunk=5, supports_prompt_cache=False):
        self.text = text
        self.chunk = chunk
        self.supports_prompt_cache = supports_prompt_cache
        self.last_completion_metadata = {}
        self.prompts = []
        self.received = []

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None, on_delta=None):
        self.prompts.append(prompt)
        self.received = []
        if on_delta is not None:
            for index in range(0, len(self.text), self.chunk):
                piece = self.text[index:index + self.chunk]
                self.received.append(piece)
                on_delta(piece)
        return self.text


def build(tmp_path, text, **kwargs):
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".hepilot" / "sessions")
    return Hepilot(
        model_client=StreamingFake(text, **kwargs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        max_steps=4,
    )


def test_detect_api_style_prefers_known_hosts():
    assert _detect_api_style("https://api.openai.com/v1") == "responses"
    assert _detect_api_style("https://www.right.codes/codex/v1") == "responses"
    # DeepSeek 与大多数兼容服务只有 chat/completions
    assert _detect_api_style("https://api.deepseek.com/v1") == "chat"
    assert _detect_api_style("http://127.0.0.1:8000/v1") == "chat"
    # 显式指定永远优先
    assert _detect_api_style("https://api.deepseek.com/v1", "responses") == "responses"


def test_stream_kind_transitions_from_text_to_tool_to_final():
    assert SessionHandle._stream_kind("") == "text"
    assert SessionHandle._stream_kind("思考中<to") == "text"
    assert SessionHandle._stream_kind("<tool>{\"name\":\"read_file\"}") == "tool"
    assert SessionHandle._stream_kind("<final>done</final>") == "final"
    # tool 先出现时以 tool 为准
    assert SessionHandle._stream_kind("<tool></tool><final>x</final>") == "tool"


def test_streaming_reaches_model_client_when_sink_is_set(tmp_path):
    agent = build(tmp_path, "<final>hello</final>")
    chunks = []
    agent.stream_sink = chunks.append
    agent.stream_flush = lambda: None

    agent.ask("hi")

    assert agent.model_client.received, "on_delta should have been passed through"
    # 第一块是 None：表示"新一轮模型调用开始"，供前端切新气泡。
    assert chunks[0] is None
    assert "".join(chunk for chunk in chunks if chunk) == "<final>hello</final>"


def test_no_sink_means_no_on_delta_and_no_streaming(tmp_path):
    agent = build(tmp_path, "<final>hello</final>")
    # stream_sink 为 None 时不该给 client 传 on_delta
    agent.ask("hi")

    assert agent.model_client.received == []


def test_assistant_message_carries_kind(tmp_path):
    agent = build(tmp_path, '<tool>{"name":"list_files","args":{"path":"."}}</tool>')
    agent.model_client.text = '<tool>{"name":"list_files","args":{"path":"."}}</tool>'

    agent.ask("hi")

    events = agent.session_store.read_events(agent.session["id"])
    assistant = [event for event in events if event["type"] == "assistant_message"]
    assert assistant, "expected an assistant_message event"
    assert assistant[0]["data"]["kind"] == "tool"


def test_final_kind_is_final(tmp_path):
    agent = build(tmp_path, "<final>done</final>")

    agent.ask("hi")

    events = agent.session_store.read_events(agent.session["id"])
    assistant = [event for event in events if event["type"] == "assistant_message"]
    assert assistant[-1]["data"]["kind"] == "final"
