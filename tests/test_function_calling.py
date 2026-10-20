"""function calling 注入：工具定义进 tools 字段，响应里的 tool_calls 翻译回文本协议。

设计取舍：
- 模型走结构化通道（tools / tool_calls），但内部整条链路（parse / history /
  trace / UI）建在 <tool>{...}</tool> 文本上。在客户端做单向翻译，下游零改动。
- 只有支持 tools 的后端才注入；Ollama 的 /api/generate 不支持，继续走文本协议，
  所以老路径必须保留。
"""

import json

import pytest

from hepilot import models as models_mod
from hepilot.tools import (
    BASE_TOOL_SPECS,
    DELEGATE_MAX_STEPS,
    DELEGATE_MIN_STEPS,
    DELEGATE_TOOL_SPEC,
    build_function_tools,
)
from hepilot.models import (
    FakeModelClient,
    OllamaModelClient,
    OpenAICompatibleModelClient,
    synthesize_tool_call_text,
)


def registry():
    tools = dict(BASE_TOOL_SPECS)
    tools["delegate"] = DELEGATE_TOOL_SPEC
    return tools


# ------------------------------------------------------------ 工具定义


def test_build_function_tools_covers_every_registered_tool():
    tools = build_function_tools(registry())

    assert len(tools) == 7
    names = [item["function"]["name"] for item in tools]
    assert set(names) == set(registry())


def test_every_tool_has_object_rooted_parameters():
    for item in build_function_tools(registry()):
        function = item["function"]
        assert item["type"] == "function"
        assert function["description"].strip()
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        # required 里的每个键都必须真的在 properties 里
        for key in parameters.get("required") or []:
            assert key in parameters["properties"], f"{function['name']} 缺少 {key} 定义"


def test_runtime_limits_are_visible_in_the_schema():
    """以前 timeout 的区间只写在运行时校验里，模型看不到就会撞上再重试。"""
    by_name = {item["function"]["name"]: item["function"] for item in build_function_tools(registry())}

    timeout = by_name["run_shell"]["parameters"]["properties"]["timeout"]
    assert timeout["minimum"] == 1
    assert timeout["maximum"] == 600

    steps = by_name["delegate"]["parameters"]["properties"]["max_steps"]
    assert steps["minimum"] == DELEGATE_MIN_STEPS == 50
    assert steps["maximum"] == DELEGATE_MAX_STEPS == 100


def test_unknown_tool_is_skipped():
    tools = build_function_tools({"totally_unknown_tool": {"schema": {}, "risky": False, "description": ""}})

    assert tools == []


# ------------------------------------------------- 响应翻译回文本协议


def test_single_tool_call_becomes_a_text_block():
    slots = {0: {"name": "read_file", "arguments": '{"path": "a.go", "start": 1}'}}

    text = synthesize_tool_call_text(slots)

    assert text == '<tool>{"name": "read_file", "args": {"path": "a.go", "start": 1}}</tool>'


def test_multiple_tool_calls_become_multiple_blocks():
    slots = {
        0: {"name": "list_files", "arguments": '{"path": "."}'},
        1: {"name": "read_file", "arguments": '{"path": "a.go"}'},
    }

    text = synthesize_tool_call_text(slots)

    assert text.count("<tool>") == 2
    assert text.count("</tool>") == 2


def test_truncated_arguments_become_an_unclosed_tag():
    """参数被截断时留一个未闭合开标签，让 looks_truncated() 认出来走截断提示。"""
    slots = {0: {"name": "write_file", "arguments": '{"path": "a.go", "content": "pack'}}

    text = synthesize_tool_call_text(slots)

    assert text == '<tool name="write_file">'
    assert text.count("<tool") > text.count("</tool>")


def test_nameless_call_is_dropped():
    assert synthesize_tool_call_text({0: {"name": "", "arguments": "{}"}}) == ""


def test_extract_openai_text_handles_tool_calls_only():
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "list_files", "arguments": '{"path": "."}'}}
                    ],
                }
            }
        ]
    }

    text = models_mod._extract_openai_text(data)

    assert text.startswith("<tool>")
    assert "list_files" in text


def test_extract_openai_text_keeps_preamble_and_tool_calls():
    data = {
        "choices": [
            {
                "message": {
                    "content": "我先看一下目录。",
                    "tool_calls": [{"function": {"name": "list_files", "arguments": "{}"}}],
                }
            }
        ]
    }

    text = models_mod._extract_openai_text(data)

    assert text.startswith("我先看一下目录。")
    assert "<tool>" in text


def test_anthropic_tool_use_is_translated():
    data = {
        "content": [
            {"type": "text", "text": "查一下"},
            {"type": "tool_use", "name": "search", "input": {"pattern": "TODO"}},
        ]
    }

    text = models_mod._extract_anthropic_text(data)

    assert "查一下" in text
    assert '<tool>{"name": "search", "args": {"pattern": "TODO"}}</tool>' in text


# --------------------------------------------------------- 流式分片累积


def test_stream_deltas_accumulate_arguments():
    slots = {}
    models_mod._accumulate_tool_call_deltas(slots, {"tool_calls": [{"index": 0, "function": {"name": "write_file"}}]})
    models_mod._accumulate_tool_call_deltas(slots, {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]})
    models_mod._accumulate_tool_call_deltas(slots, {"tool_calls": [{"index": 0, "function": {"arguments": '"a.go"}'}}]})

    text = models_mod._join_stream_and_tools([], slots)

    assert text == '<tool>{"name": "write_file", "args": {"path": "a.go"}}</tool>'


# ------------------------------------------------------- 请求里真的带 tools


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {"Content-Type": "application/json"}

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_chat_payload_includes_tools(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "<final>ok</final>"}}]}))

    monkeypatch.setattr(models_mod.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleModelClient(
        model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key="k",
        temperature=None, timeout=30, api_style="chat",
    )

    client.complete("prompt", 64, tools=build_function_tools(registry()))

    assert len(captured["payload"]["tools"]) == 7
    assert captured["payload"]["tool_choice"] == "auto"


def test_chat_payload_omits_tools_when_not_given(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "<final>ok</final>"}}]}))

    monkeypatch.setattr(models_mod.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleModelClient(
        model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key="k",
        temperature=None, timeout=30, api_style="chat",
    )

    client.complete("prompt", 64)

    assert "tools" not in captured["payload"]


def test_capability_flags_per_client():
    assert OpenAICompatibleModelClient(
        model="m", base_url="https://x/v1", api_key="k", temperature=None, timeout=1, api_style="chat"
    ).supports_functions is True
    assert OpenAICompatibleModelClient(
        model="m", base_url="https://x/v1", api_key="k", temperature=None, timeout=1, api_style="responses"
    ).supports_functions is False
    assert FakeModelClient([]).supports_functions is False
    assert OllamaModelClient(
        model="m", host="http://127.0.0.1:11434", temperature=None, top_p=None, timeout=1
    ).supports_functions is False
