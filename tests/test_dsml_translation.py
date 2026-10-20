"""DSML（DeepSeek 原生调用方言）要被翻译成工具调用，而不是打回重试。

背景（"测试2"会话，提问"用子agent读一下这个仓库的功能"）：
10 条 assistant 消息里 8 条是 retry、0 次工具执行，两次 run 都被用户取消。
原文（从流式增量重组）全都是：

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="delegate">
    <｜｜DSML｜｜ parameter name="task" string="true">...</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>

上一版把这种输出一律判成 foreign 打回重试，模型不知道用什么替代，
于是一遍遍原样重发。现在改成翻译成正常工具调用。
"""

import pytest

from hepilot.runtime import Hepilot

# 真实样本：来自 20261020-143337-6c719f 的第 2 段响应
SAMPLE = (
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>\n"
    '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="delegate">\n'
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="task" string="true">'
    "阅读仓库 D:\\Documents\\Hepilot测试仓库 的代码，总结这个项目的主要功能。"
    "</\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="max_steps" string="false">25'
    "</\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
    "</\uff5c\uff5cDSML\uff5c\uff5c invoke>\n"
    "</\uff5c\uff5cDSML\uff5c\uff5c calls>"
)

# 真实样本：被截断的尾巴（第 3 段响应）
TRUNCATED = (
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>\n"
    '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="delegate">\n'
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="task" string="true">Inspect README.md'
    "</\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="max_steps" string="false">25'
    "</\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
    "</\uff5c\uff5cDSML\uff5c\uff5c invoke>\n"
    "</\uff5c\uff5cDSML\uff5c\uff5c ca"
)


def test_dsml_is_translated_into_a_tool_call():
    kind, payload = Hepilot.parse(SAMPLE)

    assert kind == "tool_batch"
    assert len(payload) == 1
    call = payload[0]
    assert call["name"] == "delegate"
    assert "Hepilot测试仓库" in call["args"]["task"]
    assert call["args"]["max_steps"] == "25"
    # 每个调用都要有稳定 id，供 history 与 tool result 配对
    assert call["id"].startswith("toolu_")


def test_truncated_dsml_still_yields_the_call():
    kind, payload = Hepilot.parse(TRUNCATED)

    assert kind == "tool_batch"
    assert payload[0]["name"] == "delegate"
    assert payload[0]["args"]["task"] == "Inspect README.md"


def test_multiple_dsml_invokes_become_a_batch():
    raw = (
        "<\uff5c\uff5cDSML\uff5c\uff5c calls>"
        '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="read_file">'
        '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="path" string="true">README.md'
        "</\uff5c\uff5cDSML\uff5c\uff5c parameter>"
        "</\uff5c\uff5cDSML\uff5c\uff5c invoke>"
        '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="list_files">'
        '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="path" string="true">.'
        "</\uff5c\uff5cDSML\uff5c\uff5c parameter>"
        "</\uff5c\uff5cDSML\uff5c\uff5c invoke>"
        "</\uff5c\uff5cDSML\uff5c\uff5c calls>"
    )

    kind, payload = Hepilot.parse(raw)

    assert kind == "tool_batch"
    assert [call["name"] for call in payload] == ["read_file", "list_files"]


def test_halfwidth_pipe_variant_is_also_handled():
    raw = (
        "<|DSML| invoke name=\"list_files\">"
        '<|DSML| parameter name="path" string="true">.'
        "</|DSML| parameter>"
        "</|DSML| invoke>"
    )

    kind, payload = Hepilot.parse(raw)

    assert kind == "tool_batch"
    assert payload[0]["args"]["path"] == "."


def test_unparseable_foreign_dialect_still_goes_back_for_retry():
    raw = '<antml:invoke name="read_file">{"path": "a"}</antml:invoke>'

    kind, payload = Hepilot.parse(raw)

    assert kind == "retry"
    assert "foreign tool-call dialect" in payload


def test_plain_prose_is_unaffected():
    kind, payload = Hepilot.parse("这个项目是一个 gin 的 MVC 骨架。")

    assert kind == "final"
