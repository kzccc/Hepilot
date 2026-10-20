"""模型输出别家原生函数调用方言时的处理。

背景（"测试1"会话第 3 轮）：模型把 DeepSeek 的原生调用格式直接写进了回答——

    <｜｜DSML｜｜ calls>
    <｜｜DSML｜｜ invoke name="read_file">
    <｜｜DSML｜｜ parameter name="path" string="true">repository/user_repository.go</｜｜DSML｜｜ parameter>

parse() 没认出来，于是走"裸文本 ⇒ 最终答案"的兜底，
这一轮被标记成 completed，用户收到的"回答"就是这段乱码。
"""

import pytest

from hepilot.runtime import Hepilot

DSML = (
    'I will inspect the files.\n\n'
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>\n"
    '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="read_file">\n'
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="path" string="true">main.go'
    "<\uff5c\uff5cDSML\uff5c\uff5c parameter>\n"
    "<\uff5c\uff5cDSML\uff5c\uff5c invoke>\n"
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>"
)


def test_dsml_markup_is_not_accepted_as_final_answer():
    kind, payload = Hepilot.parse(DSML)

    assert kind == "retry"
    assert "foreign tool-call dialect" in payload


def test_foreign_notice_tells_the_model_the_exact_syntax():
    _, payload = Hepilot.parse(DSML)

    assert '"name"' in payload and '"args"' in payload
    assert "<|DSML|>" in payload


@pytest.mark.parametrize(
    "raw",
    [
        '<antml:invoke name="read_file">x</antml:invoke>',
        '<function_calls><invoke name="read_file"></invoke></function_calls>',
        '<invoke name="write_file">',
        '<tool_call>{"name": "read_file"}</tool_call>',
    ],
)
def test_other_dialects_are_also_rejected(raw):
    kind, _ = Hepilot.parse(raw)
    assert kind == "retry"


def test_dsml_inside_final_tag_is_rejected():
    raw = "<final>" + DSML + "</final>"

    kind, payload = Hepilot.parse(raw)

    assert kind == "retry"
    assert "foreign" in payload


def test_plain_prose_is_still_a_final_answer():
    kind, payload = Hepilot.parse("这个仓库是 gin 的 MVC 骨架，service 层还没写完。")

    assert kind == "final"
    assert payload.startswith("这个仓库是")


def test_prose_mentioning_our_own_tool_syntax_stays_final():
    # 不要误伤：解释本项目协议的自然语言回答仍然算最终答案
    raw = '用法是 <tool>{"name": "read_file", "args": {"path": "a.go"}}</tool>，注意尖括号要完整。'

    kind, _ = Hepilot.parse(raw)

    # 这里有 <tool 但没闭合，会被当成工具批次；关键是它不能是"原样 final"
    assert kind in {"tool_batch", "retry"}
