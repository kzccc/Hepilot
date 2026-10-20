"""输出被 max_new_tokens 截断时的解析行为。

背景：模型一次想写 8 个 Go 文件（约 7k 字符），撞上输出上限被截断在
第 8 个 <content> 中间。旧的 extract_tool_blocks 遇到没闭合的块会
`return []`，把前面 7 个**完整合法**的调用一起丢掉，然后笼统报
"malformed tool output"，模型于是原样重试，连撞三次。
"""

from hepilot.runtime import Hepilot


def test_looks_truncated_detects_unclosed_tool_block():
    assert Hepilot.looks_truncated('<tool name="write_file" path="a.go"><content>x') is True
    assert Hepilot.looks_truncated('<tool name="write_file" path="a.go"><content>x</content></tool>') is False
    assert Hepilot.looks_truncated("just prose") is False


def test_extract_keeps_complete_blocks_when_tail_is_cut():
    raw = (
        '<tool name="write_file" path="a.go"><content>A</content></tool>'
        '<tool name="write_file" path="b.go"><content>B</content></tool>'
        '<tool name="write_file" path="c.go"><content>C'
    )

    blocks = Hepilot.extract_tool_blocks(raw)

    assert len(blocks) == 2, "截断的尾巴可以被丢弃，但前面闭合的块必须留下"


def test_parse_runs_valid_calls_even_with_truncated_tail():
    raw = (
        'I will create the files.\n'
        '<tool name="write_file" path="a.go"><content>package a</content></tool>\n'
        '<tool name="write_file" path="b.go"><content>package b'
    )

    kind, payload = Hepilot.parse(raw)

    assert kind == "tool_batch"
    assert [call["args"]["path"] for call in payload] == ["a.go"]


def test_parse_reports_truncation_instead_of_malformed():
    raw = 'I will create the files.\n<tool name="write_file" path="a.go"><content>package a'

    kind, payload = Hepilot.parse(raw)

    assert kind == "retry"
    # 关键：要让模型知道是"被长度截断了"，并让它少写一点
    assert "length limit" in payload
    assert "ONE <tool> call" in payload
    assert "malformed" not in payload


def test_parse_still_reports_malformed_for_real_syntax_errors():
    raw = '<tool>{"name": "write_file", "args": {broken json</tool>'

    kind, payload = Hepilot.parse(raw)

    assert kind == "retry"
    assert "malformed" in payload
