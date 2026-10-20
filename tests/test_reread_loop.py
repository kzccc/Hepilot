"""反复读同一批文件的坏循环。

背景（"测试1"会话最新一轮，提问"再来一次"）：
模型把同样的 7 个 Go 文件读了 80 次以上，始终不进入写文件阶段。

原因有两层：
1. history section 的 recent_window 写死 6、每条 line_limit 写死 900。
   模型一轮读 7 个文件 → 永远有一个在窗口外被折叠；120 行的 Go 文件
   约 4k 字符，900 装不下 → 它不停换 start/end 重读。
2. 每重读一轮，上一轮的内容就被挤出窗口折叠成一行，
   于是"看不到 → 再读 → 把前一读挤掉"自激放大。
   实测 collapsed_duplicate_reads 从 5 一路涨到 41，raw 97k 被压到 8.9k。
"""

from hepilot import FakeModelClient
from hepilot.runtime import READ_REPEAT_LIMIT, Hepilot
from hepilot.session_store import SessionStore
from hepilot.workspace import WorkspaceContext
from hepilot import context_manager


def make_agent(tmp_path, replies, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    for index in range(3):
        (tmp_path / f"f{index}.go").write_text("\n".join(f"line {n}" for n in range(1, 60)) + "\n", encoding="utf-8")
    return Hepilot(
        model_client=FakeModelClient(replies),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


# ------------------------------------------------ 预算与窗口

def test_history_budget_is_roomy_enough_for_a_small_repo():
    assert context_manager.DEFAULT_SECTION_BUDGETS["history"] >= 16000
    assert context_manager.DEFAULT_TOTAL_BUDGET >= 28000


def test_recent_window_scales_with_budget():
    """recent_window 不能再写死 6：预算给多少，就要能容纳多少条。"""
    source = (context_manager.__file__ and open(context_manager.__file__, encoding="utf-8").read())

    assert "recent_window = 6" not in source, "窗口不该再写死成 6"
    assert "budget // 1200" in source


# ------------------------------------------------ 重复读闸门

def test_repeated_reads_of_same_file_are_cut_off(tmp_path):
    # 每次微调 end，绕过"最近两次完全相同"那道闸——这正是真实的打转形态
    replies = [
        f'<tool>{{"name":"read_file","args":{{"path":"f0.go","start":1,"end":{10 + index * 4}}}}}</tool>'
        for index in range(10)
    ]
    replies.append("<final>done</final>")
    agent = make_agent(tmp_path, replies, max_steps=20)

    agent.ask("read f0.go")

    results = [str(item["content"]) for item in agent.session["history"] if item.get("role") == "tool"]
    cutoff = [text for text in results if "has already been read" in text]
    assert cutoff, "读到第 N 次之后必须被闸门拦住"
    # 前 READ_REPEAT_LIMIT 次仍然允许真的读
    allowed = [text for text in results if "has already been read" not in text]
    assert len(allowed) == READ_REPEAT_LIMIT


def test_different_line_ranges_are_not_deduped(tmp_path):
    """1-40 和 27-120 是两段不同内容，不能被当成重复读折叠掉。"""
    replies = [
        '<tool>{"name":"read_file","args":{"path":"f0.go","start":1,"end":20}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"f0.go","start":21,"end":40}}</tool>',
        "<final>ok</final>",
    ]
    agent = make_agent(tmp_path, replies, max_steps=20)

    agent.ask("read two ranges")

    tool_items = [item for item in agent.session["history"] if item.get("role") == "tool"]
    assert len(tool_items) == 2
    for item in tool_items:
        assert "has already been read" not in str(item["content"])
    # 两段内容都真的读上来了
    assert tool_items[0]["content"] != tool_items[1]["content"]
