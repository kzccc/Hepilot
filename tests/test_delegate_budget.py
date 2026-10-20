"""子 agent 的预算与信息完整性。

背景（"测试1"会话）：9 次 delegate 全部用来查"当前目录叫什么"。
- 2 次死于 `step_limit_reached`（默认 max_steps=3 太小）
- 其余 7 次回答了 "路径被截断成 D:\\Documents\\Hepilot测...，无法确定"
  因为父级历史被 clip 到 300 字符，路径正好断在中间
"""

from pathlib import Path

from hepilot import FakeModelClient
from hepilot.runtime import Hepilot
from hepilot.session_store import SessionStore
from hepilot.workspace import WorkspaceContext, MAX_TOOL_OUTPUT, MAX_HISTORY
from hepilot import context_manager


def make_agent(tmp_path, replies=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Hepilot(
        model_client=FakeModelClient(replies or []),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
    )


def test_delegate_default_steps_is_not_tiny():
    from hepilot.tools import DELEGATE_TOOL_SPEC, DELEGATE_DEFAULT_STEPS

    assert DELEGATE_DEFAULT_STEPS >= 20
    assert DELEGATE_TOOL_SPEC["schema"]["max_steps"].endswith(str(DELEGATE_DEFAULT_STEPS))


def test_delegate_gives_child_full_workspace_path(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, ["<final>ok</final>"])
    captured = {}

    real_hepilot = Hepilot

    class Spy(real_hepilot):
        def ask(self, user_message, **kwargs):
            # notes 是在 __init__ 之后才由 tool_delegate 写进去的，
            # 所以必须在 ask 时读，不能在 __init__ 里读。
            captured["steps"] = self.max_steps
            captured["attempts"] = self.max_attempts
            captured["notes"] = list(self.session["memory"]["notes"])
            return "spy-final"

    monkeypatch.setattr("hepilot.runtime.Hepilot", Spy)

    result = agent.tool_delegate({"task": "find the cwd", "max_steps": 5})

    assert result.startswith("delegate_result:")
    # 子 agent 的预算被抬高了，而不是沿用 3
    assert captured["steps"] == 5
    assert captured["attempts"] >= 10
    # 绝对路径原样给出，且带明确前缀
    notes = "\n".join(captured["notes"])
    assert "Workspace root (absolute, verbatim)" in notes
    assert str(agent.root) in notes


def test_delegate_history_clip_is_roomier_than_300():
    from hepilot.tools import DELEGATE_HISTORY_CLIP

    assert DELEGATE_HISTORY_CLIP >= 1000


def test_prompt_budgets_were_raised():
    assert context_manager.DEFAULT_TOTAL_BUDGET >= 20000
    assert context_manager.DEFAULT_SECTION_BUDGETS["prefix"] >= 6000
    assert context_manager.DEFAULT_SECTION_FLOORS["prefix"] >= 2000


def test_tool_output_clamps_were_raised():
    assert MAX_TOOL_OUTPUT >= 8000
    assert MAX_HISTORY >= 24000


def test_tail_clip_keeps_the_head_so_workspace_block_survives():
    text = "Workspace:\n- cwd: D:\\Documents\\Hepilot测试仓库\n" + ("x" * 500)
    clipped = context_manager._tail_clip(text, 60)

    assert len(clipped) == 60
    assert clipped.startswith("Workspace:")
    assert clipped.endswith("...")
