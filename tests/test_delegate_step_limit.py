"""子 agent 步数下限 + 步数打满时的收尾兜底。

背景（"测试1"会话最新一轮）：模型调 delegate 时自己传 max_steps=12/15，
三次子 agent run 仍然 step_limit_reached——但那一刻它其实已经把
该读的文件都读完了，只是没轮次下结论，于是返回
"Stopped after reaching the step limit without a final answer"，
把整段调查工作丢掉，父 agent 只能重跑。
"""

from hepilot import FakeModelClient
from hepilot.runtime import Hepilot
from hepilot.session_store import SessionStore
from hepilot.tools import (
    DELEGATE_MAX_STEPS,
    DELEGATE_MIN_STEPS,
    DELEGATE_TOOL_SPEC,
)
from hepilot.workspace import WorkspaceContext


def make_agent(tmp_path, replies, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Hepilot(
        model_client=FakeModelClient(replies),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


# --------------------------------------------------- 子 agent 步数下限


def test_delegate_schema_advertises_roomier_default():
    assert DELEGATE_MIN_STEPS >= 25
    assert DELEGATE_TOOL_SPEC["schema"]["max_steps"].endswith(str(DELEGATE_MIN_STEPS))


def test_model_supplied_small_max_steps_is_raised_to_the_floor(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, ["<final>ok</final>"])
    captured = {}

    real = Hepilot

    class Spy(real):
        def ask(self, user_message, **kwargs):
            captured["steps"] = self.max_steps
            return "spy"

    monkeypatch.setattr("hepilot.runtime.Hepilot", Spy)

    # 模型自己传 3，实际必须被抬到下限
    agent.tool_delegate({"task": "look around", "max_steps": 3})

    assert captured["steps"] == DELEGATE_MIN_STEPS


def test_model_supplied_huge_max_steps_is_capped(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, ["<final>ok</final>"])
    captured = {}

    real = Hepilot

    class Spy(real):
        def ask(self, user_message, **kwargs):
            captured["steps"] = self.max_steps
            return "spy"

    monkeypatch.setattr("hepilot.runtime.Hepilot", Spy)

    agent.tool_delegate({"task": "look around", "max_steps": 999})

    assert captured["steps"] == DELEGATE_MAX_STEPS


# --------------------------------------------- 步数打满时的收尾兜底


def test_step_limit_still_yields_a_useful_answer(tmp_path):
    agent = make_agent(
        tmp_path,
        [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "<final>根据已读到的文件：README 存在，其余未验证。</final>",
        ],
        max_steps=1,
    )

    final = agent.ask("看看仓库里有什么")

    assert "README 存在" in final
    assert "step limit" not in final
    assert agent.current_task_state.stop_reason == "step_limit_reached"


def test_salvage_rejects_foreign_tool_markup(tmp_path):
    agent = make_agent(
        tmp_path,
        [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="read_file">',
        ],
        max_steps=1,
    )

    final = agent.ask("看看仓库里有什么")

    # 救不回来就老老实实报步数打满，不能把外部方言当答案
    assert "step limit" in final


def test_salvage_not_attempted_when_no_tool_ran(tmp_path):
    agent = make_agent(
        tmp_path,
        ["<final>直接回答</final>"],
        max_steps=1,
    )

    assert agent.ask("hi") == "直接回答"
