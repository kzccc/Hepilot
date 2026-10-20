"""Web 接口层：原生目录选择、重命名、轨迹折叠。"""

import pytest

from hepilot import web as web_mod
from hepilot.session_store import SessionStore
from hepilot.web import build_trajectory


# --------------------------------------------------------- 原生目录选择


class _FakeRun:
    def __init__(self, stdout):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def test_pick_folder_returns_last_stdout_line(monkeypatch):
    monkeypatch.setattr(web_mod.subprocess, "run", lambda *a, **k: _FakeRun("noise\nD:\\repo\\proj\n"))
    assert web_mod.pick_folder_native() == "D:\\repo\\proj"


def test_pick_folder_passes_initial_path_via_env(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["env"] = kwargs.get("env") or {}
        captured["script"] = args[-1]
        return _FakeRun("")

    monkeypatch.setattr(web_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no folder selected"):
        web_mod.pick_folder_native("D:\\")
    # 盘符根以反斜杠结尾，绝不能拼进脚本字符串
    assert captured["env"]["HEPILOT_PICK_INITIAL"] == "D:\\"
    assert "D:\\'" not in captured["script"]


def test_pick_folder_raises_when_user_cancels(monkeypatch):
    monkeypatch.setattr(web_mod.subprocess, "run", lambda *a, **k: _FakeRun(""))
    with pytest.raises(RuntimeError, match="no folder selected"):
        web_mod.pick_folder_native()


# ------------------------------------------------------------- 重命名


def test_rename_keeps_id_and_sets_name(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create({"id": "s1", "workspace_root": str(tmp_path / "repo")})

    store.update_header("s1", name="我的会话")

    meta = store.stat("s1")
    assert meta["id"] == "s1"          # id 仍然是唯一标识
    assert meta["name"] == "我的会话"    # name 只是展示名


def test_blank_name_falls_back_to_id(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.create({"id": "s1", "workspace_root": str(tmp_path / "repo")})
    store.update_header("s1", name="")

    assert store.stat("s1")["name"] == ""


# --------------------------------------------------------------- 轨迹


def seed(store, session):
    store.append_event(session, "user_message", {"text": "看看 readme"}, run_id="r1")
    store.append_event(session, "assistant_message", {"kind": "tool", "text": "<tool>"}, run_id="r1")
    store.append_event(
        session,
        "tool_executed",
        {"tool_use_id": "t1", "name": "read_file", "args": {"path": "README.md"}, "result": "hello"},
        run_id="r1",
    )
    store.append_event(session, "assistant_message", {"kind": "final", "text": "看完了"}, run_id="r1")
    store.append_event(
        session,
        "run_finished",
        {
            "status": "completed",
            "stop_reason": "final_answer_returned",
            "final_answer": "看完了",
            "run_duration_ms": 1234,
        },
        run_id="r1",
    )
    store.append_event(session, "user_message", {"text": "第二个问题"}, run_id="r2")


def test_trajectory_folds_events_into_turns(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = {"id": "s1", "workspace_root": str(tmp_path / "repo")}
    store.create({"id": "s1", "workspace_root": str(tmp_path / "repo")})
    seed(store, session)

    trail = build_trajectory(store, "s1")

    assert len(trail["turns"]) == 2
    first = trail["turns"][0]
    assert first["index"] == 1
    assert first["prompt"] == "看看 readme"
    assert first["answer"] == "看完了"
    assert first["status"] == "completed"
    assert first["durationMs"] == 1234
    # 第二轮还没结束
    assert trail["turns"][1]["prompt"] == "第二个问题"
    assert trail["turns"][1]["answer"] == ""


def test_trajectory_unknown_session_raises(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with pytest.raises(KeyError):
        build_trajectory(store, "nope")


def test_user_message_never_carries_the_previous_run_id(tmp_path):
    """用户消息写在 TaskState 之前，不能把上一轮的 run_id 贴上去。

    贴错的后果：轨迹页里第 N 轮的步数会显示成第 N-1 轮的时间线。
    """
    from hepilot import FakeModelClient
    from hepilot.runtime import Hepilot
    from hepilot.workspace import WorkspaceContext

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".hepilot" / "sessions")
    agent = Hepilot(
        model_client=FakeModelClient(["<final>one</final>", "<final>two</final>"]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )

    agent.ask("first")
    agent.ask("second")

    events = store.read_events(agent.session["id"])
    users = [event for event in events if event["type"] == "user_message"]
    assert len(users) == 2
    assert [event["run"] for event in users] == ["", ""]

    trail = build_trajectory(store, agent.session["id"])
    runs = [turn["run"] for turn in trail["turns"]]
    assert len(set(runs)) == 2, "每一轮必须挂到各自的 run 上"
