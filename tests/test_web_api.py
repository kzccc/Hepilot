"""Web 接口层：原生目录选择、重命名、轨迹折叠、盘符与建目录。"""

import os
import threading
import time
from pathlib import Path

import pytest

from hepilot import web as web_mod
from hepilot.session_store import SessionStore
from hepilot.web import build_trajectory


# ------------------------------------------------- 盘符切换 / 新建文件夹


def test_list_drives_returns_usable_roots():
    drives = web_mod.list_drives()["drives"]
    assert drives
    for drive in drives:
        if os.name == "nt":
            assert len(drive) == 3 and drive.endswith(":\\")
            assert os.path.isdir(drive)
        else:
            assert drive == "/"


def test_list_directory_normalises_bare_drive_letter():
    if os.name != "nt":
        pytest.skip("windows only")
    data = web_mod.list_directory("C:")
    assert data["path"] == "C:\\"


def test_make_directory_creates_folder(tmp_path):
    created = web_mod.make_directory(str(tmp_path), "新建的目录")
    assert Path(created).is_dir()
    assert Path(created).name == "新建的目录"


@pytest.mark.parametrize("bad", ["", "  ", "..", ".", "a/b", "a\\b", "x:y", "c*", "q?"])
def test_make_directory_rejects_bad_names(tmp_path, bad):
    with pytest.raises(ValueError):
        web_mod.make_directory(str(tmp_path), bad)


def test_make_directory_rejects_existing(tmp_path):
    web_mod.make_directory(str(tmp_path), "dup")
    with pytest.raises(ValueError, match="已存在"):
        web_mod.make_directory(str(tmp_path), "dup")


def test_make_directory_rejects_missing_parent(tmp_path):
    with pytest.raises(ValueError, match="父目录不存在"):
        web_mod.make_directory(str(tmp_path / "nope"), "x")


# ------------------------------------------------- 前端渲染的静态约束


def _index_html():
    import hepilot

    return (Path(hepilot.__file__).parent / "web_assets" / "index.html").read_text(encoding="utf-8")


def test_kbadge_covers_every_message_kind():
    """徽标表要覆盖后端所有 msgKind。

    `stopped`（被取消的运行）曾经没在表里，查不到就回退到一个不存在的键，
    解构 undefined 抛异常，整个 renderChat 中断——会话直接一片空白。
    """
    html = _index_html()
    for kind in ("text", "tool", "final", "stopped", "retry"):
        assert f"{kind}: [" in html, f"KBADGE 缺少 {kind}"


def test_kbadge_fallback_points_at_an_existing_key():
    import re

    html = _index_html()
    # 只检查真实的兜底调用点（|| KBADGE.xxx），注释里提到旧写法不算
    assert not re.search(r"\|\|\s*KBADGE\.ok\b", html), "KBADGE 里没有 ok 这个键"
    assert "BADGE_FALLBACK" in html


def test_render_isolates_a_broken_turn():
    """单轮渲染失败不能拖垮整个对话列表。"""
    html = _index_html()
    assert "轮渲染失败" in html


def test_streaming_events_do_not_force_chat_to_bottom():
    """工具调用等高频流事件不能改变用户当前的阅读位置。"""
    import re

    html = _index_html()
    match = re.search(r"function openStream\(id, after\) \{([\s\S]*?)\n\}\n\nfunction closeStream", html)
    assert match, "找不到 openStream 实现"
    assert "scrollBottom()" not in match.group(1)


def test_expanded_tool_steps_stay_open_when_chat_rerenders():
    """追加工具事件重绘对话时，已展开的旧工具卡片仍保持展开。"""
    html = _index_html()
    assert "expandedSteps: new Set()" in html
    assert "stepKey: itemIndex" in html
    assert 'data-step-key="${step.stepKey}"' in html
    assert "state.expandedSteps.has(step.stepKey)" in html
    assert "state.expandedSteps.add(stepKey)" in html
    assert "state.expandedSteps.delete(stepKey)" in html
    assert "state.expandedSteps = new Set()" in html


def test_pending_tool_calls_render_as_one_collapsible_group():
    """最终回答产生前，本轮工具调用以一条摘要呈现并可整体展开。"""
    html = _index_html()
    assert "expandedToolGroups: new Set()" in html
    assert 'const pendingTools = turn.steps.length > 0 && turn.answerKind !== "final"' in html
    assert "tool-group-head" in html
    assert 'turn.steps.length} 个工具调用' in html
    assert "state.expandedToolGroups.has(groupKey)" in html
    assert "state.expandedToolGroups.add(groupKey)" in html
    assert "state.expandedToolGroups.delete(groupKey)" in html


def test_answers_use_safe_markdown_renderer():
    """模型回答应按安全 Markdown 渲染，而不是作为转义后的纯文本。"""
    import re

    html = _index_html()
    assert "function renderMarkdown(" in html
    assert "function safeMarkdownHref(" in html
    assert '<h${heading[1].length}>' in html and "<blockquote>" in html and "<pre><code${fence.lang" in html
    assert re.search(r"safeMarkdownHref[\s\S]{0,500}javascript:", html, re.IGNORECASE)
    assert re.search(r"renderMarkdown\(turn\.live\)", html)
    assert re.search(r"renderMarkdown\(turn\.answer\)", html)


# --------------------------------------------------------- 原生目录选择


def fake_popen(stdout="", capture=None):
    """构造一个假的 Popen：只关心 stdout 与 env。"""

    class FakePopen:
        def __init__(self, args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.killed = False
            if capture is not None:
                capture["env"] = kwargs.get("env") or {}
                capture["args"] = args

        def communicate(self, timeout=None):
            return stdout, ""

        def kill(self):
            self.killed = True

    return FakePopen


def test_run_folder_picker_returns_last_stdout_line(monkeypatch):
    monkeypatch.setattr(web_mod.subprocess, "Popen", fake_popen("noise\nD:\\repo\\proj\n"))
    assert web_mod.run_folder_picker() == "D:\\repo\\proj"


def test_run_folder_picker_passes_initial_path_via_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(web_mod.subprocess, "Popen", fake_popen("", captured))
    with pytest.raises(RuntimeError, match="no folder selected"):
        web_mod.run_folder_picker("D:\\")
    # 盘符根以反斜杠结尾，绝不能拼进脚本字符串
    assert captured["env"]["HEPILOT_PICK_INITIAL"] == "D:\\"
    assert "D:\\'" not in captured["args"][-1]


def test_run_folder_picker_raises_when_user_cancels(monkeypatch):
    monkeypatch.setattr(web_mod.subprocess, "Popen", fake_popen(""))
    with pytest.raises(RuntimeError, match="no folder selected"):
        web_mod.run_folder_picker()


def test_picker_start_is_idempotent_and_reports_result(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def fake_pick(initial="", on_start=None):
        started.set()
        release.wait(3)
        return "D:\\picked"

    monkeypatch.setattr(web_mod, "run_folder_picker", fake_pick)
    picker = web_mod.FolderPicker()
    assert picker.result()["status"] == "idle"

    assert picker.start()["status"] == "pending"
    started.wait(2)
    # 重复点击不该报错
    assert picker.start()["status"] == "pending"

    release.set()
    for _ in range(100):
        if picker.result()["status"] == "done":
            break
        time.sleep(0.02)
    assert picker.result() == {"status": "done", "path": "D:\\picked", "error": ""}


def test_picker_cancel_is_reported_as_error(monkeypatch):
    release = threading.Event()

    def fake_pick(initial="", on_start=None):
        release.wait(3)
        return "x"

    monkeypatch.setattr(web_mod, "run_folder_picker", fake_pick)
    picker = web_mod.FolderPicker()
    picker.start()
    assert picker.cancel() is True
    assert picker.result()["status"] == "error"
    # 已经不在 pending，再取消返回 False
    assert picker.cancel() is False
    release.set()


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
