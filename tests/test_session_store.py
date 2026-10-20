"""集中式会话存储的行为测试。"""

import json

from hepilot.session_store import SessionStore, workspace_slug


def make_session(store, workspace_root, session_id="s1"):
    session = {
        "id": session_id,
        "title": "demo",
        "created_at": "2026-01-01T00:00:00+00:00",
        "workspace_root": str(workspace_root),
        "history": [],
        "memory": {},
    }
    store.create({"id": session_id, "title": "demo", "workspace_root": str(workspace_root)})
    return session


def test_layout_groups_sessions_by_workspace(tmp_path):
    root = tmp_path / "sessions"
    store = SessionStore(root)
    session = make_session(store, tmp_path / "repo-a")

    directory = store.session_dir("s1")
    assert directory.name == "s1"
    assert directory.parent.name == workspace_slug(tmp_path / "repo-a")
    assert (directory / "session.json").exists()


def test_list_returns_metadata_without_reading_events(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    make_session(store, tmp_path / "repo")

    items = store.list()
    assert len(items) == 1
    assert items[0]["id"] == "s1"
    assert items[0]["event_count"] == 0


def test_events_are_appended_and_replayed_into_history(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")

    store.append_event(session, "user_message", {"text": "hello"}, run_id="r1")
    store.append_event(session, "assistant_message", {"text": "done"}, run_id="r1")
    store.write_snapshot(session)

    reloaded = store.load("s1")
    assert [item["content"] for item in reloaded["history"]] == ["hello", "done"]


def test_tool_events_replay_into_tool_history(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")

    store.append_event(
        session,
        "tool_executed",
        {"tool_use_id": "t1", "name": "read_file", "args": {"path": "a.py"}, "result": "body"},
    )

    reloaded = store.load("s1")
    tool = reloaded["history"][0]
    assert tool["role"] == "tool"
    assert tool["name"] == "read_file"
    assert tool["parent_tool_use_id"] == "t1"


def test_session_reset_is_a_logical_truncation_point(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")

    store.append_event(session, "user_message", {"text": "old"})
    store.append_event(session, "session_reset", {})
    store.append_event(session, "user_message", {"text": "new"})

    reloaded = store.load("s1")
    assert [item["content"] for item in reloaded["history"]] == ["new"]


def test_snapshot_round_trips_memory_and_checkpoints(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")
    session["memory"] = {"working": {"task_summary": "task", "recent_files": ["a.py"]}}
    session["checkpoints"] = {"current_id": "ckpt_1", "items": {"ckpt_1": {"checkpoint_id": "ckpt_1"}}}

    store.write_snapshot(session)
    saved = json.loads(store.snapshot_path("s1").read_text(encoding="utf-8"))

    assert saved["memory"]["working"]["task_summary"] == "task"
    assert saved["checkpoints"]["current_id"] == "ckpt_1"
    reloaded = store.load("s1")
    assert reloaded["memory"]["working"]["task_summary"] == "task"


def test_seq_is_monotonic_across_appends(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")

    seqs = [store.append_event(session, "user_message", {"text": str(i)})["seq"] for i in range(3)]

    assert seqs == [1, 2, 3]
    assert store.read_header("s1")["last_seq"] == 3


def test_read_events_honours_after_seq(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = make_session(store, tmp_path / "repo")
    for i in range(3):
        store.append_event(session, "user_message", {"text": str(i)})

    events = store.read_events("s1", after_seq=1)
    assert [event["seq"] for event in events] == [2, 3]
