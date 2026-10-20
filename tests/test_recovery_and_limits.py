"""进程重启后的状态回收、工作区路径校验、快照上限。"""

from pathlib import Path

import pytest

from hepilot import FakeModelClient
from hepilot import runtime as runtime_mod
from hepilot.runtime import Hepilot
from hepilot.session_store import SessionStore
from hepilot.web import AgentConfig, SessionManager
from hepilot.workspace import WorkspaceContext


# ------------------------------------------------------- 孤儿 running 回收


def test_orphaned_running_session_is_reconciled(tmp_path):
    root = tmp_path / "sessions"
    store = SessionStore(root)
    store.create({"id": "s1", "workspace_root": str(tmp_path / "repo")})
    # 模拟进程被 kill：状态停在 running，但没有 run_finished。
    store.update_header("s1", status="running")

    SessionManager(root, AgentConfig())

    assert store.read_header("s1")["status"] == "interrupted"
    events = store.read_events("s1")
    assert [e["type"] for e in events if e["type"] == "run_interrupted"] == ["run_interrupted"]
    assert events[-1]["data"]["reason"] == "process_restarted"


def test_reconcile_leaves_finished_sessions_alone(tmp_path):
    root = tmp_path / "sessions"
    store = SessionStore(root)
    store.create({"id": "s1", "workspace_root": str(tmp_path / "repo")})
    store.update_header("s1", status="idle")

    SessionManager(root, AgentConfig())

    assert store.read_header("s1")["status"] == "idle"
    assert not any(e["type"] == "run_interrupted" for e in store.read_events("s1"))


# --------------------------------------------------------- 工作区路径校验


def test_create_rejects_filesystem_root(tmp_path):
    manager = SessionManager(tmp_path / "sessions", AgentConfig())
    root = Path(tmp_path.anchor)  # Windows: C:\  POSIX: /

    with pytest.raises(ValueError, match="filesystem root"):
        manager.create(str(root))


def test_create_rejects_missing_directory(tmp_path):
    manager = SessionManager(tmp_path / "sessions", AgentConfig())

    with pytest.raises(ValueError, match="does not exist"):
        manager.create(str(tmp_path / "nope"))


# ------------------------------------------------------------- 快照上限


def make_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Hepilot(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
    )


def test_snapshot_skips_oversized_files(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    (tmp_path / "big.bin").write_bytes(b"x" * 64)
    (tmp_path / "small.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(runtime_mod, "MAX_SNAPSHOT_FILE_BYTES", 8)

    snapshot = agent.capture_workspace_snapshot()

    assert "small.txt" in snapshot
    assert "big.bin" not in snapshot


def test_snapshot_respects_file_count_cap(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(runtime_mod, "MAX_SNAPSHOT_FILES", 2)

    snapshot = agent.capture_workspace_snapshot()

    assert len(snapshot) <= 2


def test_snapshot_ignores_vcs_and_cache_dirs(tmp_path):
    agent = make_agent(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_text("x", encoding="utf-8")

    snapshot = agent.capture_workspace_snapshot()

    assert all(".git" not in path for path in snapshot)
    assert all("__pycache__" not in path for path in snapshot)
