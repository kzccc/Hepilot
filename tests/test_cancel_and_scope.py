"""取消，以及 durable memory 的工作区/用户两级拆分。"""

import json
import threading

from hepilot import FakeModelClient
from hepilot.memory import scope_for_topic
from hepilot.runtime import Hepilot
from hepilot.session_store import SessionStore
from hepilot.task_state import STATUS_STOPPED, STOP_REASON_CANCELLED
from hepilot.workspace import WorkspaceContext


def build(tmp_path, outputs, user_root=None, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".hepilot" / "sessions")
    return Hepilot(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        user_memory_root=user_root,
        **kwargs,
    )


def read_report(agent):
    return json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 取消


def test_cancel_before_first_model_call_skips_the_model(tmp_path):
    agent = build(tmp_path, ["<final>should not run</final>"])
    agent.cancel_token = threading.Event()
    agent.cancel_token.set()

    answer = agent.ask("do something")

    assert answer == "Cancelled by user."
    # 模型一次都没被调用（FakeModelClient 的 outputs 还在）
    assert agent.model_client.outputs == ["<final>should not run</final>"]

    report = read_report(agent)
    assert report["status"] == STATUS_STOPPED
    assert report["stop_reason"] == STOP_REASON_CANCELLED


def test_cancel_during_model_call_is_not_reported_as_model_error(tmp_path):
    class CancellingClient:
        """模拟"流到一半用户点了停止"：模型层因取消而抛错。"""

        supports_prompt_cache = False
        last_completion_metadata = {}

        def complete(self, prompt, max_new_tokens, cancel=None, **kwargs):
            assert cancel is not None, "runtime should pass the cancel token"
            cancel.set()
            raise RuntimeError("cancelled by user")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    agent = Hepilot(
        model_client=CancellingClient(),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
    )
    agent.cancel_token = threading.Event()

    answer = agent.ask("do something")

    assert answer == "Cancelled by user."
    report = read_report(agent)
    assert report["status"] == STATUS_STOPPED
    assert report["stop_reason"] == STOP_REASON_CANCELLED


def test_cancel_event_is_recorded_in_session_log(tmp_path):
    agent = build(tmp_path, ["<final>x</final>"])
    agent.cancel_token = threading.Event()
    agent.cancel_token.set()

    agent.ask("do something")

    events = agent.session_store.read_events(agent.session["id"])
    assert any(event["type"] == "run_cancelled" for event in events)


def test_cancel_token_is_optional(tmp_path):
    """没有取消令牌时行为完全不变。"""
    agent = build(tmp_path, ["<final>done</final>"])

    assert agent.ask("hi") == "done"
    assert agent.cancel_token is None


# --------------------------------------------------------- durable 作用域


def test_topic_scopes():
    assert scope_for_topic("project-conventions") == "workspace"
    assert scope_for_topic("key-decisions") == "workspace"
    assert scope_for_topic("dependency-facts") == "workspace"
    assert scope_for_topic("user-preferences") == "user"
    # 未知 topic 按工作区级处理，避免写到用户级
    assert scope_for_topic("something-else") == "workspace"


def test_user_preferences_land_in_user_root(tmp_path):
    user_root = tmp_path / "user-memory"
    agent = build(
        tmp_path,
        [
            "<final>Preference: reply in Chinese, keep answers short.</final>",
        ],
        user_root=user_root,
    )

    agent.ask("Please remember this preference: reply in Chinese.")

    preference_file = user_root / "topics" / "user-preferences.md"
    assert preference_file.exists()
    assert "reply in Chinese" in preference_file.read_text(encoding="utf-8")
    # 不能污染工作区级 memory
    assert not (tmp_path / ".hepilot" / "memory" / "topics" / "user-preferences.md").exists()


def test_workspace_conventions_stay_in_repo_memory(tmp_path):
    user_root = tmp_path / "user-memory"
    agent = build(
        tmp_path,
        ["<final>Project convention: always run pytest before commit.</final>"],
        user_root=user_root,
    )

    agent.ask("Please remember this project convention.")

    convention_file = tmp_path / ".hepilot" / "memory" / "topics" / "project-conventions.md"
    assert convention_file.exists()
    assert "always run pytest" in convention_file.read_text(encoding="utf-8")
    # 不能写进用户级 memory
    assert not (user_root / "topics" / "project-conventions.md").exists()


def test_without_user_root_only_workspace_memory_is_written(tmp_path):
    agent = build(tmp_path, ["<final>Preference: use tabs.</final>"])

    agent.ask("Please remember this preference.")

    assert not (tmp_path / ".hepilot" / "memory" / "topics" / "user-preferences.md").exists()
