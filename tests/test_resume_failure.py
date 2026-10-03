"""恢复失败必须收口成受控运行，而不是让异常抛穿 CLI。

这里覆盖三种“恢复不了”的情形：
1. 指定的 session 不存在
2. 指定的 session 文件损坏
3. `--resume latest` 但一个 session 都没有

三种都必须：返回非零码、不打印 traceback、并把失败写成可复盘的运行工件。
最后一条是对照：能恢复成功时不应留下任何失败工件。
"""

import json
from unittest.mock import patch

from hepilot import SessionStore
from hepilot import cli as mini_cli
from hepilot.runtime import REDACTED_VALUE
from hepilot.task_state import STATUS_COMPLETED, STATUS_FAILED, STOP_REASON_RESUME_LOAD_ERROR


class DummyModelClient:
    """启动失败路径下模型绝不该被调用，所以这里直接断言。"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        raise AssertionError("startup failure must not invoke the model")


class FinalAnswerModelClient(DummyModelClient):
    """只回一句 <final>，用于验证“启动成功”的对照路径。"""

    def complete(self, prompt, max_new_tokens, **kwargs):
        return "<final>ok</final>"


def _prepare_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")


def _run_main(argv, client=DummyModelClient):
    """用假模型 client 跑一次 CLI，避免真的去连后端。"""
    with patch("hepilot.cli.OllamaModelClient", client):
        return mini_cli.main(argv)


def _load_single_run(repo_root):
    """取出本次唯一的 run 目录及其 task_state。"""
    run_dirs = sorted((repo_root / ".hepilot" / "runs").glob("*/"))
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    run_dir = run_dirs[0]
    task_state = json.loads((run_dir / "task_state.json").read_text(encoding="utf-8"))
    return run_dir, task_state


def _read_events(run_dir):
    raw = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_missing_session_becomes_controlled_failure(tmp_path, capsys):
    _prepare_workspace(tmp_path)

    exit_code = _run_main(["--cwd", str(tmp_path), "--provider", "ollama", "--resume", "no-such-session"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Could not resume session 'no-such-session'" in captured.err
    # 受控失败的关键：不给用户抛 traceback，也不打印正常的欢迎面板。
    assert "Traceback" not in captured.err
    assert captured.out.strip() == ""

    run_dir, task_state = _load_single_run(tmp_path)
    assert task_state["status"] == STATUS_FAILED
    assert task_state["stop_reason"] == STOP_REASON_RESUME_LOAD_ERROR
    assert task_state["user_request"] == "resume:no-such-session"
    # 启动阶段就失败了：没调模型、没执行工具、也没走到 checkpoint 评估。
    assert task_state["attempts"] == 0
    assert task_state["tool_steps"] == 0
    assert task_state["resume_status"] == ""

    events = _read_events(run_dir)
    assert [event["event"] for event in events] == ["resume_failed", "run_finished"]
    assert events[0]["error_type"] == "FileNotFoundError"
    assert events[0]["requested_session_id"] == "no-such-session"
    assert events[1]["status"] == STATUS_FAILED
    assert events[1]["stop_reason"] == STOP_REASON_RESUME_LOAD_ERROR

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == STATUS_FAILED
    assert report["failed_stage"] == "resume"
    assert report["requested_session_id"] == "no-such-session"
    assert report["error_type"] == "FileNotFoundError"
    # 报告保留与正常运行一致的字段形状，避免下游统计要写特殊分支。
    assert report["durable_promotions"] == []
    assert report["task_state"]["task_id"] == task_state["task_id"]


def test_corrupt_session_file_becomes_controlled_failure(tmp_path, capsys):
    _prepare_workspace(tmp_path)
    sessions_dir = tmp_path / ".hepilot" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "broken.json").write_text("{not json", encoding="utf-8")

    exit_code = _run_main(["--cwd", str(tmp_path), "--provider", "ollama", "--resume", "broken"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Could not resume session 'broken'" in captured.err
    assert "Traceback" not in captured.err

    run_dir, task_state = _load_single_run(tmp_path)
    assert task_state["status"] == STATUS_FAILED
    assert task_state["stop_reason"] == STOP_REASON_RESUME_LOAD_ERROR

    events = _read_events(run_dir)
    assert events[0]["error_type"] == "JSONDecodeError"


def test_resume_latest_without_sessions_fails_instead_of_silently_starting_new(tmp_path, capsys):
    _prepare_workspace(tmp_path)

    exit_code = _run_main(["--cwd", str(tmp_path), "--provider", "ollama", "--resume", "latest"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Could not resume session 'latest'" in captured.err
    assert "no saved sessions" in captured.err
    # 关键回归点：不能再静默新建一个 session，让用户以为自己在续跑。
    assert list((tmp_path / ".hepilot" / "sessions").glob("*.json")) == []

    _, task_state = _load_single_run(tmp_path)
    assert task_state["stop_reason"] == STOP_REASON_RESUME_LOAD_ERROR


def test_resume_failure_artifacts_redact_secret_values(tmp_path, monkeypatch):
    secret = "sk-live-secret-42"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _prepare_workspace(tmp_path)

    # 把 secret 塞进 session id：它会同时出现在异常消息和 requested_session_id 里。
    exit_code = _run_main(["--cwd", str(tmp_path), "--provider", "ollama", "--resume", secret])

    assert exit_code == 1
    run_dir, _ = _load_single_run(tmp_path)
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")

    assert secret not in trace_text
    assert secret not in report_text
    assert REDACTED_VALUE in trace_text
    assert REDACTED_VALUE in report_text


def test_successful_resume_does_not_write_failure_artifacts(tmp_path):
    """对照：恢复成功时走的是正常运行，不该被标成恢复失败。"""
    _prepare_workspace(tmp_path)
    store = SessionStore(tmp_path / ".hepilot" / "sessions")
    session_id = "20260101-000000-abc123"
    # 故意只写最小字段：_ensure_session_shape 会补齐 memory/checkpoints 等。
    store.save(
        {
            "id": session_id,
            "created_at": "2026-01-01T00:00:00+00:00",
            "workspace_root": str(tmp_path),
            "history": [],
        }
    )

    exit_code = _run_main(
        ["--cwd", str(tmp_path), "--provider", "ollama", "--resume", session_id, "say hi"],
        client=FinalAnswerModelClient,
    )

    assert exit_code == 0
    _, task_state = _load_single_run(tmp_path)
    assert task_state["status"] == STATUS_COMPLETED
    assert task_state["stop_reason"] != STOP_REASON_RESUME_LOAD_ERROR
