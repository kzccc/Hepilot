"""workspace 的文档白名单：优先级、单份上限、总量上限。"""

from pathlib import Path

from hepilot import workspace as workspace_mod
from hepilot.workspace import DOC_NAMES, DOC_PER_FILE_LIMIT, DOC_TOTAL_LIMIT, WorkspaceContext


def test_whitelist_is_priority_ordered_and_includes_agent_docs():
    # AGENTS.md / CLAUDE.md 是直接约束 agent 行为的指令，排最前
    assert DOC_NAMES[0] == "AGENTS.md"
    assert DOC_NAMES[1] == "CLAUDE.md"
    assert "README.md" in DOC_NAMES
    # prefix 超预算时从尾部截断，所以优先级顺序必须稳定
    assert len(DOC_NAMES) >= 8


def test_docs_are_collected_in_priority_order(tmp_path):
    for name in ("README.md", "CLAUDE.md", "AGENTS.md"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")

    snapshot = WorkspaceContext.build(tmp_path)

    assert list(snapshot.project_docs) == ["AGENTS.md", "CLAUDE.md", "README.md"]


def test_single_doc_is_capped(tmp_path):
    (tmp_path / "README.md").write_text("x" * (DOC_PER_FILE_LIMIT * 3), encoding="utf-8")

    snapshot = WorkspaceContext.build(tmp_path)

    content = snapshot.project_docs["README.md"]
    assert content.endswith("...[truncated]")
    assert len(content) <= DOC_PER_FILE_LIMIT + len("\n...[truncated]")


def test_total_docs_budget_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_mod, "DOC_TOTAL_LIMIT", 500)
    monkeypatch.setattr(workspace_mod, "DOC_PER_FILE_LIMIT", 400)
    for name in ("AGENTS.md", "CLAUDE.md", "README.md", "Makefile"):
        (tmp_path / name).write_text("y" * 400, encoding="utf-8")

    snapshot = WorkspaceContext.build(tmp_path)

    total = sum(len(text) for text in snapshot.project_docs.values())
    assert total <= 500 + len("\n...[truncated]")
    # 预算耗尽后剩下的更靠后的文档不再收集
    assert len(snapshot.project_docs) < 4


def test_directory_with_doc_name_is_ignored(tmp_path):
    (tmp_path / "README.md").mkdir()

    snapshot = WorkspaceContext.build(tmp_path)

    assert "README.md" not in snapshot.project_docs


def test_missing_docs_render_as_none(tmp_path):
    snapshot = WorkspaceContext.build(tmp_path)

    assert snapshot.project_docs == {}
    assert "- project_docs:\n- none" in snapshot.text()
