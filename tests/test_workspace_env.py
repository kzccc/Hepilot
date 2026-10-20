"""workspace 里的机器事实：platform / shell。

为什么必须告诉模型：
run_shell 用 shell=True，Windows 上走 %COMSPEC%（cmd.exe），不是 bash。
模型不知道这件事时，会写 export FOO=1、mkdir -p a/b 这类 POSIX 语法，
白跑一轮。之前 prefix 完全没有平台信息，靠模型自己猜。
"""

import os
import sys
from pathlib import Path

from hepilot.workspace import WorkspaceContext, shell_name


def test_platform_and_shell_are_rendered(tmp_path):
    text = WorkspaceContext.build(tmp_path).text()

    assert f"- platform: {sys.platform}" in text
    assert f"- shell: {shell_name()}" in text


def test_shell_name_reports_the_real_interpreter():
    """必须报"真正执行的那个"，不是用户的 $SHELL。"""
    name = shell_name()

    if os.name == "nt":
        # cmd.exe 的完整路径，或兜底名
        assert name.lower().endswith("cmd.exe") or name == "cmd.exe"
        # 绝不能是 bash/zsh
        assert "bash" not in name.lower()
        assert "zsh" not in name.lower()
    else:
        assert name == "/bin/sh"


def test_platform_line_sits_with_machine_facts(tmp_path):
    lines = WorkspaceContext.build(tmp_path).text().splitlines()

    # 顺序：环境事实在前，git 状态在后
    assert lines.index("- repo_root: " + str(Path(tmp_path).resolve())) < lines.index(
        f"- platform: {sys.platform}"
    )
    assert lines.index(f"- platform: {sys.platform}") < lines.index("- status:")
