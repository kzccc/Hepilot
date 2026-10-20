"""连续失败的提醒闸门 + run_shell 超时默认值。

背景（"测试1"会话，提问"可以,你试一下,然后 跑通"）：
模型为了跑通 go build，共发起 83 次 run_shell，全在换 GOPATH / GOPROXY /
HTTP_PROXY / ping / dir 的组合。原因有两层：

1. run_shell 默认超时只有 20 秒。冷启动 `go mod tidy` 要下 20+ 个模块，
   20 秒必然超时，于是被判定为"失败"。
2. 失败之后上下文里只有一条条并列的失败记录，没有任何东西告诉它
   "这已经是同一类失败的第六次了"，所以它一直换参数重试。

事后用正确的环境跑同一个仓库：`go mod tidy` 退出 0、`go build ./...` 退出 0、
`go vet ./...` 无输出——代码本身是好的，纯粹是被超时和重试拖死。
"""

from hepilot import FakeModelClient
from hepilot.runtime import FAILURE_STREAK_NOTICE, Hepilot
from hepilot.session_store import SessionStore
from hepilot.tools import BASE_TOOL_SPECS
from hepilot.workspace import WorkspaceContext


class FailingClient(FakeModelClient):
    """每次返回"稍微改一点"的失败命令，模拟真实的重试风暴。

    命令必须逐轮不同：完全相同的连续调用会先被 repeated_tool_call 拦掉，
    走不到连续失败计数这条路径。
    """

    def __init__(self, rounds, prefix="dir Z:\\nope", final="<final>blocked: no network</final>"):
        replies = [
            f'<tool>{{"name":"run_shell","args":{{"command":"{prefix}-{index}","timeout":60}}}}</tool>'
            for index in range(rounds)
        ]
        replies.append(final)
        super().__init__(replies)


def make_agent(tmp_path, client, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Hepilot(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".hepilot" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


# --------------------------------------------------------- 超时默认值


def test_run_shell_default_timeout_allows_dependency_installs():
    assert BASE_TOOL_SPECS["run_shell"]["schema"]["timeout"] == "int=60"


def test_run_shell_accepts_long_timeouts():
    # 600 秒的编译在真实项目里是正常需求；120 秒会把 go mod tidy 判成超时
    import inspect
    from hepilot import tools

    source = inspect.getsource(tools)
    assert "timeout > 600" in source
    assert "timeout > 120" not in source


# --------------------------------------------------------- 连续失败提醒


def test_consecutive_failures_trigger_a_stop_notice(tmp_path):
    # 每条命令都失败，且逐轮略有变化（绕过 repeated_tool_call）
    client = FailingClient(rounds=FAILURE_STREAK_NOTICE + 2)
    agent = make_agent(tmp_path, client)

    agent.ask("跑通构建")

    results = [
        str(item["content"])
        for item in agent.session["history"]
        if item.get("role") == "tool" and item.get("name") == "run_shell"
    ]
    notices = [text for text in results if "RUNTIME NOTICE" in text]
    assert notices, "连续失败达到阈值后必须出现停止提醒"
    assert "Do not retry another variation" in notices[0]
    # 第 FAILURE_STREAK_NOTICE 次才出现，之前不能打扰
    assert "RUNTIME NOTICE" not in results[FAILURE_STREAK_NOTICE - 2]


def test_success_resets_the_failure_streak(tmp_path):
    (tmp_path / "ok.txt").write_text("hello\n", encoding="utf-8")
    replies = []
    # 失败 3 次 → 成功 1 次 → 再失败 3 次：都不该触发（计数被重置了）
    replies += [
        f'<tool>{{"name":"run_shell","args":{{"command":"dir Z:\\\\nope-{index}","timeout":60}}}}</tool>'
        for index in range(3)
    ]
    replies += ['<tool>{"name":"read_file","args":{"path":"ok.txt","start":1,"end":5}}</tool>']
    replies += [
        f'<tool>{{"name":"run_shell","args":{{"command":"dir Z:\\\\late-{index}","timeout":60}}}}</tool>'
        for index in range(3)
    ]
    replies += ["<final>done</final>"]
    agent = make_agent(tmp_path, FakeModelClient(replies), max_steps=20)

    agent.ask("mix failures and a success")

    contents = [str(item["content"]) for item in agent.session["history"] if item.get("role") == "tool"]
    assert not any("RUNTIME NOTICE" in text for text in contents)
