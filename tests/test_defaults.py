"""默认预算只有一个来源。

之前三处各写一个字面量：Hepilot 里 512、web 的 AgentConfig 里 8192、
web 的命令行参数里 2048。命令行最后传进 AgentConfig，把 8192 覆盖掉——
"改了却不生效"，而且读代码看不出来。

现在链路是单向的：

    runtime.DEFAULT_MAX_*  →  Hepilot 构造参数
                           →  AgentConfig 字段
                           →  web 命令行默认值（经 agent_config_default）
                           →  cli 命令行默认值（直接引用常量）

任何一环都不再写字面量。
"""

import inspect

from hepilot import cli, web
from hepilot.runtime import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_STEPS,
    Hepilot,
)


def test_runtime_constants_are_the_source_of_truth():
    assert DEFAULT_MAX_STEPS == 100
    assert DEFAULT_MAX_ATTEMPTS == 100
    # 8192 是当初修截断问题时定的值：2048 装不下一次多文件写入
    assert DEFAULT_MAX_NEW_TOKENS == 8192


def test_hepilot_constructor_uses_the_constants():
    parameters = inspect.signature(Hepilot.__init__).parameters

    assert parameters["max_steps"].default == DEFAULT_MAX_STEPS
    assert parameters["max_new_tokens"].default == DEFAULT_MAX_NEW_TOKENS
    # max_attempts 保持 None：表示"按 max_steps 推导"，直接调 API 的老行为不变
    assert parameters["max_attempts"].default is None


def test_agent_config_uses_the_constants():
    config = web.AgentConfig()

    assert config.max_steps == DEFAULT_MAX_STEPS
    assert config.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS


def test_web_argument_defaults_track_agent_config():
    for field in ("max_steps", "max_attempts", "max_new_tokens"):
        assert web.agent_config_default(field) == getattr(web.AgentConfig(), field)


def test_cli_argument_defaults_match_the_constants():
    args = cli.build_arg_parser().parse_args(["hi"])

    assert args.max_steps == DEFAULT_MAX_STEPS
    assert args.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS


def test_cli_and_web_agree_on_effective_budget(tmp_path):
    """两个入口最终喂给 Hepilot 的预算必须一致。

    Web 多绕一层 AgentConfig，CLI 直接引用常量——这条断言就是防止
    将来有人只改一边。
    """
    cli_args = cli.build_arg_parser().parse_args(["hi"])
    config = web.AgentConfig()

    assert cli_args.max_steps == config.max_steps
    assert cli_args.max_attempts == config.max_attempts
    assert cli_args.max_new_tokens == config.max_new_tokens
