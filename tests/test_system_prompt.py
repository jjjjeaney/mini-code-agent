"""system prompt 怎么拼：静态那半 + 会话实情 + 档位 + AGENTS.md + 记忆。全部本地执行。"""

from __future__ import annotations

import platform
from datetime import date

import pytest

from mini_code_agent.memory import Memory, write
from mini_code_agent.permissions import PERMISSION_MODES
from mini_code_agent.system_prompt import build_system_prompt


def session_line(text: str) -> str:
    """把"这次运行：…"那一行挑出来。

    断言别直接对着整份提示词写——静态那半里什么词都可能有（`--shell`、模型、日期……），
    撞上了就成了假通过。
    """
    return next(line for line in text.splitlines() if line.startswith("这次运行："))


def test_the_session_line_says_which_day_it_is(tmp_path):
    assert f"今天 {date.today().isoformat()}" in session_line(
        build_system_prompt(tmp_path, "default")
    )


def test_the_date_carries_no_time(tmp_path):
    """日期取到**天**。取到秒的话，这段每轮重拼都会换一个前缀——缓存从这一行往后全废。

    这条钉的就是那个格式："今天 2026-10-13" 后面不能再跟东西。
    """
    assert f"今天 {date.today().isoformat()}，" in session_line(
        build_system_prompt(tmp_path, "default")
    )


def test_the_session_line_says_which_platform(tmp_path):
    """`bash` 在 Linux 容器里、文件工具在宿主机上——不说平台，模型给的路径会按错的那边来。"""
    assert platform.system() in session_line(build_system_prompt(tmp_path, "default"))


def test_the_model_is_named_when_we_know_it(tmp_path):
    line = session_line(build_system_prompt(tmp_path, "default", "some-model-1"))

    assert "模型 some-model-1" in line


def test_no_model_is_named_when_we_do_not_know_it(tmp_path):
    """`Provider` 协议故意只声明了 complete，取不到就不写——不猜一个填上。"""
    line = session_line(build_system_prompt(tmp_path, "default"))

    assert "模型" not in line


def test_the_shell_is_named_only_when_the_environment_says_so(tmp_path, monkeypatch):
    """Windows 上常常没有 `SHELL`——那就不写。与其猜一个，不如不说。"""
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    assert "终端 shell bash" in session_line(build_system_prompt(tmp_path, "default"))

    monkeypatch.delenv("SHELL", raising=False)
    assert "终端 shell" not in session_line(build_system_prompt(tmp_path, "default"))


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows 的 shell 才带 .exe")
def test_a_windows_shell_loses_its_extension(tmp_path, monkeypatch):
    """Windows 上那几个 shell 带 `.exe`。取 stem 之后两边都叫 `bash`——跨平台一个名字好认。"""
    monkeypatch.setenv("SHELL", "C:\\Program Files\\Git\\bin\\bash.exe")

    assert "终端 shell bash" in session_line(build_system_prompt(tmp_path, "default"))


def test_every_mode_gets_its_own_line(tmp_path):
    for mode in PERMISSION_MODES:
        assert mode.split()[0] in build_system_prompt(tmp_path, mode)


def test_the_plan_line_says_read_only(tmp_path):
    """plan 那句是"plan 只读"的实际依据——没有它，那一档的承诺是空的（见 F2）。"""
    text = build_system_prompt(tmp_path, "plan")

    assert "只读" in text
    assert "不要请求改文件" in text


def test_the_mode_can_be_written_the_way_the_user_typed_it(tmp_path):
    assert "plan" in build_system_prompt(tmp_path, "Plan")


def test_a_bad_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="不是权限模式"):
        build_system_prompt(tmp_path, "yolo")


def test_the_projects_rules_are_pulled_in(tmp_path):
    (tmp_path / "AGENTS.md").write_text("测试一律用 uv run --no-sync\n", encoding="utf-8")

    text = build_system_prompt(tmp_path, "default")

    assert "uv run --no-sync" in text
    assert "AGENTS.md" in text  # 得说明这段是从哪儿来的


def test_no_rules_file_is_fine(tmp_path):
    """没有这个文件是常态，不该因此报错，也不该提它。"""
    assert "AGENTS.md" not in build_system_prompt(tmp_path, "default")


def test_a_rules_file_that_is_not_utf8_says_so(tmp_path):
    """读不出来要说一声，别让它悄悄失效——用户以为规矩生效了，其实模型没看到。"""
    (tmp_path / "AGENTS.md").write_bytes(b"\xff\xfe\x80\x81 not utf-8")

    text = build_system_prompt(tmp_path, "default")

    assert "读不出来" in text


def test_a_huge_rules_file_is_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 20000, encoding="utf-8")

    text = build_system_prompt(tmp_path, "default")

    # **别图省事断言"已截断"三个字**：静态提示词里本来就有一句「结果里写着"已截断"时…」，
    # 那样写会永远成立——这条用例就变成假通过的。要断言就断言截断说明里独有的那半句。
    assert "全文共 20000 个字符" in text


def test_an_empty_rules_file_is_skipped(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")

    assert "AGENTS.md" not in build_system_prompt(tmp_path, "default")


def test_the_memory_section_tells_it_to_look_before_working(tmp_path):
    """光把清单摆出来模型不会主动取（实测）——所以那句得是"动手之前先扫一遍"这种口气。"""
    write(
        tmp_path,
        Memory(name="tabs-for-indent", description="缩进用 tab", type="user", body="按这个来"),
    )

    text = build_system_prompt(tmp_path, "default")

    assert "tabs-for-indent" in text  # 清单进来了
    assert "按这个来" not in text  # 正文没进来——那要按需取
    assert "先扫一遍" in text  # 而且得催它去取
    assert "read_memory" in text
    assert "以眼前的为准" in text  # 记忆是快照：跟现状对不上时以现状为准


def test_no_memories_means_no_memory_section(tmp_path):
    assert "长期记忆" not in build_system_prompt(tmp_path, "default")
