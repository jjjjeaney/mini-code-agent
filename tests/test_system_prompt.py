"""system prompt 怎么拼：静态那半 + 会话实情 + 档位 + AGENTS.md + 记忆。全部本地执行。"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import date

import pytest

from mini_code_agent.memory import Memory, write
from mini_code_agent.permissions import PERMISSION_MODES
from mini_code_agent.system_prompt import _branch_of, build_system_prompt
from mini_code_agent.tools.safety import MAX_TEXT_CHARS


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


def git_line(text: str) -> str:
    """把"这个仓库现在…"那一行挑出来，理由同 session_line。"""
    return next(line for line in text.splitlines() if line.startswith("这个仓库现在"))


def run_git(cwd, *args, check=True):
    # 提交要带身份，不然 git 会拒绝——临时仓库里没有全局配置。
    # `encoding="utf-8"` 也不能省：不给的话 Python 按本地编码（这台机器是 GBK）去解，而 git
    # 吐的是 UTF-8——中文提交信息会解炸（生产代码那边 _run_git 同样是写死 utf-8 的）。
    return subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture
def repo(tmp_path):
    """一个真有提交的仓库：init、提交一笔。"""
    run_git(tmp_path, "init")
    (tmp_path / "a.txt").write_text("原来的\n", encoding="utf-8")
    run_git(tmp_path, "add", "a.txt")
    run_git(tmp_path, "commit", "-m", "第一笔提交")
    return tmp_path


class TestGitSection:
    def test_it_says_which_branch(self, repo):
        branch = run_git(repo, "branch", "--show-current").stdout.strip()

        assert f"分支 {branch}" in git_line(build_system_prompt(repo, "default"))

    def test_a_clean_tree_says_clean(self, repo):
        assert "工作区干净" in git_line(build_system_prompt(repo, "default"))

    def test_it_counts_changes_and_untracked_separately(self, repo):
        """改动和未跟踪分开数——"我改过的"和"新加的"是两回事。"""
        (repo / "a.txt").write_text("改了\n", encoding="utf-8")
        (repo / "新加的.txt").write_text("新\n", encoding="utf-8")

        assert "1 个改动、1 个未跟踪" in git_line(build_system_prompt(repo, "default"))

    def test_it_lists_the_recent_commits(self, repo):
        text = build_system_prompt(repo, "default")

        assert "最近几条提交" in text
        assert "第一笔提交" in text

    def test_a_chinese_file_name_is_not_escaped(self, repo):
        """不加 `-c core.quotepath=false` 的话，git 会把中文名转义成 `\\344\\275\\240` 那种。

        而且这个开关**得放在子命令前面**——放后面 git 直接报 unknown switch c。
        """
        (repo / "新加的.txt").write_text("新\n", encoding="utf-8")

        assert "新加的.txt" in build_system_prompt(repo, "default")

    def test_an_empty_repo_reports_the_branch_but_no_commits(self, tmp_path):
        """`git init` 之后还没提交：分支报得出来，"最近提交"那半没有。"""
        run_git(tmp_path, "init")

        text = build_system_prompt(tmp_path, "default")

        assert "分支 " in git_line(text)
        assert "最近几条提交" not in text

    def test_a_directory_that_is_not_a_repo_says_nothing(self, tmp_path):
        """不是仓库就**整段不写**——别在提示词里留个空壳，也别每发都印一句"git 用不了"。"""
        assert "这个仓库现在" not in build_system_prompt(tmp_path, "default")

    def test_a_hostile_config_cannot_hide_the_untracked_files(self, repo):
        """`status.showUntrackedFiles=no` 会把未跟踪的文件整个藏掉，而输出看着一切正常（实测）。

        `--porcelain` 保证的是**格式**稳定，不保证这些开关还是默认值——所以得显式钉住。
        """
        run_git(repo, "config", "status.showUntrackedFiles", "no")
        (repo / "新加的.txt").write_text("新\n", encoding="utf-8")

        assert "1 个未跟踪" in git_line(build_system_prompt(repo, "default"))


class TestBranchParsing:
    """从 `git status --branch` 的第一行里取分支名。三种写法都得认，这层不用起进程。"""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("## main", "main"),
            ("## main...origin/main", "main"),
            ("## main...origin/main [ahead 1]", "main"),
            ("## No commits yet on main", "main"),
            ("## HEAD (no branch)", ""),  # 分离头指针：没有分支可报
        ],
    )
    def test_it_picks_the_branch_out(self, header, expected):
        assert _branch_of(header) == expected


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
    size = MAX_TEXT_CHARS + 500  # 跟常量走，别写死——限额改了这条也得跟着对
    (tmp_path / "AGENTS.md").write_text("x" * size, encoding="utf-8")

    text = build_system_prompt(tmp_path, "default")

    # **别图省事断言"已截断"三个字**：静态提示词里本来就有一句「结果里写着"已截断"时…」，
    # 那样写会永远成立——这条用例就变成假通过的。要断言就断言截断说明里独有的那半句。
    assert f"全文共 {size} 个字符" in text


def test_an_empty_rules_file_is_skipped(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")

    assert "AGENTS.md" not in build_system_prompt(tmp_path, "default")


class TestRulesWalkUp:
    """从 cwd 往上收集 AGENTS.md，到**仓库根**为止，多的拼起来。"""

    def test_it_collects_every_level_from_the_root_down(self, repo):
        """两份都拼上，而且**根上的在前**——越靠后越贴近当前目录。"""
        (repo / "AGENTS.md").write_text("根上的规矩\n", encoding="utf-8")
        sub = repo / "packages" / "api"
        sub.mkdir(parents=True)
        (sub / "AGENTS.md").write_text("子目录的规矩\n", encoding="utf-8")

        text = build_system_prompt(sub, "default")

        assert "根上的规矩" in text
        assert "子目录的规矩" in text
        assert text.index("根上的规矩") < text.index("子目录的规矩")
        assert "越靠后的越贴近当前目录" in text  # 顺序得说明白，不然模型没法判
        # 引导句和第一个标题之间要有空行——不然 markdown 会把标题吸进上一段
        assert "：\n\n### " in text

    def test_each_block_says_how_far_up_it_came_from(self, repo):
        (repo / "AGENTS.md").write_text("根上的\n", encoding="utf-8")
        sub = repo / "sub"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("这层的\n", encoding="utf-8")

        text = build_system_prompt(sub, "default")

        assert f"### {os.path.join('..', 'AGENTS.md')}" in text
        assert "### AGENTS.md" in text

    def test_a_single_file_is_not_labelled(self, repo):
        """只有一份时输出跟没这个功能时一模一样——不白多两行。"""
        (repo / "AGENTS.md").write_text("就这一份\n", encoding="utf-8")

        text = build_system_prompt(repo, "default")

        assert "就这一份" in text
        assert "### " not in text
        assert "越靠后" not in text

    def test_no_repo_means_no_walking_up(self, tmp_path):
        """**这条最要紧**：找不到 `.git` 就**不许往上走**。

        不设这条的话，cwd 之外、跟这个项目无关的 AGENTS.md 也会被卷进提示词——而它进的是
        system prompt，等于谁能往上级目录写文件，谁就能给这个 agent 下指令。
        """
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "AGENTS.md").write_text("上面那一层的\n", encoding="utf-8")
        (inner / "AGENTS.md").write_text("我这一层的\n", encoding="utf-8")

        text = build_system_prompt(inner, "default")

        assert "我这一层的" in text
        assert "上面那一层的" not in text

    def test_it_stops_at_the_repo_root(self, tmp_path):
        """仓库根**再往上**就不是这个仓库了——那一层的 AGENTS.md 不该进来。

        （`.git` 是目录还是文件都算，worktree / submodule 里它是文件。这里造个目录就够。）
        """
        (tmp_path / "AGENTS.md").write_text("仓库外面的\n", encoding="utf-8")
        root = tmp_path / "repo"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("根上的\n", encoding="utf-8")
        (sub / "AGENTS.md").write_text("这层的\n", encoding="utf-8")

        text = build_system_prompt(sub, "default")

        assert "根上的" in text
        assert "这层的" in text
        assert "仓库外面的" not in text

    def test_an_unreadable_one_says_which_file_it_was(self, repo):
        """多份的时候不说是哪一份，等于没说。"""
        (repo / "AGENTS.md").write_bytes(b"\xff\xfe\x80\x81 not utf-8")
        sub = repo / "sub"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("这层的\n", encoding="utf-8")

        text = build_system_prompt(sub, "default")

        assert "读不出来" in text
        assert f"（{os.path.join('..', 'AGENTS.md')} 在" in text


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
