"""list_files / project_tree 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import (
    MAX_GREP_MATCHES,
    glob,
    grep,
    list_files,
    project_tree,
    read_file,
)
from mini_code_agent.tools.safety import (
    MAX_READ_BYTES,
    MAX_TEXT_CHARS,
    NotTextFileError,
    OutOfSandboxError,
    SensitiveFileError,
    UndecodableTextError,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src" / "core").mkdir(parents=True)
    (tmp_path / "src" / "cli.py").write_text("x", encoding="utf-8")
    (tmp_path / "src" / "core" / "agent.py").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(project: Path) -> ToolContext:
    return ToolContext(cwd=project)


class TestListFiles:
    """只看一层，不往下钻。"""

    def test_lists_only_the_top_level(self, project, ctx):
        assert list_files({}, ctx) == "\n".join(
            [
                "├── src/",
                "└── README.md",
            ]
        )

    def test_does_not_descend_into_subdirectories(self, project, ctx):
        out = list_files({}, ctx)

        assert "cli.py" not in out
        assert "core" not in out

    def test_can_list_a_subdirectory_one_level(self, project, ctx):
        assert list_files({"path": "src"}, ctx) == "\n".join(
            [
                "├── core/",
                "└── cli.py",
            ]
        )

    def test_directories_come_before_files(self, project, ctx):
        lines = list_files({}, ctx).splitlines()

        assert lines[0].endswith("src/")
        assert lines[1].endswith("README.md")

    def test_missing_path_means_the_root(self, project, ctx):
        assert list_files({}, ctx) == list_files({"path": "."}, ctx)

    def test_empty_path_string_also_means_the_root(self, project, ctx):
        assert list_files({"path": ""}, ctx) == list_files({}, ctx)

    def test_skipped_entries_are_not_listed(self, project, ctx):
        out = list_files({}, ctx)

        assert ".git" not in out
        assert "__pycache__" not in out
        assert ".env" not in out

    def test_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")

        with pytest.raises(OutOfSandboxError):
            list_files({"path": str(outside)}, ctx)

    def test_dotdot_escape_is_rejected(self, project, ctx):
        with pytest.raises(OutOfSandboxError):
            list_files({"path": "../"}, ctx)

    def test_a_file_is_not_a_directory(self, project, ctx):
        with pytest.raises(NotADirectoryError):
            list_files({"path": "README.md"}, ctx)

    def test_a_missing_directory_is_reported(self, project, ctx):
        with pytest.raises(FileNotFoundError):
            list_files({"path": "nope"}, ctx)

    def test_an_empty_directory_says_so(self, project, ctx):
        (project / "empty").mkdir()

        assert "没有可列出的内容" in list_files({"path": "empty"}, ctx)

    def test_huge_output_is_truncated(self, project, ctx):
        (project / "many").mkdir()
        for index in range(200):
            name = f"{index:04d}" + "x" * 55
            (project / "many" / f"{name}.txt").write_text("x", encoding="utf-8")

        assert "已截断" in list_files({"path": "many"}, ctx)


class TestProjectTree:
    """从 cwd 起递归，用 max_depth 和 max_nodes 控制规模。"""

    def test_renders_the_whole_project(self, project, ctx):
        assert project_tree({}, ctx) == "\n".join(
            [
                "├── src/",
                "│   ├── core/",
                "│   │   └── agent.py",
                "│   └── cli.py",
                "└── README.md",
            ]
        )

    def test_directories_come_before_files(self, project, ctx):
        lines = project_tree({}, ctx).splitlines()

        assert lines[0] == "├── src/"
        assert lines[-1] == "└── README.md"

    def test_max_depth_counts_levels_below_the_root(self, project, ctx):
        assert project_tree({"max_depth": 1}, ctx) == "\n".join(
            [
                "├── src/",
                "└── README.md",
            ]
        )

    def test_max_depth_two_stops_before_the_deepest_file(self, project, ctx):
        assert project_tree({"max_depth": 2}, ctx) == "\n".join(
            [
                "├── src/",
                "│   ├── core/",
                "│   └── cli.py",
                "└── README.md",
            ]
        )

    def test_max_nodes_keeps_the_shallow_levels_complete(self, project, ctx):
        """按层挑：撞上限时保住的是浅的层，砍掉的是最深的那部分。

        要是改成按深度递归着挑，这里会变成只显示 src/ 更深的内容，
        而顶层的 README.md 反而看不见。
        """
        out = project_tree({"max_nodes": 2}, ctx)

        assert out.splitlines()[:2] == ["├── src/", "└── README.md"]
        assert "已截断" in out
        assert "上限 2" in out

    def test_skipped_entries_are_hidden(self, project, ctx):
        out = project_tree({}, ctx)

        for name in (".git", "__pycache__", ".env"):
            assert name not in out

    def test_skipped_directories_are_not_walked_into(self, project, ctx):
        """剪枝：.git 和 __pycache__ 里面的文件一个都不该出现。"""
        out = project_tree({}, ctx)

        assert "HEAD" not in out
        assert "junk.pyc" not in out

    def test_an_empty_project_says_so(self, ctx, tmp_path):
        empty = tmp_path / "empty-project"
        empty.mkdir()

        assert "没有可列出的内容" in project_tree({}, ToolContext(cwd=empty))

    def test_symlinked_directory_is_shown_but_not_expanded(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.py").write_text("x", encoding="utf-8")
        try:
            os.symlink(outside, project / "link", target_is_directory=True)
        except OSError:
            pytest.skip("这个环境建不了符号链接（Windows 上需要开发者模式或管理员权限）")

        out = project_tree({}, ctx)

        assert "link/" in out
        assert "secret.py" not in out


class TestReadFile:
    def test_reads_a_small_file_whole(self, project, ctx):
        (project / "a.txt").write_text("第一行\n第二行\n", encoding="utf-8")

        assert read_file({"path": "a.txt"}, ctx) == "第一行\n第二行"

    def test_no_range_note_when_the_whole_file_was_read(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert "[这是第" not in read_file({"path": "a.txt"}, ctx)

    def test_a_line_range_slices_and_says_which_part(self, project, ctx):
        body = "\n".join(f"line{i}" for i in range(1, 11))
        (project / "a.txt").write_text(body, encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 3, "end_line": 5}, ctx)

        assert out.splitlines()[:3] == ["line3", "line4", "line5"]
        assert "[这是第 3-5 行，文件共 10 行]" in out

    def test_start_line_alone_reads_to_the_end(self, project, ctx):
        (project / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 4}, ctx)

        assert out.splitlines()[:2] == ["line4", "line5"]
        assert "[这是第 4-5 行，文件共 5 行]" in out

    def test_line_numbers_start_at_one(self, project, ctx):
        (project / "a.txt").write_text("first\nsecond\n", encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 1, "end_line": 1}, ctx)

        assert out.startswith("first")
        assert "second" not in out

    def test_end_line_beyond_the_file_is_clamped(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert read_file({"path": "a.txt", "end_line": 999}, ctx) == "one\ntwo"

    def test_starting_past_the_end_says_how_long_the_file_is(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert "只有 2 行" in read_file({"path": "a.txt", "start_line": 50}, ctx)

    def test_a_reversed_range_is_rejected(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

        with pytest.raises(ValueError, match="不成立"):
            read_file({"path": "a.txt", "start_line": 3, "end_line": 1}, ctx)

    def test_crlf_files_come_back_with_plain_newlines(self, project, ctx):
        (project / "a.txt").write_bytes(b"one\r\ntwo\r\n")

        assert read_file({"path": "a.txt"}, ctx) == "one\ntwo"

    def test_a_directory_is_rejected(self, project, ctx):
        with pytest.raises(IsADirectoryError):
            read_file({"path": "src"}, ctx)

    def test_a_missing_file_is_reported(self, project, ctx):
        with pytest.raises(FileNotFoundError):
            read_file({"path": "nope.txt"}, ctx)

    def test_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "secret.txt"
        outside.write_text("x", encoding="utf-8")

        with pytest.raises(OutOfSandboxError):
            read_file({"path": str(outside)}, ctx)

    def test_binary_is_rejected(self, project, ctx):
        (project / "logo.png").write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(200, 256))
        )

        with pytest.raises(NotTextFileError):
            read_file({"path": "logo.png"}, ctx)

    def test_a_too_large_file_is_rejected_before_reading_it(self, project, ctx):
        (project / "big.txt").write_bytes(b"x" * (MAX_READ_BYTES + 1))

        with pytest.raises(ValueError, match="too large"):
            read_file({"path": "big.txt"}, ctx)

    def test_gbk_is_reported_as_an_encoding_problem_not_as_binary(self, project, ctx):
        """是文本、只是编码不对——这跟"是二进制"要分开报，模型该做的事不一样。"""
        (project / "gbk.txt").write_bytes("你好，世界\n".encode("gbk"))

        with pytest.raises(UndecodableTextError):
            read_file({"path": "gbk.txt"}, ctx)

    def test_long_output_is_truncated(self, project, ctx):
        (project / "long.txt").write_text("x" * (MAX_TEXT_CHARS + 500), encoding="utf-8")

        assert "已截断" in read_file({"path": "long.txt"}, ctx)

    def test_env_is_refused(self, project, ctx):
        """fixture 里那个 .env 装着 KEY=1——读了就等于把它发出去。"""
        with pytest.raises(SensitiveFileError):
            read_file({"path": ".env"}, ctx)

    def test_env_variants_are_refused(self, project, ctx):
        (project / ".env.production").write_text("K=1", encoding="utf-8")

        with pytest.raises(SensitiveFileError):
            read_file({"path": ".env.production"}, ctx)

    def test_key_files_are_refused(self, project, ctx):
        (project / "server.pem").write_text("x", encoding="utf-8")

        with pytest.raises(SensitiveFileError):
            read_file({"path": "server.pem"}, ctx)

    def test_a_file_inside_git_is_refused(self, project, ctx):
        """.git/config 的 remote URL 里经常嵌着 token。"""
        (project / ".git" / "config").write_text(
            "[remote]\n\turl = https://user:token@example.com/x.git\n", encoding="utf-8"
        )

        with pytest.raises(SensitiveFileError):
            read_file({"path": ".git/config"}, ctx)

    def test_a_normal_file_is_still_readable(self, project, ctx):
        (project / "app.py").write_text("print(1)", encoding="utf-8")

        assert read_file({"path": "app.py"}, ctx) == "print(1)"


class TestGlob:
    def test_star_lists_everything_sorted(self, project, ctx):
        assert glob({"pattern": "*"}, ctx) == "\n".join(
            [
                "README.md",
                "src/",
                "src/cli.py",
                "src/core/",
                "src/core/agent.py",
            ]
        )

    def test_a_bare_name_matches_at_any_depth(self, project, ctx):
        out = glob({"pattern": "*.py"}, ctx)

        assert "src/cli.py" in out
        assert "src/core/agent.py" in out

    def test_a_pattern_with_a_slash_is_anchored_at_the_root(self, project, ctx):
        assert glob({"pattern": "src/*.py"}, ctx) == "src/cli.py"

    def test_it_does_not_match_a_deeper_tail_the_way_purepath_match_would(self, project, ctx):
        """就是为了这条才自己写匹配的。

        `PurePath("deep/src/cli.py").match("src/*.py")` 返回 True——它是拿模式尾部跟
        路径尾部对齐的。那不符合"相对项目根目录"的意思，所以不能用它。
        """
        (project / "deep" / "src").mkdir(parents=True)
        (project / "deep" / "src" / "cli.py").write_text("x", encoding="utf-8")

        assert glob({"pattern": "src/*.py"}, ctx) == "src/cli.py"

    def test_double_star_crosses_levels(self, project, ctx):
        assert glob({"pattern": "src/**/*.py"}, ctx) == "src/cli.py\nsrc/core/agent.py"

    def test_a_bare_name_finds_a_directory_too(self, project, ctx):
        assert glob({"pattern": "src"}, ctx) == "src/"

    def test_question_mark_matches_exactly_one_character(self, project, ctx):
        assert glob({"pattern": "cli.p?"}, ctx) == "src/cli.py"

    def test_a_leading_dot_slash_is_ignored(self, project, ctx):
        assert glob({"pattern": "./src/*.py"}, ctx) == "src/cli.py"

    def test_skipped_entries_are_not_returned(self, project, ctx):
        out = glob({"pattern": "*"}, ctx)

        assert ".env" not in out
        assert "__pycache__" not in out

    def test_skipped_directories_are_not_walked_into(self, project, ctx):
        assert "没有匹配" in glob({"pattern": "HEAD"}, ctx)

    def test_no_match_says_so(self, project, ctx):
        assert "没有匹配" in glob({"pattern": "*.rs"}, ctx)

    def test_an_absolute_pattern_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="相对项目根目录"):
            glob({"pattern": str(project / "*.py")}, ctx)

    def test_a_parent_escape_pattern_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="不能有"):
            glob({"pattern": "../*.py"}, ctx)

    def test_an_empty_pattern_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="不能为空"):
            glob({"pattern": "   "}, ctx)


class TestGrep:
    """纯 Python 那条路——这台机器上 shutil.which("rg") 就是 None，走的一直是它。"""

    def test_finds_path_line_and_text(self, project, ctx):
        body = "import os\n\ndef main():\n    return os\n"
        (project / "app.py").write_text(body, encoding="utf-8")

        assert grep({"pattern": "os"}, ctx) == "app.py:1:import os\napp.py:4:    return os"

    def test_results_are_sorted_by_path_then_line(self, project, ctx):
        (project / "src" / "b.py").write_text("needle\n", encoding="utf-8")
        (project / "a.py").write_text("needle\n", encoding="utf-8")

        assert grep({"pattern": "needle"}, ctx) == "a.py:1:needle\nsrc/b.py:1:needle"

    def test_path_limits_the_search(self, project, ctx):
        (project / "a.py").write_text("needle\n", encoding="utf-8")
        (project / "src" / "b.py").write_text("needle\n", encoding="utf-8")

        assert grep({"pattern": "needle", "path": "src"}, ctx) == "src/b.py:1:needle"

    def test_glob_limits_which_files_are_searched(self, project, ctx):
        (project / "a.py").write_text("needle\n", encoding="utf-8")
        (project / "a.txt").write_text("needle\n", encoding="utf-8")

        assert grep({"pattern": "needle", "glob": "*.py"}, ctx) == "a.py:1:needle"

    def test_case_sensitive_by_default(self, project, ctx):
        (project / "a.txt").write_text("Needle\n", encoding="utf-8")

        assert "没有找到" in grep({"pattern": "needle"}, ctx)

    def test_ignore_case_finds_it_anyway(self, project, ctx):
        (project / "a.txt").write_text("Needle\n", encoding="utf-8")

        assert grep({"pattern": "needle", "ignore_case": True}, ctx) == "a.txt:1:Needle"

    def test_the_pattern_is_a_regex(self, project, ctx):
        (project / "a.txt").write_text("def foo():\ndef bar():\n", encoding="utf-8")

        assert grep({"pattern": r"def \w+\(\):", "glob": "a.txt"}, ctx) == (
            "a.txt:1:def foo():\na.txt:2:def bar():"
        )

    def test_sensitive_files_are_not_searched(self, project, ctx):
        """fixture 里 .env 的内容是 KEY=1——它不该出现在任何结果里。"""
        assert "没有找到" in grep({"pattern": "KEY"}, ctx)

    def test_skipped_directories_are_not_walked_into(self, project, ctx):
        (project / "__pycache__" / "deep.pyc").write_text("uniqueneedle\n", encoding="utf-8")

        assert "没有找到" in grep({"pattern": "uniqueneedle"}, ctx)

    def test_matches_are_capped(self, project, ctx):
        (project / "many.txt").write_text("hit\n" * (MAX_GREP_MATCHES + 50), encoding="utf-8")

        out = grep({"pattern": "hit", "glob": "many.txt"}, ctx)

        assert "已截断" in out
        assert f"上限 {MAX_GREP_MATCHES}" in out

    def test_no_match_says_so(self, project, ctx):
        assert "没有找到" in grep({"pattern": "绝对不存在的字符串"}, ctx)

    def test_a_bad_regex_is_reported(self, project, ctx):
        with pytest.raises(ValueError, match="正则"):
            grep({"pattern": "([未闭合"}, ctx)

    def test_an_empty_pattern_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="不能为空"):
            grep({"pattern": ""}, ctx)

    def test_an_absolute_glob_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="相对项目根目录"):
            grep({"pattern": "x", "glob": str(project / "*.py")}, ctx)

    def test_a_parent_escape_glob_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="不能有"):
            grep({"pattern": "x", "glob": "../*.py"}, ctx)

    def test_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")

        with pytest.raises(OutOfSandboxError):
            grep({"pattern": "x", "path": str(outside)}, ctx)


class FakeRgProcess:
    """顶掉 subprocess.Popen 的假进程：吐出预设的几行，并记下有没有被 kill。"""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return self.returncode

    def __enter__(self) -> FakeRgProcess:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class TestGrepWithRipgrep:
    """rg 那条分支。

    这台机器上没装 ripgrep（`shutil.which("rg")` 返回 None），所以只能拿假进程测"参数
    怎么拼、输出怎么解析、结果怎么过滤"。真正的 rg 行为验不了——这点得说清楚。
    """

    def fake_rg(self, monkeypatch, lines: list[str], returncode: int = 0) -> FakeRgProcess:
        import mini_code_agent.tools.files as files

        process = FakeRgProcess(lines, returncode)
        monkeypatch.setattr(files.shutil, "which", lambda _name: "/fake/rg")
        monkeypatch.setattr(files.subprocess, "Popen", lambda *args, **kwargs: process)
        return process

    def test_rg_output_is_parsed_into_path_line_text(self, project, ctx, monkeypatch):
        self.fake_rg(
            monkeypatch,
            ["src/cli.py:3:import os\n", "src/cli.py:9:    return os\n"],
        )

        assert grep({"pattern": "os"}, ctx) == "src/cli.py:3:import os\nsrc/cli.py:9:    return os"

    @pytest.mark.skipif(os.name != "nt", reason="只有 Windows 上 rg 才吐反斜杠路径")
    def test_windows_style_paths_from_rg_are_normalized(self, project, ctx, monkeypatch):
        r"""rg 在 Windows 上吐的是 `.\src\cli.py`，要统一成 `src/cli.py`。

        这条不是推断出来的——是拿真 rg（15.2.0）跑出来才发现的：两条后端给的路径长得
        不一样。以后谁要动这段解析，先看看这条。
        """
        self.fake_rg(monkeypatch, [".\\src\\cli.py:3:import os\n"])

        assert grep({"pattern": "os"}, ctx) == "src/cli.py:3:import os"

    def test_rg_results_are_still_filtered_for_sensitive_paths(self, project, ctx, monkeypatch):
        """就算 rg 把 .env 的结果吐出来，最后也必须被滤掉——这是唯一保证得住的一道。"""
        self.fake_rg(monkeypatch, [".env:1:KEY=1\n", "src/cli.py:1:fine\n"])

        assert grep({"pattern": ".*"}, ctx) == "src/cli.py:1:fine"

    def test_an_rg_error_falls_back_to_python(self, project, ctx, monkeypatch):
        """退出码 2 表示 rg 自己出错，这时空结果不是"没有匹配"，该让纯 Python 去跑。"""
        self.fake_rg(monkeypatch, [], returncode=2)
        (project / "a.txt").write_text("needle\n", encoding="utf-8")

        assert grep({"pattern": "needle"}, ctx) == "a.txt:1:needle"
