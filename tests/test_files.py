"""list_files / project_tree 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import list_files, project_tree, read_file
from mini_code_agent.tools.safety import (
    MAX_READ_BYTES,
    MAX_TEXT_CHARS,
    NotTextFileError,
    OutOfSandboxError,
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
