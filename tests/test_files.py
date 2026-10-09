"""list_files / project_tree 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import list_files, project_tree
from mini_code_agent.tools.safety import OutOfSandboxError


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
