"""list_files 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import list_files
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


def test_renders_the_whole_tree(project, ctx):
    """最直接的一条：一棵小树该长什么样，整段比对。

    顺带钉住缩进规则——`src/` 是最后一个孩子，所以它的子项不垫竖线；
    要是垫了，这里会显示成 `│   ├── cli.py`。
    """
    assert list_files({}, ctx) == "\n".join(
        [
            "├── README.md",
            "└── src/",
            "    ├── cli.py",
            "    └── core/",
            "        └── agent.py",
        ]
    )


def test_directories_get_a_slash_and_files_do_not(project, ctx):
    lines = list_files({}, ctx).splitlines()

    assert lines[0] == "├── README.md"
    assert lines[1] == "└── src/"


def test_listing_a_subdirectory_renders_that_subtree(project, ctx):
    assert list_files({"path": "src"}, ctx) == "\n".join(
        [
            "├── cli.py",
            "└── core/",
            "    └── agent.py",
        ]
    )


def test_missing_path_means_the_root(project, ctx):
    assert list_files({}, ctx) == list_files({"path": "."}, ctx)


def test_empty_path_string_also_means_the_root(project, ctx):
    assert list_files({"path": ""}, ctx) == list_files({}, ctx)


def test_skipped_entries_are_not_listed(project, ctx):
    out = list_files({}, ctx)

    assert ".git" not in out
    assert "__pycache__" not in out
    assert ".env" not in out


def test_skipped_directories_are_not_walked_into(project, ctx):
    """剪枝：.git 和 __pycache__ 里面的文件一个都不该出现。"""
    out = list_files({}, ctx)

    assert "HEAD" not in out
    assert "junk.pyc" not in out


def test_path_outside_the_root_is_rejected(project, ctx, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")

    with pytest.raises(OutOfSandboxError):
        list_files({"path": str(outside)}, ctx)


def test_dotdot_escape_is_rejected(project, ctx):
    with pytest.raises(OutOfSandboxError):
        list_files({"path": "../"}, ctx)


def test_a_file_is_not_a_directory(project, ctx):
    with pytest.raises(NotADirectoryError):
        list_files({"path": "README.md"}, ctx)


def test_a_missing_directory_is_reported(project, ctx):
    with pytest.raises(FileNotFoundError):
        list_files({"path": "nope"}, ctx)


def test_an_empty_directory_says_so(project, ctx):
    (project / "empty").mkdir()

    assert "没有可列出的内容" in list_files({"path": "empty"}, ctx)


def test_huge_output_is_truncated(project, ctx):
    (project / "many").mkdir()
    for index in range(200):
        name = f"{index:04d}" + "x" * 55
        (project / "many" / f"{name}.txt").write_text("x", encoding="utf-8")

    out = list_files({"path": "many"}, ctx)

    assert "已截断" in out
