"""list_files / project_tree 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from mini_code_agent.tools.base import (
    BackupFailedError,
    ReplaceFailedError,
    StaleReadError,
    ToolContext,
    WriteFailedError,
)
from mini_code_agent.tools.files import (
    MAX_GREP_MATCHES,
    edit_file,
    glob,
    grep,
    list_files,
    project_tree,
    read_file,
    replace_lines,
    write_file,
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

        assert read_file({"path": "a.txt"}, ctx) == "1 | 第一行\n2 | 第二行"

    def test_no_range_note_when_the_whole_file_was_read(self, project, ctx):
        """整篇读下来时，行号本身已经说清了范围——多那句"共 N 行"是噪音。"""
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert "[文件共" not in read_file({"path": "a.txt"}, ctx)

    def test_a_line_range_numbers_lines_by_their_place_in_the_file(self, project, ctx):
        """行号是**文件里的位置**，不是 1..N——只读一段时这两个数不一样，而 replace_lines
        要的是前者。"""
        body = "\n".join(f"line{i}" for i in range(1, 11))
        (project / "a.txt").write_text(body, encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 3, "end_line": 5}, ctx)

        assert out.splitlines()[:3] == ["3 | line3", "4 | line4", "5 | line5"]
        assert "[文件共 10 行]" in out

    def test_start_line_alone_reads_to_the_end(self, project, ctx):
        (project / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 4}, ctx)

        assert out.splitlines()[:2] == ["4 | line4", "5 | line5"]
        assert "[文件共 5 行]" in out

    def test_the_width_is_set_by_the_last_line_number(self, project, ctx):
        """读第 9-11 行时，三行的号数都是两位——不按最后一行的宽度对齐的话正文会差一格。"""
        body = "\n".join(f"line{i}" for i in range(1, 12))
        (project / "a.txt").write_text(body, encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 9, "end_line": 11}, ctx)

        assert out.splitlines()[:3] == [" 9 | line9", "10 | line10", "11 | line11"]

    def test_line_numbers_start_at_one(self, project, ctx):
        (project / "a.txt").write_text("first\nsecond\n", encoding="utf-8")

        out = read_file({"path": "a.txt", "start_line": 1, "end_line": 1}, ctx)

        assert out.startswith("1 | first")
        assert "second" not in out

    def test_end_line_beyond_the_file_is_clamped(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert read_file({"path": "a.txt", "end_line": 999}, ctx) == "1 | one\n2 | two"

    def test_starting_past_the_end_says_how_long_the_file_is(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

        assert "只有 2 行" in read_file({"path": "a.txt", "start_line": 50}, ctx)

    def test_a_reversed_range_is_rejected(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

        with pytest.raises(ValueError, match="不成立"):
            read_file({"path": "a.txt", "start_line": 3, "end_line": 1}, ctx)

    def test_crlf_files_come_back_with_plain_newlines(self, project, ctx):
        (project / "a.txt").write_bytes(b"one\r\ntwo\r\n")

        assert read_file({"path": "a.txt"}, ctx) == "1 | one\n2 | two"

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

    def test_numbering_eats_into_the_budget_not_the_other_way_round(self, project, ctx):
        """行号是正文的一部分，跟着一起受 MAX_TEXT_CHARS 管着——所以超长文件是**少给几行**，
        而不是"正文给满限额、行号另外加"那样溢出。

        （行数不能太多：write_text 会撑成 CRLF，文件太大就先被"超过 MAX_READ_BYTES"挡掉了。）
        """
        (project / "long.txt").write_text("x\n" * 5000, encoding="utf-8")

        out = read_file({"path": "long.txt"}, ctx)

        body, _, note = out.partition("\n\n[已截断")
        assert len(body) == MAX_TEXT_CHARS  # 切点正好落在限额上，切的是含行号的那份
        assert note

    def test_the_truncation_note_is_not_numbered(self, project, ctx):
        """那句说明不是文件里的一行，编上号会让它看起来像第 N 行。"""
        (project / "long.txt").write_text("x\n" * 5000, encoding="utf-8")

        out = read_file({"path": "long.txt"}, ctx)

        assert out.splitlines()[-1].startswith("[已截断")

    def test_the_line_count_note_survives_truncation(self, project, ctx):
        """"共 N 行"是元信息，接在截断之后——以前它拼在正文里，长文件一截就把它一起吞了。"""
        (project / "long.txt").write_text("x\n" * 5000, encoding="utf-8")

        out = read_file({"path": "long.txt", "start_line": 2}, ctx)

        assert "已截断" in out
        assert "[文件共 5000 行]" in out

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

    def test_a_file_inside_agent_is_refused(self, project, ctx):
        """.agent 里是自己人写的（改前备份、会话 transcript），但装的正是模型看过的一切。"""
        (project / ".agent" / "sessions").mkdir(parents=True)
        (project / ".agent" / "sessions" / "20261011-233012.jsonl").write_text(
            "{}", encoding="utf-8"
        )

        with pytest.raises(SensitiveFileError):
            read_file({"path": ".agent/sessions/20261011-233012.jsonl"}, ctx)

    def test_a_normal_file_is_still_readable(self, project, ctx):
        (project / "app.py").write_text("print(1)", encoding="utf-8")

        assert read_file({"path": "app.py"}, ctx) == "1 | print(1)"

    def test_a_successful_read_is_recorded(self, project, ctx):
        # 用 write_bytes 而不是 write_text：文本模式在 Windows 上会把 \n 翻成 \r\n，
        # 磁盘上就多出字符来，这条测的正是"记的长度等于文件实际有多长"。
        (project / "a.txt").write_bytes(b"hello\n")

        read_file({"path": "a.txt"}, ctx)

        mtime_ns, size_bytes = ctx.read_state.entries[(project / "a.txt").resolve()]
        assert mtime_ns == (project / "a.txt").stat().st_mtime_ns
        assert size_bytes == (project / "a.txt").stat().st_size

    def test_an_unread_file_has_no_record(self, project, ctx):
        (project / "a.txt").write_text("hello\n", encoding="utf-8")

        assert (project / "a.txt").resolve() not in ctx.read_state.entries

    def test_a_failed_read_records_nothing(self, project, ctx):
        """读目录会抛错——那一次什么都没读到，不该留下"读过"的痕迹。"""
        with pytest.raises(IsADirectoryError):
            read_file({"path": "src"}, ctx)

        assert ctx.read_state.entries == {}

    def test_a_bad_line_range_records_nothing(self, project, ctx):
        (project / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

        with pytest.raises(ValueError, match="不成立"):
            read_file({"path": "a.txt", "start_line": 3, "end_line": 1}, ctx)

        assert ctx.read_state.entries == {}

    def test_a_partial_read_records_the_whole_file_length(self, project, ctx):
        """记的是整篇的字节数，不是给模型的那一段——后面要拿它跟磁盘上的现状比。"""
        (project / "a.txt").write_bytes(b"one\ntwo\nthree\n")

        read_file({"path": "a.txt", "start_line": 2, "end_line": 2}, ctx)

        assert ctx.read_state.entries[(project / "a.txt").resolve()][1] == (
            project / "a.txt"
        ).stat().st_size

    def test_a_second_read_updates_the_record(self, project, ctx):
        (project / "a.txt").write_text("hello\n", encoding="utf-8")
        read_file({"path": "a.txt"}, ctx)
        before = ctx.read_state.entries[(project / "a.txt").resolve()]

        (project / "a.txt").write_text("hello world\n", encoding="utf-8")
        read_file({"path": "a.txt"}, ctx)

        assert ctx.read_state.entries[(project / "a.txt").resolve()] != before

    def test_the_record_is_keyed_by_the_resolved_path(self, project, ctx):
        """后面改文件的工具也会把路径解析一遍，键必须是解析后的绝对路径才对得上。"""
        (project / "src" / "b.py").write_text("x\n", encoding="utf-8")

        read_file({"path": "src/../src/b.py"}, ctx)

        assert (project / "src" / "b.py").resolve() in ctx.read_state.entries

    def test_a_zero_line_number_is_rejected(self, project, ctx):
        """行号从 1 数起。以前 0 会被悄悄当成第 1 行，模型不会知道自己数错了。"""
        (project / "a.txt").write_bytes(b"one\ntwo\n")

        with pytest.raises(ValueError, match="从 1 数起"):
            read_file({"path": "a.txt", "start_line": 0}, ctx)

    def test_a_negative_line_number_is_rejected(self, project, ctx):
        (project / "a.txt").write_bytes(b"one\ntwo\n")

        with pytest.raises(ValueError, match="从 1 数起"):
            read_file({"path": "a.txt", "end_line": -3}, ctx)

    def test_a_non_numeric_line_number_is_rejected(self, project, ctx):
        (project / "a.txt").write_bytes(b"one\ntwo\n")

        with pytest.raises(ValueError, match="得是整数"):
            read_file({"path": "a.txt", "start_line": "abc"}, ctx)

    def test_a_float_line_number_is_rejected(self, project, ctx):
        """int(2.7) 会静默截成 2——那一样是把错误藏起来。"""
        (project / "a.txt").write_bytes(b"one\ntwo\nthree\n")

        with pytest.raises(ValueError, match="得是整数"):
            read_file({"path": "a.txt", "start_line": 2.7}, ctx)

    def test_a_boolean_line_number_is_rejected(self, project, ctx):
        """bool 是 int 的子类，不挡的话 True 会变成第 1 行。"""
        (project / "a.txt").write_bytes(b"one\ntwo\n")

        with pytest.raises(ValueError, match="得是整数"):
            read_file({"path": "a.txt", "start_line": True}, ctx)

    def test_a_numeric_string_is_still_accepted(self, project, ctx):
        """模型偶尔会把数字包成字符串，这个不用为难它。"""
        (project / "a.txt").write_bytes(b"one\ntwo\nthree\n")

        assert read_file({"path": "a.txt", "start_line": "3"}, ctx).startswith("3 | three")

    def test_a_file_changed_while_reading_is_reported(self, project, ctx, monkeypatch):
        """读到一半文件被改了：手上这份可能是撕裂的，必须报错，也不能记下"读过"。"""
        target = project / "a.txt"
        target.write_bytes(b"one\ntwo\n")

        real_open = Path.open

        def open_then_touch(self, *args, **kwargs):
            handle = real_open(self, *args, **kwargs)
            # 模拟"打开之后、读完之前，别人动了这个文件"
            bumped = self.stat().st_mtime_ns + 1_000_000_000
            os.utime(self, ns=(bumped, bumped))
            return handle

        monkeypatch.setattr(Path, "open", open_then_touch)

        with pytest.raises(ValueError, match="读的过程中被改过"):
            read_file({"path": "a.txt"}, ctx)
        assert ctx.read_state.entries == {}

    def test_a_file_rewritten_at_the_same_mtime_while_reading_is_reported(
        self, project, ctx, monkeypatch
    ):
        """mtime 一样、长度变了，照样是撕裂的——只比 mtime 会漏掉这一次。

        同一个时钟刻度里的两次写，时间戳会完全一样（实测这台机器上 10ms 以内经常撞），
        这里把 mtime 拨回读之前的值，模拟的就是那种情况。关键是读句柄得拿到**旧**字节
        （真读到一半时文件被换掉就是这个样子），所以这里把旧内容拍个快照返回。
        """
        target = project / "a.txt"
        target.write_bytes(b"one\ntwo\n")

        real_open = Path.open

        def open_then_rewrite(self, mode="r", *args, **kwargs):
            if "w" in mode or "a" in mode or "+" in mode:
                return real_open(self, mode, *args, **kwargs)  # 写走真身，不递归
            with real_open(self, mode, *args, **kwargs) as source:
                snapshot = source.read()
            same_mtime = self.stat().st_mtime_ns  # 读之前量到的就是这个值
            with real_open(self, "wb") as writer:
                writer.write(b"one\ntwo\nthree\n")
            os.utime(self, ns=(same_mtime, same_mtime))
            return io.BytesIO(snapshot)

        monkeypatch.setattr(Path, "open", open_then_rewrite)

        with pytest.raises(ValueError, match="读的过程中被改过"):
            read_file({"path": "a.txt"}, ctx)
        assert ctx.read_state.entries == {}


    def test_a_path_through_a_symlink_is_refused(self, project, ctx):
        """顺着链接读，读到的可能是另一个文件——模型以为读的是 A，实际读的是 B。"""
        (project / "real.txt").write_bytes(b"real\n")
        try:
            (project / "link.txt").symlink_to(project / "real.txt")
        except OSError:
            pytest.skip("这个环境建不了符号链接（Windows 上需要开发者模式或管理员权限）")

        with pytest.raises(ValueError, match="符号链接"):
            read_file({"path": "link.txt"}, ctx)

    def test_the_symlink_check_looks_at_every_path_component(self, project, ctx, monkeypatch):
        """路径上**任何一段**是链接都要拦，不只是最后一段。

        用打桩而不是真建链接：Windows 上建链接要权限，这条验的是判断逻辑本身，
        顺便钉住"走的是没解析过的那条路径"——解析之后链接信息就没了。
        """
        real_is_symlink = Path.is_symlink

        def fake_is_symlink(self):
            return self.name == "src" or real_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

        with pytest.raises(ValueError, match="符号链接"):
            read_file({"path": "src/cli.py"}, ctx)


class TestWriteFile:
    def test_creates_a_file_that_did_not_exist(self, project, ctx):
        message = write_file({"path": "new.py", "content": "print(1)\n"}, ctx)

        assert (project / "new.py").read_bytes() == b"print(1)\n"
        assert "新建了" in message

    def test_a_new_file_needs_no_prior_read(self, project, ctx):
        """不存在的文件没有"会盖掉什么"的问题，所以不要求先读。"""
        assert ctx.read_state.entries == {}

        write_file({"path": "new.py", "content": "x\n"}, ctx)

        assert (project / "new.py").exists()

    def test_overwrites_a_file_that_was_read(self, project, ctx):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        message = write_file({"path": "a.txt", "content": "new\n"}, ctx)

        assert (project / "a.txt").read_bytes() == b"new\n"
        assert "覆盖了" in message

    def test_refuses_to_overwrite_a_file_that_was_never_read(self, project, ctx):
        """整篇盖掉一份模型从没看过的内容，而且它自己还不知道——这就是要挡的事。"""
        (project / "a.txt").write_bytes(b"old\n")

        with pytest.raises(StaleReadError, match="还没读过"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert (project / "a.txt").read_bytes() == b"old\n"

    def test_refuses_to_overwrite_a_file_changed_since_it_was_read(self, project, ctx):
        target = project / "a.txt"
        target.write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        # 明确把 mtime 推远，而不是指望"再写一次文件系统就一定更新 mtime"。那条在
        # Windows 上偶尔不成立（这条用例因此闪过一次，单独连跑 20 遍却复现不出来）。
        # 这里要验的是"读完之后文件被动了就拒绝"，不是文件系统怎么记时间戳。
        bumped = target.stat().st_mtime_ns + 1_000_000_000
        os.utime(target, ns=(bumped, bumped))

        with pytest.raises(StaleReadError, match="被改过了"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert target.read_bytes() == b"old\n"

    def test_refuses_when_only_the_length_changed(self, project, ctx):
        """把 mtime 拨回原样，长度那道闸还拦得住——这正是两个信号一起比的意义。

        不是假想：两次写落在同一个时钟刻度里，时间戳会**一模一样**（实测这台机器上
        间隔 10ms 以内经常撞）。只比 mtime 的话，这一段就被静默盖掉了。
        """
        target = project / "a.txt"
        target.write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)
        recorded_mtime = target.stat().st_mtime_ns

        target.write_bytes(b"a much longer line\n")
        os.utime(target, ns=(recorded_mtime, recorded_mtime))

        with pytest.raises(StaleReadError, match="被改过了"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert target.read_bytes() == b"a much longer line\n"

    def test_the_backup_keeps_the_previous_content(self, project, ctx):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        write_file({"path": "a.txt", "content": "new\n"}, ctx)

        backups = list((project / ".agent" / ".history").rglob("a.txt.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == b"old\n"

    def test_the_history_directory_is_hidden_from_listing(self, project, ctx):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)
        write_file({"path": "a.txt", "content": "new\n"}, ctx)

        assert ".agent" not in list_files({}, ctx)

    def test_no_temp_file_is_left_behind(self, project, ctx):
        write_file({"path": "a.txt", "content": "hi\n"}, ctx)

        assert [p.name for p in project.iterdir() if p.suffix == ".tmp"] == []

    def test_a_failed_replace_leaves_the_original_and_cleans_up(self, project, ctx, monkeypatch):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        def explode(*args, **kwargs):
            raise OSError("模拟顶替失败")

        monkeypatch.setattr(os, "replace", explode)

        with pytest.raises(ReplaceFailedError, match="顶替"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)

        assert (project / "a.txt").read_bytes() == b"old\n"
        assert [p.name for p in project.iterdir() if p.suffix == ".tmp"] == []

    def test_a_file_just_written_can_be_written_again(self, project, ctx):
        """写完就等于"我知道它现在是什么"了，第二次不用再读一遍。"""
        write_file({"path": "a.txt", "content": "one\n"}, ctx)

        write_file({"path": "a.txt", "content": "two\n"}, ctx)

        assert (project / "a.txt").read_bytes() == b"two\n"

    def test_content_is_written_verbatim(self, project, ctx):
        """newline="" ：模型给什么就写什么，不在 Windows 上偷偷把 \\n 翻成 \\r\\n。"""
        write_file({"path": "a.txt", "content": "one\ntwo\n"}, ctx)

        assert (project / "a.txt").read_bytes() == b"one\ntwo\n"

    def test_a_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "x.txt"

        with pytest.raises(OutOfSandboxError):
            write_file({"path": str(outside), "content": "x"}, ctx)
        assert not outside.exists()

    def test_writing_to_a_directory_is_rejected(self, project, ctx):
        with pytest.raises(IsADirectoryError):
            write_file({"path": "src", "content": "x"}, ctx)

    def test_a_missing_parent_directory_is_reported(self, project, ctx):
        """这个工具不替模型建目录——多半是路径写错了，包出来比悄悄建一个树好。"""
        with pytest.raises(FileNotFoundError, match="上级目录"):
            write_file({"path": "nope/deep/x.txt", "content": "x"}, ctx)

    def test_non_string_content_is_rejected(self, project, ctx):
        with pytest.raises(ValueError, match="得是字符串"):
            write_file({"path": "a.txt", "content": 123}, ctx)

    def test_a_sensitive_file_is_refused(self, project, ctx):
        """读那边靠这道防泄露，写这边靠它防破坏。"""
        with pytest.raises(SensitiveFileError, match="敏感位置"):
            write_file({"path": ".env", "content": "KEY=stolen\n"}, ctx)
        assert (project / ".env").read_bytes() == b"KEY=1"

    def test_creating_a_new_sensitive_file_is_refused(self, project, ctx):
        """新建那条路没有 read-state 兜底，全靠这道。"""
        with pytest.raises(SensitiveFileError, match="敏感位置"):
            write_file({"path": "id_rsa", "content": "fake key\n"}, ctx)
        assert not (project / "id_rsa").exists()

    def test_creating_a_file_in_a_sensitive_directory_is_refused(self, project, ctx):
        """这条是真会出事的：.git/hooks/ 下面的脚本会在下次提交时被执行。"""
        (project / ".git" / "hooks").mkdir(parents=True)

        with pytest.raises(SensitiveFileError, match="敏感位置"):
            write_file({"path": ".git/hooks/pre-commit", "content": "#!/bin/sh\n"}, ctx)
        assert not (project / ".git" / "hooks" / "pre-commit").exists()

    def test_the_temp_file_sits_beside_the_target_with_a_random_name(
        self, project, ctx, monkeypatch
    ):
        """同目录是为了 os.replace 真的是原子的（跨盘就不是了），随机名是为了别撞上真文件。"""
        seen: dict[str, Path] = {}
        real_replace = os.replace

        def capture(src, dst):
            seen["temp"] = Path(src)
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", capture)

        write_file({"path": "a.txt", "content": "x\n"}, ctx)

        assert seen["temp"].parent == project
        assert seen["temp"].name != "a.txt"

    def test_a_path_through_a_symlink_is_refused(self, project, ctx):
        """顺着链接写，改的是它指向的那个文件——方向可能完全出乎模型意料。"""
        (project / "real.txt").write_bytes(b"original\n")
        try:
            (project / "link.txt").symlink_to(project / "real.txt")
        except OSError:
            pytest.skip("这个环境建不了符号链接（Windows 上需要开发者模式或管理员权限）")

        with pytest.raises(ValueError, match="符号链接"):
            write_file({"path": "link.txt", "content": "new\n"}, ctx)
        assert (project / "real.txt").read_bytes() == b"original\n"

    def test_the_symlink_check_looks_at_every_path_component(self, project, ctx, monkeypatch):
        """路径上**任何一段**是链接都要拦，不只是最后一段。

        用打桩而不是真建链接：Windows 上建链接要权限，这条验的是判断逻辑本身，顺便
        钉住"走的是没解析过的那条路径"——解析之后链接信息就没了。
        """
        real_is_symlink = Path.is_symlink

        def fake_is_symlink(self):
            return self.name == "src" or real_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

        with pytest.raises(ValueError, match="符号链接"):
            write_file({"path": "src/x.txt", "content": "x\n"}, ctx)

    @pytest.mark.skipif(os.name == "nt", reason="Windows 的权限位和 POSIX 不是一回事")
    def test_overwriting_keeps_the_original_file_mode(self, project, ctx):
        target = project / "a.txt"
        target.write_bytes(b"old\n")
        os.chmod(target, 0o640)
        read_file({"path": "a.txt"}, ctx)

        write_file({"path": "a.txt", "content": "new\n"}, ctx)

        assert target.stat().st_mode & 0o777 == 0o640

    def test_a_backup_failure_says_the_original_was_not_touched(self, project, ctx, monkeypatch):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        def explode(*args, **kwargs):
            raise OSError("模拟备份失败")

        monkeypatch.setattr(shutil, "copy2", explode)

        with pytest.raises(BackupFailedError, match="没被改动"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert (project / "a.txt").read_bytes() == b"old\n"

    def test_a_write_failure_says_the_original_was_not_touched(self, project, ctx, monkeypatch):
        (project / "a.txt").write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        def explode(*args, **kwargs):
            raise OSError("模拟建临时文件失败")

        monkeypatch.setattr(tempfile, "mkstemp", explode)

        with pytest.raises(WriteFailedError, match="没被改动"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert (project / "a.txt").read_bytes() == b"old\n"

    def test_the_state_check_happens_after_the_temp_file_is_written(
        self, project, ctx, monkeypatch
    ):
        """状态检查放在顶替之前，不是函数开头。

        这条在"临时文件已经建好"之后才去改目标文件的 mtime：检查要是搁在函数开头，
        它看不到这次改动、写入会照常成功；搁在顶替之前才拦得住。顺带证明拦下来时
        原文件没被碰、临时文件也没留下。
        """
        target = project / "a.txt"
        target.write_bytes(b"old\n")
        read_file({"path": "a.txt"}, ctx)

        real_mkstemp = tempfile.mkstemp

        def mkstemp_then_touch(*args, **kwargs):
            result = real_mkstemp(*args, **kwargs)
            bumped = target.stat().st_mtime_ns + 1_000_000_000
            os.utime(target, ns=(bumped, bumped))
            return result

        monkeypatch.setattr(tempfile, "mkstemp", mkstemp_then_touch)

        with pytest.raises(StaleReadError, match="被改过了"):
            write_file({"path": "a.txt", "content": "new\n"}, ctx)
        assert target.read_bytes() == b"old\n"
        assert [p.name for p in project.iterdir() if p.suffix == ".tmp"] == []


class TestEditFile:
    def test_replaces_the_matched_text(self, project, ctx):
        (project / "a.py").write_bytes(b"def f():\n    return 1\n")
        read_file({"path": "a.py"}, ctx)

        message = edit_file(
            {"path": "a.py", "old_string": "return 1", "new_string": "return 2"}, ctx
        )

        assert (project / "a.py").read_bytes() == b"def f():\n    return 2\n"
        assert "替换 1 处" in message

    def test_matching_includes_indentation_and_whitespace(self, project, ctx):
        (project / "a.py").write_bytes(b"    return 1\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "    return 1", "new_string": "    return 2"}, ctx)

        assert (project / "a.py").read_bytes() == b"    return 2\n"

    def test_a_near_miss_does_not_match(self, project, ctx):
        """差一个空格就是另一个字符串——匹配是逐字符的，不是"看着差不多"。

        （注意匹配是子串匹配，所以缩进不必包含在 old_string 里；要的是一个字节不差。）
        """
        (project / "a.py").write_bytes(b"    return  1\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="找不到 old_string"):
            edit_file({"path": "a.py", "old_string": "return 1", "new_string": "x"}, ctx)

    def test_only_the_first_occurrence_is_replaced_by_default(self, project, ctx):
        (project / "a.py").write_bytes(b"x\nx\nx\n")
        read_file({"path": "a.py"}, ctx)

        message = edit_file({"path": "a.py", "old_string": "x", "new_string": "y"}, ctx)

        assert (project / "a.py").read_bytes() == b"y\nx\nx\n"
        assert "出现了 3 次" in message
        assert "只替换了第一处" in message

    def test_replace_all_replaces_every_occurrence(self, project, ctx):
        (project / "a.py").write_bytes(b"x\nx\nx\n")
        read_file({"path": "a.py"}, ctx)

        message = edit_file(
            {"path": "a.py", "old_string": "x", "new_string": "y", "replace_all": True}, ctx
        )

        assert (project / "a.py").read_bytes() == b"y\ny\ny\n"
        assert "替换 3 处" in message

    def test_deleting_text_works(self, project, ctx):
        (project / "a.py").write_bytes(b"keep\ngo away\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "go away\n", "new_string": ""}, ctx)

        assert (project / "a.py").read_bytes() == b"keep\n"

    def test_it_must_have_been_read_first(self, project, ctx):
        (project / "a.py").write_bytes(b"x\n")

        with pytest.raises(StaleReadError, match="还没读过"):
            edit_file({"path": "a.py", "old_string": "x", "new_string": "y"}, ctx)

    def test_refuses_if_the_file_changed_after_it_was_read(self, project, ctx):
        target = project / "a.py"
        target.write_bytes(b"x\n")
        read_file({"path": "a.py"}, ctx)

        bumped = target.stat().st_mtime_ns + 1_000_000_000
        os.utime(target, ns=(bumped, bumped))

        with pytest.raises(StaleReadError, match="被改过了"):
            edit_file({"path": "a.py", "old_string": "x", "new_string": "y"}, ctx)
        assert target.read_bytes() == b"x\n"

    def test_it_backs_up_the_previous_content(self, project, ctx):
        (project / "a.py").write_bytes(b"old\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "old", "new_string": "new"}, ctx)

        backups = list((project / ".agent" / ".history").rglob("a.py.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == b"old\n"

    def test_a_file_just_edited_can_be_edited_again(self, project, ctx):
        (project / "a.py").write_bytes(b"one\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "one", "new_string": "two"}, ctx)
        edit_file({"path": "a.py", "old_string": "two", "new_string": "three"}, ctx)

        assert (project / "a.py").read_bytes() == b"three\n"

    def test_a_missing_file_points_at_write_file(self, project, ctx):
        with pytest.raises(FileNotFoundError, match="用 write_file"):
            edit_file({"path": "nope.py", "old_string": "x", "new_string": "y"}, ctx)

    def test_a_directory_is_rejected(self, project, ctx):
        with pytest.raises(IsADirectoryError):
            edit_file({"path": "src", "old_string": "x", "new_string": "y"}, ctx)

    def test_a_sensitive_path_is_rejected(self, project, ctx):
        with pytest.raises(SensitiveFileError, match="敏感位置"):
            edit_file({"path": ".env", "old_string": "KEY", "new_string": "X"}, ctx)

    def test_a_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "x.txt"

        with pytest.raises(OutOfSandboxError):
            edit_file({"path": str(outside), "old_string": "x", "new_string": "y"}, ctx)

    def test_an_empty_old_string_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"x\n")

        with pytest.raises(ValueError, match="不能为空"):
            edit_file({"path": "a.py", "old_string": "", "new_string": "y"}, ctx)

    def test_an_unchanged_edit_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"x\n")

        with pytest.raises(ValueError, match="一模一样"):
            edit_file({"path": "a.py", "old_string": "x", "new_string": "x"}, ctx)

    def test_non_string_arguments_are_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"x\n")

        with pytest.raises(ValueError, match="得是字符串"):
            edit_file({"path": "a.py", "old_string": 123, "new_string": "y"}, ctx)

    def test_two_edits_in_the_same_second_keep_separate_backups(self, project, ctx):
        """备份名只精确到秒的话，同一秒里的第二次会把第一次覆盖掉。

        而第一次那份才是这轮改动**之前**的原始内容——真丢了就没得回退了。连改两次是
        很正常的事（实跑里撞到过），不是边角情况。
        """
        (project / "a.py").write_bytes(b"v1\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "v1", "new_string": "v2"}, ctx)
        edit_file({"path": "a.py", "old_string": "v2", "new_string": "v3"}, ctx)

        backups = sorted((project / ".agent" / ".history").rglob("a.py.*"))
        assert len(backups) == 2
        assert [backup.read_bytes() for backup in backups] == [b"v1\n", b"v2\n"]

    def test_no_temp_file_is_left_behind(self, project, ctx):
        (project / "a.py").write_bytes(b"x\n")
        read_file({"path": "a.py"}, ctx)

        edit_file({"path": "a.py", "old_string": "x", "new_string": "y"}, ctx)

        assert [p.name for p in project.iterdir() if p.suffix == ".tmp"] == []


class TestReplaceLines:
    def test_replaces_one_line(self, project, ctx):
        (project / "a.py").write_bytes(b"one\ntwo\nthree\n")
        read_file({"path": "a.py"}, ctx)

        message = replace_lines(
            {"path": "a.py", "start_line": 2, "end_line": 2, "new_text": "TWO"}, ctx
        )

        assert (project / "a.py").read_bytes() == b"one\nTWO\nthree\n"
        assert "第 2-2 行换成了 1 行" in message

    def test_replaces_a_range(self, project, ctx):
        (project / "a.py").write_bytes(b"one\ntwo\nthree\nfour\n")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 2, "end_line": 3, "new_text": "X"}, ctx)

        assert (project / "a.py").read_bytes() == b"one\nX\nfour\n"

    def test_new_text_can_be_longer(self, project, ctx):
        (project / "a.py").write_bytes(b"a\nb\n")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 1, "end_line": 1, "new_text": "x\ny\nz"}, ctx)

        assert (project / "a.py").read_bytes() == b"x\ny\nz\nb\n"

    def test_a_missing_trailing_newline_is_added(self, project, ctx):
        """不补的话，下一行会和它粘成一行。"""
        (project / "a.py").write_bytes(b"a\nb\nc\n")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 2, "end_line": 2, "new_text": "X"}, ctx)

        assert (project / "a.py").read_bytes() == b"a\nX\nc\n"

    def test_an_empty_new_text_deletes_the_range(self, project, ctx):
        (project / "a.py").write_bytes(b"keep\ndrop me\nalso drop\nkeep too\n")
        read_file({"path": "a.py"}, ctx)

        message = replace_lines(
            {"path": "a.py", "start_line": 2, "end_line": 3, "new_text": ""}, ctx
        )

        assert (project / "a.py").read_bytes() == b"keep\nkeep too\n"
        assert "删掉了第 2-3 行" in message

    def test_the_empty_range_inserts(self, project, ctx):
        """end_line = start_line - 1：不替换任何行，只在这个位置插入。"""
        (project / "a.py").write_bytes(b"a\nc\n")
        read_file({"path": "a.py"}, ctx)

        message = replace_lines(
            {"path": "a.py", "start_line": 2, "end_line": 1, "new_text": "b"}, ctx
        )

        assert (project / "a.py").read_bytes() == b"a\nb\nc\n"
        assert "插了 1 行" in message

    def test_appending_at_the_end(self, project, ctx):
        (project / "a.py").write_bytes(b"a\n")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 2, "end_line": 1, "new_text": "b"}, ctx)

        assert (project / "a.py").read_bytes() == b"a\nb\n"

    def test_an_empty_file_can_take_an_insert(self, project, ctx):
        (project / "a.py").write_bytes(b"")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 1, "end_line": 0, "new_text": "first"}, ctx)

        assert (project / "a.py").read_bytes() == b"first\n"

    def test_a_final_line_without_a_newline_survives_untouched(self, project, ctx):
        """区间外的内容按字节原样拼回去——包括"最后一行没有换行"这种。"""
        (project / "a.py").write_bytes(b"one\ntwo")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 1, "end_line": 1, "new_text": "X"}, ctx)

        assert (project / "a.py").read_bytes() == b"X\ntwo"

    def test_a_start_past_the_end_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\ntwo\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="超出了文件"):
            replace_lines({"path": "a.py", "start_line": 9, "end_line": 9, "new_text": "x"}, ctx)

    def test_an_end_past_the_end_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\ntwo\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="超出了文件"):
            replace_lines({"path": "a.py", "start_line": 1, "end_line": 9, "new_text": "x"}, ctx)

    def test_a_reversed_range_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\ntwo\nthree\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="小太多"):
            replace_lines({"path": "a.py", "start_line": 3, "end_line": 1, "new_text": "x"}, ctx)

    def test_a_zero_start_line_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="从 1 数起"):
            replace_lines({"path": "a.py", "start_line": 0, "end_line": 1, "new_text": "x"}, ctx)

    def test_a_negative_end_line_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\n")
        read_file({"path": "a.py"}, ctx)

        with pytest.raises(ValueError, match="不能是负数"):
            replace_lines({"path": "a.py", "start_line": 1, "end_line": -1, "new_text": "x"}, ctx)

    def test_a_missing_file_points_at_write_file(self, project, ctx):
        with pytest.raises(FileNotFoundError, match="用 write_file"):
            replace_lines({"path": "nope.py", "start_line": 1, "end_line": 1, "new_text": "x"}, ctx)

    def test_it_must_have_been_read_first(self, project, ctx):
        (project / "a.py").write_bytes(b"one\n")

        with pytest.raises(StaleReadError, match="还没读过"):
            replace_lines({"path": "a.py", "start_line": 1, "end_line": 1, "new_text": "x"}, ctx)

    def test_it_backs_up_the_previous_content(self, project, ctx):
        (project / "a.py").write_bytes(b"old\n")
        read_file({"path": "a.py"}, ctx)

        replace_lines({"path": "a.py", "start_line": 1, "end_line": 1, "new_text": "new"}, ctx)

        backups = list((project / ".agent" / ".history").rglob("a.py.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == b"old\n"

    def test_a_sensitive_path_is_rejected(self, project, ctx):
        with pytest.raises(SensitiveFileError, match="敏感位置"):
            replace_lines({"path": ".env", "start_line": 1, "end_line": 1, "new_text": "x"}, ctx)

    def test_a_path_outside_the_root_is_rejected(self, project, ctx, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "x.txt"

        with pytest.raises(OutOfSandboxError):
            replace_lines(
                {"path": str(outside), "start_line": 1, "end_line": 1, "new_text": "x"}, ctx
            )

    def test_non_string_new_text_is_rejected(self, project, ctx):
        (project / "a.py").write_bytes(b"one\n")

        with pytest.raises(ValueError, match="得是字符串"):
            replace_lines({"path": "a.py", "start_line": 1, "end_line": 1, "new_text": 1}, ctx)


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
