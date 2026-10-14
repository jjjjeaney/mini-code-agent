"""skills 那一摊：扫描、解析、索引、坏文件，以及那两个工具。全部本地执行。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mini_code_agent.skills import (
    INDEX_NAME,
    SKILL_DIR,
    SKILL_FILE,
    attachments,
    index_text,
    names,
    read,
    sync_index,
)
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import MAX_READ_BYTES, OutOfSandboxError
from mini_code_agent.tools.skill import read_skill, read_skill_file


def a_skill(
    tmp_path, name="release", *, description="发版前按这个走一遍", body="第一步：跑测试\n", **files
):
    """造一份 skill（可选带上附件）。返回它那个目录。"""
    directory = tmp_path / SKILL_DIR / name
    directory.mkdir(parents=True)
    (directory / SKILL_FILE).write_text(
        f"---\ndescription: {description}\n---\n\n{body}", encoding="utf-8"
    )
    for filename, text in files.items():
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


class TestNames:
    def test_only_directories_with_a_skill_file_count(self, tmp_path):
        a_skill(tmp_path, "release")
        (tmp_path / SKILL_DIR / "随便一个目录").mkdir()  # 没有 SKILL.md，不算
        (tmp_path / SKILL_DIR / "杂七杂八.txt").write_text("x", encoding="utf-8")

        assert names(tmp_path) == ["release"]

    def test_they_are_sorted(self, tmp_path):
        a_skill(tmp_path, "release")
        a_skill(tmp_path, "cleanup")

        assert names(tmp_path) == ["cleanup", "release"]

    def test_no_directory_means_none(self, tmp_path):
        assert names(tmp_path) == []


class TestRead:
    def test_it_gives_back_the_description_and_the_body(self, tmp_path):
        a_skill(tmp_path)

        skill = read(tmp_path, "release")

        assert skill.description == "发版前按这个走一遍"
        assert skill.body == "第一步：跑测试"

    def test_an_unknown_name_says_what_there_is(self, tmp_path):
        a_skill(tmp_path, "release")

        with pytest.raises(FileNotFoundError, match="release"):
            read(tmp_path, "nope")

    @pytest.mark.parametrize("name", ["../evil", "a/b", "Release", "-x", ""])
    def test_a_bad_name_is_refused(self, tmp_path, name):
        """名字要当目录名用，不校验的话 `../` 能顺着跑出去。"""
        with pytest.raises(ValueError, match="不是有效的 skill 名"):
            read(tmp_path, name)

    def test_a_missing_description_is_refused(self, tmp_path):
        directory = tmp_path / SKILL_DIR / "release"
        directory.mkdir(parents=True)
        (directory / SKILL_FILE).write_text("---\n---\n\n正文\n", encoding="utf-8")

        with pytest.raises(ValueError, match="description"):
            read(tmp_path, "release")

    def test_an_empty_body_is_refused(self, tmp_path):
        """正文才是模型要照着做的东西——空的就等于没这份 skill。"""
        a_skill(tmp_path, body="   \n")

        with pytest.raises(ValueError, match="正文是空的"):
            read(tmp_path, "release")


class TestAttachments:
    def test_it_lists_everything_but_the_skill_file(self, tmp_path):
        a_skill(tmp_path, "release", **{"check.sh": "echo ok", "templates/note.md": "模板"})

        files, hidden = attachments(read(tmp_path, "release"))

        assert files == ["check.sh", "templates/note.md"]
        assert hidden == 0

    def test_a_skill_without_attachments_has_none(self, tmp_path):
        a_skill(tmp_path)

        assert attachments(read(tmp_path, "release")) == ([], 0)

    def test_the_listing_is_capped(self, tmp_path, monkeypatch):
        """附件多起来的时候，把名字全铺进上下文没意义——报个数就够。"""
        monkeypatch.setattr("mini_code_agent.skills.store.MAX_LISTED_ATTACHMENTS", 2)
        a_skill(tmp_path, **{f"f{i}.txt": "x" for i in range(5)})

        files, hidden = attachments(read(tmp_path, "release"))

        assert files == ["f0.txt", "f1.txt"]
        assert hidden == 3


class TestIndex:
    def test_it_lists_the_name_and_the_description_only(self, tmp_path):
        a_skill(tmp_path)

        text = index_text(tmp_path)

        assert "release" in text
        assert "发版前按这个走一遍" in text
        assert "第一步：跑测试" not in text  # 正文不在索引里，那要按需取

    def test_a_hand_broken_skill_is_skipped(self, tmp_path):
        """一份索引不该被一个坏文件整体拖垮。"""
        a_skill(tmp_path, "good")
        broken = tmp_path / SKILL_DIR / "broken"
        broken.mkdir(parents=True)
        (broken / SKILL_FILE).write_text("没有 frontmatter", encoding="utf-8")

        text = index_text(tmp_path)

        assert "good" in text
        assert "broken" not in text

    def test_no_skills_means_an_empty_index(self, tmp_path):
        assert index_text(tmp_path) == ""


class TestSyncIndex:
    def test_it_writes_the_index(self, tmp_path):
        a_skill(tmp_path)

        assert sync_index(tmp_path) is True
        assert "release" in (tmp_path / SKILL_DIR / INDEX_NAME).read_text(encoding="utf-8")

    def test_a_hand_added_skill_lands_in_the_index(self, tmp_path):
        sync_index(tmp_path)
        a_skill(tmp_path, "release")

        assert sync_index(tmp_path) is True
        assert "release" in (tmp_path / SKILL_DIR / INDEX_NAME).read_text(encoding="utf-8")

    def test_it_does_nothing_when_already_in_sync(self, tmp_path):
        """启动时每次都会对一次——稳态下必须一个字节都不动。"""
        a_skill(tmp_path)
        sync_index(tmp_path)

        assert sync_index(tmp_path) is False

    def test_no_directory_means_nothing_happens(self, tmp_path):
        """不因为跑一次 agent，就在别人项目里造一个 .agent/skill/。"""
        assert sync_index(tmp_path) is False
        assert not (tmp_path / SKILL_DIR).exists()


class TestReadSkillTool:
    def test_it_returns_the_body(self, ctx):
        a_skill(ctx.cwd)

        assert read_skill({"name": "release"}, ctx) == "第一步：跑测试"

    def test_it_lists_the_attachments_after_the_body(self, ctx):
        """不列的话模型根本不知道有附件。"""
        a_skill(ctx.cwd, **{"check.sh": "echo ok"})

        out = read_skill({"name": "release"}, ctx)

        assert out.startswith("第一步：跑测试")
        assert "check.sh" in out
        assert "read_skill_file" in out

    def test_no_attachment_section_when_there_are_none(self, ctx):
        a_skill(ctx.cwd)

        assert "read_skill_file" not in read_skill({"name": "release"}, ctx)


class TestReadSkillFileTool:
    def test_it_reads_an_attachment(self, ctx):
        a_skill(ctx.cwd, **{"check.sh": "echo ok"})

        assert read_skill_file({"name": "release", "path": "check.sh"}, ctx) == "echo ok"

    def test_it_reads_one_in_a_subdirectory(self, ctx):
        a_skill(ctx.cwd, **{"templates/note.md": "模板正文"})

        out = read_skill_file({"name": "release", "path": "templates/note.md"}, ctx)

        assert out == "模板正文"

    @pytest.mark.parametrize("path", ["../SKILL.md", "../../secret.txt", "sub/../../SKILL.md"])
    def test_it_cannot_reach_outside_the_skill_directory(self, ctx, path):
        """**这条最要紧**：围栏是"这个 skill 自己的目录"。"""
        a_skill(ctx.cwd, **{"sub/x.txt": "x"})
        (ctx.cwd / "secret.txt").write_text("机密", encoding="utf-8")

        with pytest.raises(OutOfSandboxError):
            read_skill_file({"name": "release", "path": path}, ctx)

    def test_an_absolute_path_is_refused(self, ctx, tmp_path):
        a_skill(ctx.cwd)
        outside = tmp_path / "outside.txt"
        outside.write_text("机密", encoding="utf-8")

        with pytest.raises(OutOfSandboxError):
            read_skill_file({"name": "release", "path": str(outside)}, ctx)

    def test_a_missing_attachment_says_where_it_looked(self, ctx):
        a_skill(ctx.cwd)

        with pytest.raises(FileNotFoundError, match="release/"):
            read_skill_file({"name": "release", "path": "nope.sh"}, ctx)

    def test_a_directory_is_refused(self, ctx):
        a_skill(ctx.cwd, **{"templates/note.md": "x"})

        with pytest.raises(IsADirectoryError):
            read_skill_file({"name": "release", "path": "templates"}, ctx)

    def test_a_huge_attachment_is_refused(self, ctx):
        a_skill(ctx.cwd, **{"big.bin": "x" * (MAX_READ_BYTES + 1)})

        with pytest.raises(ValueError, match="太大"):
            read_skill_file({"name": "release", "path": "big.bin"}, ctx)

    def test_a_binary_attachment_says_so_instead_of_leaking_a_decode_error(self, ctx):
        """附件里放个图片模板是正常的——该说"读不了"，而不是甩一段解码错误。"""
        a_skill(ctx.cwd)
        (ctx.cwd / SKILL_DIR / "release" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x80")

        with pytest.raises(ValueError, match="不是 UTF-8 文本"):
            read_skill_file({"name": "release", "path": "logo.png"}, ctx)

    def test_an_unknown_skill_is_reported_first(self, ctx):
        with pytest.raises(FileNotFoundError, match="没有叫"):
            read_skill_file({"name": "nope", "path": "x.sh"}, ctx)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Windows 建软链要开发者模式或管理员权限"
    )
    def test_a_symlink_is_refused(self, ctx):
        a_skill(ctx.cwd)
        outside = ctx.cwd / "outside.txt"
        outside.write_text("机密", encoding="utf-8")
        os.symlink(outside, ctx.cwd / SKILL_DIR / "release" / "link.txt")

        with pytest.raises(ValueError, match="符号链接"):
            read_skill_file({"name": "release", "path": "link.txt"}, ctx)


def test_the_skill_file_name_is_the_documented_one():
    """`.agent/skill/<名字>/SKILL.md` 这个形状是写在文档和 skill 自己的说明里的。"""
    assert SKILL_FILE == "SKILL.md"
    assert Path(".agent") / "skill" == SKILL_DIR
