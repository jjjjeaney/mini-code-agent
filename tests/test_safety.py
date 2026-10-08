"""safety 的单元测试：路径解析、越界判断、文本/二进制判定逐个钉住，全部本地执行。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_code_agent.tools.safety import (
    MAX_TEXT_CHARS,
    detect_encoding,
    ensure_text_file,
    is_within,
    resolve_within,
    should_skip,
    truncate_text,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    return tmp_path


class TestResolveWithin:
    """只解析路径，不做任何越界判断。"""

    def test_relative_path_lands_under_root(self, root):
        assert resolve_within(root, "src/main.py") == (root / "src" / "main.py").resolve()

    def test_dotdot_is_normalized(self, root):
        assert resolve_within(root, "src/../src/main.py") == (root / "src" / "main.py").resolve()

    def test_dot_means_the_root_itself(self, root):
        assert resolve_within(root, ".") == root.resolve()

    def test_result_is_absolute_and_has_no_dotdot(self, root):
        result = resolve_within(root, "src/main.py")

        assert result.is_absolute()
        assert ".." not in result.parts

    def test_escaping_paths_are_still_resolved_not_rejected(self, root):
        """越界不归它管：照常解析，判不判是 is_within 的事。"""
        assert resolve_within(root, "../outside.txt") == (root.parent / "outside.txt").resolve()

    def test_absolute_path_is_taken_as_is(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "secret.txt"

        assert resolve_within(root, outside) == outside.resolve()


class TestIsWithin:
    def test_relative_path_inside(self, root):
        assert is_within(root, resolve_within(root, "src/main.py")) is True

    def test_dot_is_inside(self, root):
        assert is_within(root, resolve_within(root, ".")) is True

    def test_dotdot_escape_is_outside(self, root):
        assert is_within(root, resolve_within(root, "../outside.txt")) is False

    def test_sibling_with_the_same_prefix_is_outside(self, root):
        """字符串 startswith 会在这里放行，is_relative_to 不会。"""
        assert is_within(root, Path(str(root) + "-evil")) is False

    def test_absolute_path_outside_is_outside(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "secret.txt"

        assert is_within(root, outside.resolve()) is False

    def test_symlink_pointing_outside_is_outside(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("这个环境建不了符号链接（Windows 上需要开发者模式或管理员权限）")

        assert is_within(root, resolve_within(root, "link/secret.txt")) is False

    def test_unresolved_candidate_with_dotdot_is_why_you_must_resolve_first(self, root):
        """钉住"candidate 必须是解析过的"这个约定——不是想要的行为，是要提防的坑。

        没解析过的路径里 `..` 只是字面分量，is_relative_to 会数着分量判定，
        于是明明指向圈外的路径被判成"在圈里"。这就是要先 resolve 的原因。
        """
        sneaky = root / "src" / ".." / ".." / "evil.txt"

        assert is_within(root, sneaky) is True
        assert not sneaky.resolve().is_relative_to(root.resolve())

    @pytest.mark.skipif(os.name != "nt", reason="Windows 的路径比较本来就大小写不敏感")
    def test_case_differences_do_not_count_as_an_escape_on_windows(self, root):
        assert is_within(root, Path(str(root).upper() + r"\SRC\MAIN.PY")) is True

    @pytest.mark.skipif(os.name != "nt", reason="盘符相对路径是 Windows 特有的写法")
    def test_windows_drive_relative_path_is_not_a_way_in(self, root):
        r"""`D:foo` 走的是进程当前目录，不是 root——is_within 照样判在外面。"""
        assert is_within(root, resolve_within(root, r"D:foo\bar.txt")) is False


# 真 PNG：签名之后紧跟 IHDR 的长度和类型，里面就有 NUL
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


class TestDetectEncoding:
    """纯函数，吃字节，不用起临时文件。"""

    def test_utf8_bom(self):
        assert detect_encoding(b"\xef\xbb\xbfhello") == "utf-8-sig"

    def test_utf16_little_endian_bom(self):
        assert detect_encoding(b"\xff\xfeh\x00i\x00") == "utf-16-le"

    def test_utf16_big_endian_bom(self):
        assert detect_encoding(b"\xfe\xff\x00h\x00i") == "utf-16-be"

    def test_utf32_bom_is_not_mistaken_for_utf16(self):
        r"""`\xff\xfe` 是 `\xff\xfe\x00\x00` 的前缀，长 BOM 必须优先匹配。"""
        assert detect_encoding(b"\xff\xfe\x00\x00rest") == "utf-32-le"

    def test_plain_text_has_no_bom(self):
        assert detect_encoding(b"print('hi')\n") is None
        assert detect_encoding("你好".encode()) is None


class TestEnsureTextFile:
    """只回答是/否：True 是文本，False 看着像二进制。"""

    def test_ascii_source_is_text(self):
        assert ensure_text_file(b"print('hi')\n") is True

    def test_utf8_chinese_is_text(self):
        assert ensure_text_file("你好，世界\n".encode()) is True

    def test_empty_file_is_text(self):
        assert ensure_text_file(b"") is True

    def test_png_is_not_text(self):
        assert ensure_text_file(PNG_HEADER + bytes(range(200, 256))) is False

    def test_gbk_chinese_is_text(self):
        """解不成 UTF-8 不等于二进制：GBK 中文是文本，只是编码不对。"""
        assert ensure_text_file("你好，世界\n".encode("gbk")) is True

    def test_utf16_with_bom_is_text(self):
        """英文字母每个后面都跟一个 NUL，光看字节和二进制没区别——靠 BOM 救回来。"""
        assert ensure_text_file("你好 world".encode("utf-16")) is True

    def test_utf16_without_bom_is_the_known_blind_spot(self):
        """已知盲区：没 BOM 的 UTF-16 从字节层面看确实和二进制分不开。"""
        assert ensure_text_file("hi".encode("utf-16-le")) is False

    def test_control_bytes_without_nul_are_the_other_blind_spot(self):
        """另一个已知盲区：没有 NUL 的字节汤会被当成文本（git 也一样）。"""
        assert ensure_text_file(bytes(range(1, 128))) is True


class TestTruncateText:
    def test_short_text_is_untouched(self):
        text = "print('hi')\n"

        assert truncate_text(text) == text

    def test_text_exactly_at_the_limit_is_untouched(self):
        """刚好到上限不算截断，别平白多送一句提示给模型。"""
        text = "x" * MAX_TEXT_CHARS

        assert truncate_text(text) == text

    def test_one_character_over_the_limit_gets_truncated(self):
        result = truncate_text("x" * (MAX_TEXT_CHARS + 1))

        assert result.startswith("x" * MAX_TEXT_CHARS)
        assert "已截断" in result

    def test_long_text_keeps_only_the_head(self):
        assert truncate_text("a" * 100, limit=10).startswith("a" * 10)

    def test_the_notice_says_how_much_was_cut(self):
        result = truncate_text("a" * 100, limit=10)

        assert "共 100 个字符" in result
        assert "前 10 个" in result

    def test_it_counts_characters_not_bytes(self):
        """按字节裁的话，中文会被砍得比英文早得多——所以数的是字符。"""
        result = truncate_text("中" * 100, limit=50)

        assert result.startswith("中" * 50)
        assert "共 100 个字符" in result


class TestShouldSkip:
    """判定要看 path.is_dir()，所以这些用例得在磁盘上真造出文件和目录。"""

    def test_git_directory_is_skipped(self, tmp_path):
        (tmp_path / ".git").mkdir()

        assert should_skip(tmp_path / ".git") is True

    def test_dependency_directory_is_skipped(self, tmp_path):
        (tmp_path / "node_modules").mkdir()

        assert should_skip(tmp_path / "node_modules") is True

    def test_env_file_is_skipped(self, tmp_path):
        (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")

        assert should_skip(tmp_path / ".env") is True

    def test_env_variant_is_caught_by_the_pattern(self, tmp_path):
        """.env.staging 不在任何名单里，靠通配兜住——这就是要有模式的原因。"""
        (tmp_path / ".env.staging").write_text("KEY=1", encoding="utf-8")

        assert should_skip(tmp_path / ".env.staging") is True

    def test_keys_and_credentials_are_skipped(self, tmp_path):
        for name in ("server.pem", "server.key", "id_rsa", "deploy.p12", "credentials.json"):
            (tmp_path / name).write_text("x", encoding="utf-8")

            assert should_skip(tmp_path / name) is True

    def test_matching_ignores_how_deep_the_file_is(self, tmp_path):
        """config/.env 和根目录的 .env 一样敏感。"""
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / ".env").write_text("KEY=1", encoding="utf-8")

        assert should_skip(tmp_path / "config" / ".env") is True

    def test_matching_ignores_case(self, tmp_path):
        """Windows 上 .ENV 和 .env 是同一个文件，所以比较前统一转小写。"""
        (tmp_path / ".ENV").write_text("KEY=1", encoding="utf-8")

        assert should_skip(tmp_path / ".ENV") is True

    def test_normal_source_file_is_kept(self, tmp_path):
        (tmp_path / "main.py").write_text("x", encoding="utf-8")

        assert should_skip(tmp_path / "main.py") is False

    def test_normal_directory_is_kept(self, tmp_path):
        (tmp_path / "src").mkdir()

        assert should_skip(tmp_path / "src") is False

    def test_a_file_named_like_a_skipped_directory_is_kept(self, tmp_path):
        """build 在目录名单里，但一个叫 build 的文件不该被连坐。"""
        (tmp_path / "build").write_text("#!/bin/sh\n", encoding="utf-8")

        assert should_skip(tmp_path / "build") is False
