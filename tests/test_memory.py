"""memory/store.py 的单元测试：写、读、校验、索引。全部本地执行。"""

from __future__ import annotations

import pytest

from mini_code_agent.memory import index_text, names, read, sync_memory_index, write
from mini_code_agent.memory.store import INDEX_NAME, Memory, memory_dir, render
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.memory import read_memory, remember


def a_memory(name: str = "tabs-for-indent", **overrides) -> Memory:
    fields = {
        "description": "缩进用 tab，不用空格",
        "type": "user",
        "body": "改任何文件都按这个来。",
        **overrides,
    }
    return Memory(name=name, **fields)


class TestWriteAndRead:
    def test_a_memory_survives_a_round_trip(self, tmp_path):
        write(tmp_path, a_memory())

        memory = read(tmp_path, "tabs-for-indent")

        assert memory.description == "缩进用 tab，不用空格"
        assert memory.type == "user"
        assert memory.body == "改任何文件都按这个来。"
        assert memory.source == "user"  # 没写就默认是人写的
        assert memory.created  # 没写就默认今天

    def test_writing_the_same_name_overwrites(self, tmp_path):
        write(tmp_path, a_memory(description="旧的说法"))
        write(tmp_path, a_memory(description="新的说法"))

        assert read(tmp_path, "tabs-for-indent").description == "新的说法"
        assert len(names(tmp_path)) == 1

    def test_the_file_is_hand_readable(self, tmp_path):
        """这是给人看、也给人改的文件——落盘的样子得能读。"""
        path = write(tmp_path, a_memory())

        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "description: 缩进用 tab，不用空格" in text
        assert text.endswith("改任何文件都按这个来。\n")

    def test_a_description_with_a_colon_is_fine(self, tmp_path):
        """冒号只切第一个，所以描述里带冒号不会被切坏。"""
        write(tmp_path, a_memory(description="跑测试: 一律 uv run --no-sync"))

        assert read(tmp_path, "tabs-for-indent").description == "跑测试: 一律 uv run --no-sync"


class TestNameChecking:
    @pytest.mark.parametrize("bad", ["../evil", "a/b", "Tabs", "", "有中文", "memory", ".hidden"])
    def test_a_bad_name_is_refused(self, tmp_path, bad):
        with pytest.raises(ValueError, match="不是有效的记忆名"):
            write(tmp_path, a_memory(name=bad))

    def test_reading_a_bad_name_is_refused_too(self, tmp_path):
        """读这一侧也要挡——不然 ../ 能顺着跑出目录去。"""
        with pytest.raises(ValueError, match="不是有效的记忆名"):
            read(tmp_path, "../../etc/passwd")

    def test_a_missing_memory_lists_what_there_is(self, tmp_path):
        """报错里带上现有的名字，模型才知道该拿什么去取。"""
        write(tmp_path, a_memory())

        with pytest.raises(FileNotFoundError, match="tabs-for-indent"):
            read(tmp_path, "nope")


class TestValidation:
    def test_an_unknown_type_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="type"):
            write(tmp_path, a_memory(type="whatever"))

    def test_an_unknown_source_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="source"):
            write(tmp_path, a_memory(source="robot"))

    def test_an_empty_body_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="正文是空的"):
            write(tmp_path, a_memory(body="   "))

    def test_a_blank_description_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="description"):
            write(tmp_path, a_memory(description="  "))


class TestTheIndex:
    def test_no_memories_means_no_index_text(self, tmp_path):
        assert index_text(tmp_path) == ""

    def test_it_lists_the_name_and_the_description(self, tmp_path):
        write(tmp_path, a_memory())

        assert index_text(tmp_path) == (
            "- [tabs-for-indent](tabs-for-indent.md) — 缩进用 tab，不用空格"
        )

    def test_it_is_sorted(self, tmp_path):
        """顺序要稳——索引进 system prompt，排来排去会把缓存前缀弄失效。"""
        write(tmp_path, a_memory(name="zebra"))
        write(tmp_path, a_memory(name="alpha"))

        assert index_text(tmp_path).splitlines() == [
            "- [alpha](alpha.md) — 缩进用 tab，不用空格",
            "- [zebra](zebra.md) — 缩进用 tab，不用空格",
        ]

    def test_the_index_file_is_written_but_is_not_a_memory(self, tmp_path):
        write(tmp_path, a_memory())

        assert names(tmp_path) == ["tabs-for-indent"]
        assert (tmp_path / ".agent" / "memory" / INDEX_NAME).is_file()

    def test_a_hand_broken_file_is_skipped(self, tmp_path):
        """手工改坏一个，不该让整份索引读不出来。"""
        write(tmp_path, a_memory())
        broken = tmp_path / ".agent" / "memory" / "broken.md"
        broken.write_text("没有 frontmatter", encoding="utf-8")

        text = index_text(tmp_path)

        assert "tabs-for-indent" in text
        assert "broken" not in text


class TestSyncIndex:
    """磁盘上那份索引会漂——手工加、手工删都不经过 write()。启动时对一次。"""

    def hand_write(self, tmp_path, name: str = "hand-made") -> None:
        directory = memory_dir(tmp_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.md").write_text(render(a_memory(name=name)), encoding="utf-8")

    def disk_index(self, tmp_path) -> str:
        return (memory_dir(tmp_path) / INDEX_NAME).read_text(encoding="utf-8")

    def test_a_hand_added_memory_lands_in_the_index(self, tmp_path):
        self.hand_write(tmp_path)

        assert sync_memory_index(tmp_path) is True

        assert "hand-made" in self.disk_index(tmp_path)

    def test_a_hand_deleted_memory_leaves_the_index(self, tmp_path):
        write(tmp_path, a_memory(name="aaa"))
        write(tmp_path, a_memory(name="bbb"))
        (memory_dir(tmp_path) / "aaa.md").unlink()

        assert sync_memory_index(tmp_path) is True

        assert "aaa" not in self.disk_index(tmp_path)
        assert "bbb" in self.disk_index(tmp_path)

    def test_it_does_nothing_when_already_in_sync(self, tmp_path):
        """稳态下一个字节都不动——不然每次启动都重写一遍。"""
        write(tmp_path, a_memory())

        assert sync_memory_index(tmp_path) is False

    def test_no_directory_means_nothing_happens(self, tmp_path):
        """不因为跑一次 agent，就在别人项目里造一个 .agent/memory/。"""
        assert sync_memory_index(tmp_path) is False
        assert not (tmp_path / ".agent").exists()


class TestTheTool:
    def test_it_returns_the_body(self, tmp_path):
        write(tmp_path, a_memory())

        body = read_memory({"name": "tabs-for-indent"}, ToolContext(cwd=tmp_path))

        assert body == "改任何文件都按这个来。"

    def test_an_unknown_name_raises(self, tmp_path):
        """交给 dispatch 变成模型看得见的结果，它才知道自己拼错了名字。"""
        with pytest.raises(FileNotFoundError):
            read_memory({"name": "nope"}, ToolContext(cwd=tmp_path))


class TestTheRememberTool:
    def a_call(self, tmp_path, **overrides) -> str:
        fields = {
            "name": "uv-for-tests",
            "description": "跑测试一律 uv run --no-sync",
            "type": "project",
            "body": "同步失败时会返回 0，所以验证一律加 --no-sync。",
            **overrides,
        }
        return remember(fields, ToolContext(cwd=tmp_path))

    def test_it_writes_the_memory(self, tmp_path):
        message = self.a_call(tmp_path)

        memory = read(tmp_path, "uv-for-tests")
        assert memory.type == "project"
        assert memory.body == "同步失败时会返回 0，所以验证一律加 --no-sync。"
        assert "uv-for-tests" in message

    def test_the_source_is_set_here_not_by_the_model(self, tmp_path):
        """记忆是谁写的，不该让写它的模型自己填——那是给人看的置信度标记。"""
        self.a_call(tmp_path, source="user")

        assert read(tmp_path, "uv-for-tests").source == "agent"

    def test_the_index_is_rebuilt(self, tmp_path):
        self.a_call(tmp_path)

        assert "uv-for-tests" in index_text(tmp_path)

    def test_writing_the_same_name_twice_overwrites(self, tmp_path):
        self.a_call(tmp_path, body="旧的说法")
        self.a_call(tmp_path, body="新的说法")

        assert read(tmp_path, "uv-for-tests").body == "新的说法"
        assert len(names(tmp_path)) == 1

    def test_a_bad_type_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="type"):
            self.a_call(tmp_path, type="nope")

    def test_a_bad_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="不是有效的记忆名"):
            self.a_call(tmp_path, name="../evil")

    def test_an_empty_body_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="正文是空的"):
            self.a_call(tmp_path, body="  ")
