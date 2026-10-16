"""folding.py 的单元测试：只在冷的时候折旧的工具结果。全部本地执行。"""

from __future__ import annotations

import os
import time
from pathlib import Path

from mini_code_agent.folding import (
    FOLD_AFTER_SECONDS,
    HEAD_CHARS,
    KEEP_RECENT_RESULTS,
    MIN_FOLD_CHARS,
    fold_stale,
    is_stale,
)
from mini_code_agent.tools import spill
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import read_tool_result

LONG = MIN_FOLD_CHARS + 200


def a_result(text: str, *, call_id: str = "t1") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}],
    }


def a_history(count: int, *, size: int = LONG) -> list[dict]:
    """count 条够长的工具结果，每条 size 个字符。"""
    return [a_result(f"第 {i} 条" + "x" * size, call_id=f"t{i}") for i in range(count)]


def contents(messages: list[dict]) -> list[str]:
    return [block["content"] for message in messages for block in message["content"]]


class TestFoldStale:
    def test_the_recent_ones_are_left_alone(self):
        """最近这几条正是模型手头在推理的，一条都不许动。"""
        messages = a_history(KEEP_RECENT_RESULTS + 3)

        folded = contents(fold_stale(messages))
        original = contents(messages)

        for index in range(len(messages) - KEEP_RECENT_RESULTS):
            assert "已清除" in folded[index]
        for index in range(len(messages) - KEEP_RECENT_RESULTS, len(messages)):
            assert folded[index] == original[index]

    def test_it_keeps_the_head(self):
        messages = a_history(KEEP_RECENT_RESULTS + 1)
        original = contents(messages)[0]

        first = contents(fold_stale(messages))[0]

        assert first.startswith(original[:HEAD_CHARS])
        assert f"{len(original) - HEAD_CHARS:,} 个字符已清除" in first

    def test_it_says_the_tool_did_not_fail(self):
        """模型看到结果突然变短，第一反应是"工具坏了、输出被截了"，然后白跑一次去确认。

        说清这是历史整理，它才知道该不该重新要。
        """
        assert "不是工具出错" in contents(fold_stale(a_history(6)))[0]

    def test_only_the_content_changes(self):
        """**消息条数、块的条数、`tool_use_id`、`is_error`，一个都不许动。**

        这一条不是洁癖：消息条数一变，`Session.record_summary` 的文件下标换算就错——它假设
        内存里的下标和文件里的消息行一一对应；`tool_use_id` 错了，整个请求都不合法。
        """
        messages = a_history(KEEP_RECENT_RESULTS + 1)
        messages[0]["content"][0]["is_error"] = True
        messages[0]["content"][0]["tool_use_id"] = "不许动"

        folded = fold_stale(messages)

        assert len(folded) == len(messages)
        assert [len(m["content"]) for m in folded] == [len(m["content"]) for m in messages]
        assert folded[0]["content"][0]["tool_use_id"] == "不许动"
        assert folded[0]["content"][0]["is_error"] is True

    def test_the_input_is_not_touched(self):
        """和 `runagent` 一个契约：返回新的一份，不改传进来那份。

        消息 dict 和它里面的块列表在别处是**共享引用**（`runagent` 只复制最外层那个列表），
        就地改会连调用方手里那份一起改掉。
        """
        messages = a_history(6)
        before = contents(messages)

        folded = fold_stale(messages)

        assert folded is not messages
        assert contents(messages) == before

    def test_other_messages_are_left_alone(self):
        """只认 `tool_result` 块：真用户轮（字符串 content）、文本块、tool_use 都不碰。"""
        plain = {"role": "user", "content": "x" * LONG}
        messages = [plain, *a_history(KEEP_RECENT_RESULTS + 1)]

        folded = fold_stale(messages)

        assert folded[0] is plain

    def test_short_results_are_left_alone(self):
        """折完那条说明自己就有两百多字符，原文比它还短的话折了反而变长。"""
        messages = a_history(KEEP_RECENT_RESULTS + 3, size=10)

        assert fold_stale(messages) is messages

    def test_a_mixed_history_folds_only_what_is_worth_it(self):
        messages = [
            a_result("x" * LONG, call_id="long"),
            a_result("短得很", call_id="short"),
            *a_history(KEEP_RECENT_RESULTS, size=10),
        ]

        folded = fold_stale(messages)

        assert "已清除" in folded[0]["content"][0]["content"]
        assert folded[1]["content"][0]["content"] == "短得很"

    def test_the_same_object_comes_back_when_nothing_changed(self):
        """调用方靠 `is` 判断"这一轮折没折"——和 `_compact` 用的是同一个信号。

        条数刚好等于"要留的条数"时，一条都不该折。
        """
        messages = a_history(KEEP_RECENT_RESULTS)

        assert fold_stale(messages) is messages

    def test_folding_twice_is_folding_once(self):
        """折过的结果正好落在 `MIN_FOLD_CHARS` 那道闸以下，所以不会套娃。"""
        once = fold_stale(a_history(8))

        assert fold_stale(once) is once

    def test_a_spilled_result_is_still_reachable(self, tmp_path: Path):
        """**这是"保留开头一段"而不是"一句固定占位符"的全部理由。**

        落过盘的结果，存哪儿和 `read_tool_result` 要的 id 就写在开头那几行——留着这一段，
        折完之后它照样取得回来。
        """
        directory = tmp_path / spill.TOOL_RESULTS_DIR
        directory.mkdir(parents=True)
        (directory / "bash-abc123.txt").write_text("整篇输出\n" * 300, encoding="utf-8")
        preview = spill.preview_message(
            "整篇输出\n" * 300,
            name="bash-abc123",
            where=".agent/tool-results/bash-abc123.txt",
        )
        messages = [a_result(preview, call_id="t0"), *a_history(KEEP_RECENT_RESULTS + 1)]

        first = contents(fold_stale(messages))[0]

        assert "已清除" in first  # 真折了
        assert "bash-abc123" in first  # id 还在留着的那段里
        got = read_tool_result({"id": "bash-abc123"}, ToolContext(cwd=tmp_path))
        assert "整篇输出" in got  # 照着它取，照样取得到


class TestFoldIsIdempotentByConstruction:
    def test_what_folding_produces_falls_below_the_threshold(self):
        """折完那条必须落在这道闸下面，否则"折过的又折一遍"，上面那条幂等就不成立了。

        这条断言的是两个常量之间的关系（`MIN_FOLD_CHARS > HEAD_CHARS + 说明的长度`），
        所以改动那两个数时它会跟着响。
        """
        folded = contents(fold_stale(a_history(6)))[0]

        assert len(folded) <= MIN_FOLD_CHARS


class TestIsStale:
    def test_a_fresh_file_is_not_stale(self, tmp_path: Path):
        """刚写过 = 上一轮刚结束 = 缓存还热着。这时候折是亏的，所以不许折。"""
        path = tmp_path / "s.jsonl"
        path.write_text("{}", encoding="utf-8")

        assert is_stale(path) is False

    def test_an_old_file_is_stale(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        path.write_text("{}", encoding="utf-8")
        old = time.time() - FOLD_AFTER_SECONDS - 60
        os.utime(path, (old, old))

        assert is_stale(path) is True

    def test_a_missing_file_is_not_stale(self, tmp_path: Path):
        """还没说过话的会话盘上没有文件——没什么可折的，也不该因此炸。"""
        assert is_stale(tmp_path / "nope.jsonl") is False

    def test_the_threshold_is_the_boundary(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        path.write_text("{}", encoding="utf-8")
        now = time.time()
        os.utime(path, (now - 100, now - 100))

        assert is_stale(path, now=now, after=99) is True
        assert is_stale(path, now=now, after=101) is False
