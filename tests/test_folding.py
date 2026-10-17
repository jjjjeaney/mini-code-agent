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
    fold_if_stale,
    fold_stale,
    is_stale,
)
from mini_code_agent.tools import spill
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.files import read_tool_result

LONG = MIN_FOLD_CHARS + 200

# 测试里"允许被折"的那几种。真那份在 tools/registry.py 里，由 Tool.foldable 标出来；
# 那几条名单本身由 test_tools.py 守。
FOLDABLE = frozenset({"read_file", "grep", "bash"})


def a_pair(text: str, *, call_id: str = "t1", tool: str = "read_file") -> list[dict]:
    """**一对**消息：模型要求调工具 + 那条结果。

    名字得配对——`tool_result` 块自己不带工具名，折叠要靠前面那条 assistant 里的 `tool_use`
    才认得出这是哪个工具的结果。
    """
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call_id, "name": tool, "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}],
        },
    ]


def a_history(count: int, *, size: int = LONG, tool: str = "read_file") -> list[dict]:
    """count 条够长的工具结果，每条连着它那条 `tool_use`。"""
    messages: list[dict] = []
    for index in range(count):
        messages += a_pair(f"第 {index} 条" + "x" * size, call_id=f"t{index}", tool=tool)
    return messages


def contents(messages: list[dict]) -> list[str]:
    """所有 tool_result 的 content，按出现顺序。"""
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append(str(block.get("content")))
    return found


def fold(messages: list[dict], *, keep: int = KEEP_RECENT_RESULTS) -> list[dict]:
    return fold_stale(messages, foldable=FOLDABLE, keep=keep)


class TestWhichTools:
    """**只有标了 `foldable` 的工具的结果才进候选。**"""

    def test_a_tool_that_is_not_foldable_is_left_alone(self):
        """子 agent 的结论、项目概览、模型特意去取的记忆——那些折了就真丢了。

        这一条是整件事的重点：子 agent 交回来的方案就是"方向"本身，而且它背后是一整趟子
        agent（重跑一遍是几分钟和几毛钱）。
        """
        messages = a_history(8, tool="project_tree")

        assert fold(messages) is messages

    def test_the_keep_window_only_counts_foldable_ones(self):
        """可折的留最近几张"名额"，不可折的那些**既不占名额、也不会被动**。

        和 Claude Code 的 `keepRecent` 一个数法（它数的也是"最近 N 个**可压缩工具**的结果"）。
        """
        messages = [
            *a_pair("x" * LONG, call_id="f0", tool="read_file"),
            *a_pair("y" * LONG, call_id="p0", tool="project_tree"),
            *a_pair("x" * LONG, call_id="f1", tool="read_file"),
            *a_pair("y" * LONG, call_id="p1", tool="project_tree"),
            *a_pair("x" * LONG, call_id="f2", tool="read_file"),
        ]

        folded = contents(fold(messages, keep=2))

        assert "已清除" in folded[0]  # 最早那条 read_file 折了
        assert "已清除" not in folded[1]  # project_tree 一条都没动
        assert "已清除" not in folded[3]
        assert "已清除" not in folded[2]  # 最近两条 read_file 留着
        assert "已清除" not in folded[4]

    def test_a_result_with_no_matching_tool_use_is_left_alone(self):
        """认不出名字的当"不可折"处理——宁可留着。

        理论上不该有（配对错了请求本身就不合法），但真出现时不许猜。
        """
        orphan = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "没人认领", "content": "x" * LONG}],
        }
        messages = [orphan, *a_history(8)]

        folded = contents(fold(messages, keep=1))

        assert folded[0] == "x" * LONG


class TestFoldStale:
    def test_the_recent_ones_are_left_alone(self):
        """最近这几条正是模型手头在推理的，一条都不许动。"""
        messages = a_history(KEEP_RECENT_RESULTS + 3)

        folded = contents(fold(messages))
        original = contents(messages)

        for index in range(len(original) - KEEP_RECENT_RESULTS):
            assert "已清除" in folded[index]
        for index in range(len(original) - KEEP_RECENT_RESULTS, len(original)):
            assert folded[index] == original[index]

    def test_it_keeps_the_head(self):
        messages = a_history(KEEP_RECENT_RESULTS + 1)
        original = contents(messages)[0]

        first = contents(fold(messages))[0]

        assert first.startswith(original[:HEAD_CHARS])
        assert f"{len(original) - HEAD_CHARS:,} 个字符已清除" in first

    def test_it_names_the_tool_it_came_from(self):
        """`tool_result` 块自己不带工具名——不写进去，模型看到的是一段没头没尾的说明，
        判不出被清掉的是什么、也就判不出该不该重新要。
        """
        messages = a_history(KEEP_RECENT_RESULTS + 1, tool="grep")

        assert "grep 的结果" in contents(fold(messages))[0]

    def test_it_says_how_to_get_it_back(self):
        """只说"清掉了"不够——得说清怎么拿回来，不然模型要么放弃、要么白跑一趟。"""
        first = contents(fold(a_history(6)))[0]

        assert "不是工具出错" in first
        assert "read_tool_result" in first
        assert "重跑" in first

    def test_only_the_content_changes(self):
        """**消息条数、块的条数、`tool_use_id`、`is_error`，一个都不许动。**

        这一条不是洁癖：消息条数一变，`Session.record_summary` 的文件下标换算就错——它假设
        内存里的下标和文件里的消息行一一对应；`tool_use_id` 错了，整个请求都不合法。
        """
        messages = a_history(KEEP_RECENT_RESULTS + 1)
        messages[1]["content"][0]["is_error"] = True
        messages[1]["content"][0]["tool_use_id"] = "不许动"

        folded = fold(messages)

        assert len(folded) == len(messages)
        assert [len(m["content"]) for m in folded] == [len(m["content"]) for m in messages]
        assert folded[1]["content"][0]["tool_use_id"] == "不许动"
        assert folded[1]["content"][0]["is_error"] is True

    def test_the_input_is_not_touched(self):
        """和 `runagent` 一个契约：返回新的一份，不改传进来那份。

        消息 dict 和它里面的块列表在别处是**共享引用**（`runagent` 只复制最外层那个列表），
        就地改会连调用方手里那份一起改掉。
        """
        messages = a_history(6)
        before = contents(messages)

        folded = fold(messages)

        assert folded is not messages
        assert contents(messages) == before

    def test_other_messages_are_left_alone(self):
        """只认 `tool_result` 块：真用户轮（字符串 content）、文本块、tool_use 都不碰。"""
        plain = {"role": "user", "content": "x" * LONG}
        messages = [plain, *a_history(KEEP_RECENT_RESULTS + 1)]

        folded = fold(messages)

        assert folded[0] is plain

    def test_short_results_are_left_alone(self):
        """折完那条说明自己就有两百多字符，原文比它还短的话折了反而变长。"""
        messages = a_history(KEEP_RECENT_RESULTS + 3, size=10)

        assert fold(messages) is messages

    def test_a_mixed_history_folds_only_what_is_worth_it(self):
        messages = [
            *a_pair("x" * LONG, call_id="long"),
            *a_pair("短得很", call_id="short"),
            *a_history(KEEP_RECENT_RESULTS, size=10),
        ]

        folded = contents(fold(messages))

        assert "已清除" in folded[0]
        assert folded[1] == "短得很"

    def test_the_same_object_comes_back_when_nothing_changed(self):
        """调用方靠 `is` 判断"这一轮折没折"——和 `_compact` 用的是同一个信号。"""
        messages = a_history(KEEP_RECENT_RESULTS)

        assert fold(messages) is messages

    def test_folding_twice_is_folding_once(self):
        """折过的结果正好落在 `MIN_FOLD_CHARS` 那道闸以下，所以不会套娃。"""
        once = fold(a_history(8))

        assert fold(once) is once

    def test_a_spilled_result_is_still_reachable(self, tmp_path: Path):
        """**这是"保留开头一段"而不是"一句固定占位符"的全部理由。**

        落过盘的结果，存哪儿和 `read_tool_result` 要的 id 就写在开头那几行——留着这一段，
        折完之后它照样取得回来。
        """
        directory = tmp_path / spill.TOOL_RESULTS_DIR
        directory.mkdir(parents=True)
        (directory / "toolu_abc123.txt").write_text("整篇输出\n" * 300, encoding="utf-8")
        preview = spill.preview_message(
            "整篇输出\n" * 300,
            name="toolu_abc123",
            where=".agent/tool-results/toolu_abc123.txt",
        )
        messages = [
            *a_pair(preview, call_id="t0", tool="bash"),
            *a_history(KEEP_RECENT_RESULTS + 1),
        ]

        first = contents(fold(messages))[0]

        assert "已清除" in first  # 真折了
        assert "toolu_abc123" in first  # id 还在留着的那段里
        got = read_tool_result({"id": "toolu_abc123"}, ToolContext(cwd=tmp_path))
        assert "整篇输出" in got  # 照着它取，照样取得到


class TestFoldIsIdempotentByConstruction:
    def test_what_folding_produces_falls_below_the_threshold(self):
        """折完那条必须落在这道闸下面，否则"折过的又折一遍"，上面那条幂等就不成立了。

        这条断言的是两个常量之间的关系（`MIN_FOLD_CHARS > HEAD_CHARS + 说明的长度`），
        所以改动那两个数、或者把工具名加长时，它会跟着响。
        """
        folded = contents(fold(a_history(6)))[0]

        assert len(folded) <= MIN_FOLD_CHARS


class TestFoldIfStale:
    """把"冷不冷"和"怎么折"合成一次调用——调用方那边因此只剩一行。

    **判据不该由调用方自己拼**：两个常量（阈值、折叠规则）分在两处，改一个忘一个就是悄悄
    走错，而这条路上走错的表现是"在缓存还热的时候折"，也就是钱。
    """

    def test_a_warm_session_is_left_alone(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        path.write_text("{}", encoding="utf-8")
        messages = a_history(8)

        assert fold_if_stale(messages, foldable=FOLDABLE, path=path) is messages

    def test_a_stale_session_is_folded(self, tmp_path: Path):
        path = tmp_path / "s.jsonl"
        path.write_text("{}", encoding="utf-8")
        old = time.time() - FOLD_AFTER_SECONDS - 60
        os.utime(path, (old, old))
        messages = a_history(8)

        folded = fold_if_stale(messages, foldable=FOLDABLE, path=path)

        assert folded is not messages
        assert "已清除" in contents(folded)[0]

    def test_a_session_that_never_landed_is_left_alone(self, tmp_path: Path):
        """还没说过话的会话盘上没有文件，`is_stale` 那边就返回 False——没什么可折的。"""
        messages = a_history(8)

        assert fold_if_stale(messages, foldable=FOLDABLE, path=tmp_path / "nope.jsonl") is messages


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
