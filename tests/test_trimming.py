"""trimming.py 的单元测试：折旧的、长的工具结果，块结构一个不动。全部本地执行。"""

from __future__ import annotations

from mini_code_agent.trimming import (
    KEEP_RECENT_RESULTS,
    MIN_FOLD_CHARS,
    fold_old_tool_results,
)


def a_tool_use(tool_use_id: str, name: str = "read_file") -> dict:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": name}]}


def a_tool_result(tool_use_id: str, content: str | list) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


def a_conversation(count: int, *, size: int = MIN_FOLD_CHARS + 1) -> list[dict]:
    """count 轮「模型调工具 -> 拿到结果」。"""
    messages: list[dict] = []
    for i in range(count):
        messages.append(a_tool_use(f"t{i}"))
        messages.append(a_tool_result(f"t{i}", "x" * size))
    return messages


def results_in(messages: list[dict]) -> list[dict]:
    """所有 tool_result 块，按出现顺序。"""
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]


class TestFolding:
    def test_only_the_old_ones_are_folded(self):
        messages = a_conversation(KEEP_RECENT_RESULTS + 2)

        folded = fold_old_tool_results(messages)

        assert folded.count == 2
        contents = [block["content"] for block in results_in(folded.messages)]
        assert all("折叠" in text for text in contents[:2])
        assert all(text.startswith("x") for text in contents[2:])

    def test_a_short_result_is_left_alone(self):
        """短的折它省不下地方，却丢了细节——"3 passed" 那种要留着。"""
        messages = [a_tool_use("t0"), a_tool_result("t0", "3 passed")]

        folded = fold_old_tool_results(messages, keep_recent=0)

        assert folded.count == 0
        assert results_in(folded.messages)[0]["content"] == "3 passed"

    def test_the_placeholder_says_which_tool_and_how_long(self):
        """模型得知道丢的是什么、丢了多少，不然它以为自己从没读过。"""
        messages = [a_tool_use("t0", "bash"), a_tool_result("t0", "y" * 2500)]

        folded = fold_old_tool_results(messages, keep_recent=0)

        placeholder = results_in(folded.messages)[0]["content"]
        assert "bash" in placeholder
        assert "2500" in placeholder
        assert "重读" in placeholder

    def test_the_block_structure_is_intact(self):
        """折的是 content，不是块——tool_use_id 少一个，整个请求就不合法了。"""
        messages = a_conversation(3)

        folded = fold_old_tool_results(messages, keep_recent=0)

        blocks = results_in(folded.messages)
        assert [block["tool_use_id"] for block in blocks] == ["t0", "t1", "t2"]
        assert all(block["type"] == "tool_result" for block in blocks)
        # 工具调用那侧原样
        assert folded.messages[0] == messages[0]

    def test_the_caller_list_is_not_touched(self):
        messages = a_conversation(3)
        before = [dict(message) for message in messages]

        fold_old_tool_results(messages, keep_recent=0)

        assert messages == before
        assert results_in(messages)[0]["content"].startswith("x")

    def test_a_short_history_has_nothing_to_fold(self):
        messages = a_conversation(2)

        folded = fold_old_tool_results(messages)

        assert folded.count == 0
        assert folded.chars == 0
        assert [block["content"] for block in results_in(folded.messages)] == [
            block["content"] for block in results_in(messages)
        ]

    def test_plain_string_messages_are_untouched(self):
        """用户那句话是以字符串存的，不是块——不能碰。"""
        messages = [{"role": "user", "content": "帮我看个文件"}] + a_conversation(1)

        folded = fold_old_tool_results(messages, keep_recent=0)

        assert folded.messages[0] == {"role": "user", "content": "帮我看个文件"}

    def test_a_non_string_result_is_left_alone(self):
        """这块形状上不该出现（我们的工具总是返回字符串），真出现了也别乱动。"""
        messages = [a_tool_use("t0"), a_tool_result("t0", [{"type": "text", "text": "x" * 3000}])]

        folded = fold_old_tool_results(messages, keep_recent=0)

        assert folded.count == 0

    def test_the_two_knobs_work(self):
        messages = a_conversation(3, size=10)

        folded = fold_old_tool_results(messages, keep_recent=1, min_chars=1)

        assert folded.count == 2
        assert [block["content"] for block in results_in(folded.messages)][2].startswith("x")
