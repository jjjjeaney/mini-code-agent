"""summarizing.py 的单元测试：什么时候压、切在哪儿。全部本地执行。"""

from __future__ import annotations

from mini_code_agent.providers.base import ModelResponse
from mini_code_agent.summarizing import (
    KEEP_RECENT_TURNS,
    SUMMARY_PROMPT,
    cut_point,
    summarize,
)


def a_history(turns: int, *, size: int = 0) -> list[dict]:
    """turns 轮「用户说话 -> 模型调工具 -> 结果」。size 是工具结果有多大。"""
    messages: list[dict] = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"第 {i} 句"})
        if size:
            call = {"type": "tool_use", "id": f"t{i}", "name": "read_file"}
            result = {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * size}
            messages.append({"role": "assistant", "content": [call]})
            messages.append({"role": "user", "content": [result]})
    return messages


def orphans(messages: list[dict]) -> list[str]:
    """找不到对应 tool_use 的 tool_result——有的话，这个请求不合法。"""
    seen: set[str] = set()
    lost: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                seen.add(str(block.get("id")))
            elif block.get("type") == "tool_result" and str(block.get("tool_use_id")) not in seen:
                lost.append(str(block.get("tool_use_id")))
    return lost


class ScriptedProvider:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.messages: list[list[dict]] = []
        self.tools: list[object] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.messages.append([dict(m) for m in messages])
        self.tools.append(tools)
        return ModelResponse(
            text=self.text,
            assistant_content=[{"type": "text", "text": self.text or ""}],
            stop_reason="end_turn",
        )


class TestCutPoint:
    def test_a_short_history_is_left_alone(self):
        assert cut_point(a_history(3, size=1_000)) is None

    def test_a_long_history_gets_a_cut(self):
        messages = a_history(KEEP_RECENT_TURNS + 5, size=30_000)

        cut = cut_point(messages)

        assert cut is not None
        # 切点是真正的用户轮——字符串 content，不是那批工具结果
        assert isinstance(messages[cut]["content"], str)
        # 留下来的正好是最近 KEEP_RECENT_TURNS 轮
        kept_turns = [m for m in messages[cut:] if isinstance(m["content"], str)]
        assert len(kept_turns) == KEEP_RECENT_TURNS

    def test_it_never_cuts_inside_a_tool_exchange(self):
        """切在 tool_use 和它的 tool_result 中间，后半段就从孤儿结果开头——请求不合法。"""
        messages = a_history(KEEP_RECENT_TURNS + 5, size=30_000)

        cut = cut_point(messages)

        assert cut is not None
        assert orphans(messages[cut:]) == []

    def test_not_enough_turns_is_left_alone(self):
        """轮数不够，就没有安全的切点——宁可先不压。"""
        assert cut_point(a_history(2, size=200_000)) is None


class TestSummarize:
    def test_it_asks_without_tools(self):
        """不带 tools：模型手里没工具可调，就只能出文本。"""
        provider = ScriptedProvider("前面聊了点东西")
        history = a_history(2)

        text = summarize(history, provider)

        assert text == "前面聊了点东西"
        assert provider.tools == [None]
        # 历史原样发过去，后面跟上那句摘要指令
        assert provider.messages[0][: len(history)] == history
        assert provider.messages[0][-1]["content"] == SUMMARY_PROMPT

    def test_a_textless_answer_is_an_empty_string(self):
        assert summarize(a_history(1), ScriptedProvider(None)) == ""
