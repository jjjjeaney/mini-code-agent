"""summarizing.py 的单元测试：这段历史算多大、什么时候压、切在哪儿。全部本地执行。"""

from __future__ import annotations

from mini_code_agent.providers.base import ModelResponse
from mini_code_agent.summarizing import (
    COMPACT_AT,
    KEEP_RECENT_TURNS,
    SUMMARY_PROMPT,
    context_size,
    cut_point,
    rough_tokens,
    summarize,
)

# 判断用的两个数：窗口 100 万，够到 80% 就该压。
WINDOW = 1_000_000
OVER = int(WINDOW * COMPACT_AT) + 1  # 过线
UNDER = int(WINDOW * COMPACT_AT) - 1  # 没过线


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
    def test_below_the_threshold_is_left_alone(self):
        assert cut_point(a_history(3), size=UNDER, window=WINDOW) is None

    def test_over_the_threshold_gets_a_cut(self):
        messages = a_history(KEEP_RECENT_TURNS + 5)

        cut = cut_point(messages, size=OVER, window=WINDOW)

        assert cut is not None
        # 切点是真正的用户轮——字符串 content，不是那批工具结果
        assert isinstance(messages[cut]["content"], str)
        # 留下来的正好是最近 KEEP_RECENT_TURNS 轮
        kept_turns = [m for m in messages[cut:] if isinstance(m["content"], str)]
        assert len(kept_turns) == KEEP_RECENT_TURNS

    def test_it_never_cuts_inside_a_tool_exchange(self):
        """切在 tool_use 和它的 tool_result 中间，后半段就从孤儿结果开头——请求不合法。"""
        messages = a_history(KEEP_RECENT_TURNS + 5, size=30_000)

        cut = cut_point(messages, size=OVER, window=WINDOW)

        assert cut is not None
        assert orphans(messages[cut:]) == []

    def test_not_enough_turns_is_left_alone(self):
        """轮数不够，就没有安全的切点——宁可先不压。

        **`size` 得给到过线**：给个小的话它返回 None 是因为"没到阈值"，而不是这条要测的
        "轮数不够"——那就成了一条永远绿的假用例（它绿的原因和它说的不是一回事）。
        """
        assert cut_point(a_history(2), size=OVER, window=WINDOW) is None

    def test_force_skips_the_size_check(self):
        """`/compact`：用户点名要压，没到阈值也压。"""
        messages = a_history(KEEP_RECENT_TURNS + 5, size=30_000)

        assert cut_point(messages, size=UNDER, window=WINDOW, force=True) is not None

    def test_an_unknown_window_never_triggers(self):
        """窗口不知道（0）就不自动压——阈值这一侧宁可偏保守。"""
        assert cut_point(a_history(KEEP_RECENT_TURNS + 5), size=10**9, window=0) is None


class TestContextSize:
    """这段历史现在有多大——压缩阈值判的就是它。"""

    def test_without_an_anchor_it_estimates_the_whole_thing(self):
        messages = a_history(3)

        assert context_size(messages, tokens=None) == rough_tokens(messages)

    def test_with_an_anchor_it_only_estimates_what_came_after(self):
        """锚点是上一发真请求报的，它自带工具表和系统提示词——那两块估算够不着。
        所以有锚点时只补"锚点之后新加的那几条"。"""
        messages = a_history(3)
        covered = 2

        assert context_size(messages, tokens=50_000, covered=covered) == (
            50_000 + rough_tokens(messages[covered:])
        )

    def test_an_anchor_covering_everything_adds_nothing(self):
        messages = a_history(3)

        assert context_size(messages, tokens=50_000, covered=len(messages)) == 50_000

    def test_an_empty_tail_adds_nothing(self):
        assert context_size([], tokens=50_000, covered=0) == 50_000

    def test_the_estimate_leans_high(self):
        """按 2 字符 1 token 估——**宁可估大**（早压），不估小（晚压）。"""
        messages = [{"role": "user", "content": "x" * 1_000}]

        # 连块的外壳一起算（那部分也是要发的），所以只会多
        assert 500 <= rough_tokens(messages) <= 600


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
