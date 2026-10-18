"""工具批内并行：分段规则、真的重叠、顺序、异常隔离、并发上限。

线程只用于验证重叠；其余全是同步的剧本化 provider。
"""

from __future__ import annotations

import threading
import time

from mini_code_agent.core.agent import (
    PARALLEL_LIMIT,
    _is_parallel,
    _segments,
    runagent,
)
from mini_code_agent.permissions import PermissionRequest, decide_permission
from mini_code_agent.providers.base import ModelResponse, ToolCall
from mini_code_agent.tools import Tool, ToolRegistry, default_tools


class ScriptedProvider:
    """按剧本依次回答，记下每次请求收到的消息。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def batch(*calls: ToolCall) -> ModelResponse:
    """一条 assistant 消息里并排放多个 tool_use。"""
    return ModelResponse(
        text=None,
        tool_calls=list(calls),
        assistant_content=[
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in calls
        ],
        stop_reason="tool_use",
    )


def text_turn(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        assistant_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
    )


def a_call(call_id: str, name: str = "probe") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


def a_tool(name: str, fn, *, read_only: bool) -> Tool:
    return Tool(name=name, fn=fn, read_only=read_only, params={})


def a_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            a_tool("read_a", lambda kwargs, ctx: "", read_only=True),
            a_tool("read_b", lambda kwargs, ctx: "", read_only=True),
            a_tool("write", lambda kwargs, ctx: "", read_only=False),
        ]
    )


class TestSegmentation:
    def test_consecutive_reads_form_one_segment(self):
        registry = a_registry()
        calls = [
            a_call("1", "read_a"),
            a_call("2", "read_b"),
            a_call("3", "write"),
            a_call("4", "read_a"),
        ]

        assert _segments(calls, registry) == [
            [calls[0], calls[1]],
            [calls[2]],
            [calls[3]],
        ]

    def test_a_write_splits_the_reads(self):
        """读跨过写去并行会改变"后面的读看到什么"——必须切成三段。"""
        registry = a_registry()
        calls = [a_call("1", "read_a"), a_call("2", "write"), a_call("3", "read_b")]

        assert _segments(calls, registry) == [[calls[0]], [calls[1]], [calls[2]]]

    def test_all_reads_are_one_segment(self):
        registry = a_registry()
        calls = [a_call(str(index), "read_a") for index in range(5)]

        assert _segments(calls, registry) == [calls]

    def test_every_write_is_its_own_segment(self):
        registry = a_registry()
        calls = [a_call("1", "write"), a_call("2", "write")]

        assert _segments(calls, registry) == [[calls[0]], [calls[1]]]

    def test_an_unknown_tool_is_serial(self):
        """查不到的工具交给 dispatch 回错误；不能把它和只读的混在一段里跑。"""
        registry = a_registry()
        calls = [a_call("1", "read_a"), a_call("2", "nope")]

        assert _segments(calls, registry) == [[calls[0]], [calls[1]]]

    def test_web_tools_are_parallel_too(self):
        registry = default_tools()

        assert _is_parallel(a_call("1", "web_search"), registry) is True
        assert _is_parallel(a_call("2", "web_fetch"), registry) is True
        assert _is_parallel(a_call("3", "bash"), registry) is False


class TestExecution:
    def test_adjacent_reads_really_overlap(self):
        """串行时第一条会卡在 Barrier 上超时；两条都通过就证明它们在并行。"""
        barrier = threading.Barrier(2, timeout=2)
        outcomes: list[bool] = []

        def paired(kwargs, ctx):
            try:
                barrier.wait()
                outcomes.append(True)
            except threading.BrokenBarrierError:
                outcomes.append(False)
            return "ok"

        provider = ScriptedProvider(
            [batch(a_call("a", "paired"), a_call("b", "paired")), text_turn("好了")]
        )

        runagent("读两下", provider, ToolRegistry([a_tool("paired", paired, read_only=True)]))

        assert outcomes == [True, True]

    def test_a_write_between_reads_keeps_them_apart(self):
        barrier = threading.Barrier(2, timeout=0.3)
        outcomes: list[object] = []

        def read(kwargs, ctx):
            try:
                barrier.wait()
                outcomes.append(True)
            except threading.BrokenBarrierError:
                outcomes.append(False)
            return "ok"

        def write(kwargs, ctx):
            outcomes.append("write")
            return "ok"

        registry = ToolRegistry(
            [
                a_tool("read_a", read, read_only=True),
                a_tool("write", write, read_only=False),
                a_tool("read_c", read, read_only=True),
            ]
        )
        provider = ScriptedProvider(
            [
                batch(
                    a_call("a", "read_a"),
                    a_call("w", "write"),
                    a_call("c", "read_c"),
                ),
                text_turn("好了"),
            ]
        )

        runagent("混着来", provider, registry)

        # 两个读各自单独成段：Barrier 永远等不到第二个，所以都超时；中间那个写按顺序发生。
        assert outcomes == [False, "write", False]

    def test_results_and_trace_keep_the_original_order(self):
        def slow(kwargs, ctx):
            time.sleep(0.08)
            return "慢"

        def fast(kwargs, ctx):
            return "快"

        registry = ToolRegistry(
            [
                a_tool("slow", slow, read_only=True),
                a_tool("fast", fast, read_only=True),
            ]
        )
        provider = ScriptedProvider(
            [batch(a_call("a", "slow"), a_call("b", "fast")), text_turn("好了")]
        )

        result = runagent("跑", provider, registry)

        blocks = provider.calls[1][-1]["content"]
        assert [block["tool_use_id"] for block in blocks] == ["a", "b"]
        assert result.trace == [
            "→ slow()",
            "→ fast()",
            "← slow: 慢",
            "← fast: 快",
            "好了",
        ]

    def test_one_failure_does_not_stop_the_others(self):
        def boom(kwargs, ctx):
            raise RuntimeError("坏了")

        def fine(kwargs, ctx):
            return "好的"

        registry = ToolRegistry(
            [
                a_tool("boom", boom, read_only=True),
                a_tool("fine", fine, read_only=True),
            ]
        )
        provider = ScriptedProvider(
            [batch(a_call("a", "boom"), a_call("b", "fine")), text_turn("好了")]
        )

        runagent("跑", provider, registry)

        blocks = provider.calls[1][-1]["content"]
        assert blocks[0]["is_error"] is True
        assert "坏了" in blocks[0]["content"]
        assert blocks[1]["content"] == "好的"

    def test_a_parallel_segment_respects_the_limit(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def probe(kwargs, ctx):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return "ok"

        registry = ToolRegistry([a_tool("probe", probe, read_only=True)])
        calls = [a_call(str(index), "probe") for index in range(8)]
        provider = ScriptedProvider([batch(*calls), text_turn("好了")])

        runagent("跑一批", provider, registry)

        assert peak <= PARALLEL_LIMIT
        assert peak >= 2  # 真的并行过，不是碰巧串行


class TestTheParallelInvariant:
    def test_every_read_only_tool_is_allowed_without_asking(self):
        """并行段在 worker 线程里跑 `_gate`；这条保证那里永远不会弹权限问题。"""
        registry = default_tools()

        for tool in registry.list():
            if not tool.read_only:
                continue
            decision = decide_permission(
                PermissionRequest(tool_name=tool.name, tool_args={}),
                allow_shell=True,
                permission_mode="default",
            )
            assert decision.decision == "allow", tool.name
