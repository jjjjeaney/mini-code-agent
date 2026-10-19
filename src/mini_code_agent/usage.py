"""一轮任务的 token 记账。

一次用户消息可能触发多发布请求：主 agent 的每一步、工具调用之后的续发、以及派出去的子
agent。结尾要把它们加起来，所以 `runagent` 每轮新建一个 `UsageMeter`，子 agent 通过
`ToolContext.usage` 拿到同一个对象，把用量记回这一轮。

只记 token，不换钱：缓存命中大约是全价的 1/50，输出单价通常又比输入高好几倍。没有确认过
的价格表就换算，只会得到一个看起来很确定的错数。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

from mini_code_agent.providers.base import ModelResponse


@dataclass
class UsageTotals:
    """一轮任务累计的用量。字段和 `ModelResponse` 的用量字段一一对应。"""

    requests: int = 0
    # 没走缓存、按全价算的输入
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input(self) -> int:
        """三个输入字段**不重叠**，加起来才是真正送进去的量。"""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def add(self, response: ModelResponse) -> None:
        self.requests += 1
        self.input_tokens += response.input_tokens
        self.cache_read_input_tokens += response.cache_read_input_tokens
        self.cache_creation_input_tokens += response.cache_creation_input_tokens
        self.output_tokens += response.output_tokens


class UsageMeter:
    """线程安全的累加器。

    后台任务在 daemon 线程里发请求；同一个 meter 还会被子 agent 共享，所以写要加锁。读用
    快照——打印到一半时另一条线程继续记，不会把输出搅成半新半旧。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = UsageTotals()

    def record(self, response: ModelResponse) -> None:
        """记一发。**全 0 的不记**——那是假 provider，不是"这次真的没花钱"。"""
        if not any(
            (
                response.input_tokens,
                response.cache_read_input_tokens,
                response.cache_creation_input_tokens,
                response.output_tokens,
            )
        ):
            return
        with self._lock:
            self._totals.add(response)

    def snapshot(self) -> UsageTotals:
        with self._lock:
            return replace(self._totals)
