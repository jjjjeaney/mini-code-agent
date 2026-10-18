"""后台任务的权限请求队列。

后台子 agent 跑在另一条线程上，**不能直接读 stdin**——那会和 REPL 抢同一个输入流。所以它们把
`PermissionRequest` 放进这里，由读键盘线程在**它持有终端**的时候取出来问用户。

几条定下来的：

- **谁问**：只有读键盘线程调 `take()`，然后用真正的 `TerminalPrompter` 渲染和读答案。这个模块
  不 import typer，也不 import core——它是两边中间的一层。
- **为什么用 `Future`**：`submit` 要能等、能取消、能被 `close` 一次性拒掉；`Future` 自带
  "只完成一次"，省掉手写的 answer/Event。
- **取消**：等待循环每 0.1 秒醒一次检查任务的 cancel event；取消后请求从队列里摘掉，`take`
  也不会再把死任务的请求端出来。
- **退出**：`close()` 是单向的；之后 `submit` 立刻拒绝，队列里剩下的也全部拒掉。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from mini_code_agent.permissions import EDIT_TOOLS, Grants, PermissionRequest
from mini_code_agent.prompts import Answer, NobodyToAskError, Prompter


@dataclass
class PendingApproval:
    """一个还没被回答的请求。

    `label` 是 `[general-purpose] ` 这种来源标记，读键盘线程在提问前打出来——后台那一半
    不能自己包 `LabeledPrompter`，那会从后台线程往终端打东西。
    """

    request: PermissionRequest
    label: str
    future: Future[Answer] = field(default_factory=Future)
    cancel: threading.Event | None = None


class PermissionBroker:
    """后台线程和读键盘线程之间的那张请求表。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: list[PendingApproval] = []
        self._closed = False
        # 队列从空变非空时调一次。交互循环拿它往事件队列里放一个"有请求了"，
        # 好让停在提示符上的主循环醒过来打一句提示。
        self.on_defer: Callable[[], None] | None = None

    def defer(
        self,
        request: PermissionRequest,
        label: str,
        cancel: threading.Event | None = None,
    ) -> PendingApproval | None:
        """只入队，不阻塞。测试和 `/tasks` 的构造用它；已经 close 了就返回 None。"""
        with self._lock:
            if self._closed:
                return None
            was_empty = not self._pending
            item = PendingApproval(request=request, label=label, cancel=cancel)
            self._pending.append(item)
        if was_empty and self.on_defer is not None:
            self.on_defer()
        return item

    def submit(
        self,
        request: PermissionRequest,
        label: str,
        cancel: threading.Event | None = None,
    ) -> Answer:
        """后台线程用：入队，然后一直等到有人回答（或者任务被取消 / 会话结束）。"""
        item = self.defer(request, label, cancel)
        if item is None:
            return Answer(False, note="会话已经结束，后台任务不能再请求授权。")

        while True:
            try:
                return item.future.result(timeout=0.1)
            except FutureTimeout:
                if cancel is not None and cancel.is_set():
                    with self._lock:
                        if item in self._pending:
                            self._pending.remove(item)
                    self._resolve(item, Answer(False, note="任务已取消。"))
                    return item.future.result()

    def take(self) -> PendingApproval | None:
        """读键盘线程用：取下一条。已经取消的直接拒掉并跳过，不端给用户。"""
        with self._lock:
            while self._pending:
                item = self._pending.pop(0)
                if item.cancel is not None and item.cancel.is_set():
                    self._resolve_locked(item, Answer(False, note="任务已取消。"))
                    continue
                return item
        return None

    def resolve(self, item: PendingApproval, answer: Answer) -> None:
        """交回答案。幂等：同一个请求被两边同时收尾时只有第一次生效。"""
        with self._lock:
            self._resolve_locked(item, answer)

    def pending(self) -> list[str]:
        """还在排队、没被取走的那些请求的来源标记。给 `/status` 和 `/tasks` 报数。"""
        with self._lock:
            return [item.label for item in self._pending]

    def close(self, reason: str) -> None:
        """收尾：之后的 `submit` 立刻拒绝，队列里剩下的全部拒掉，不等正在被问的那一项。"""
        with self._lock:
            self._closed = True
            items = list(self._pending)
            self._pending.clear()
            for item in items:
                self._resolve_locked(item, Answer(False, note=reason))

    def _resolve(self, item: PendingApproval, answer: Answer) -> None:
        with self._lock:
            self._resolve_locked(item, answer)

    @staticmethod
    def _resolve_locked(item: PendingApproval, answer: Answer) -> None:
        if not item.future.done():
            item.future.set_result(answer)


class BrokerPrompter:
    """后台那一半的 prompter：把问题交给 broker，然后等读键盘线程去问。

    **不打印任何东西**——它跑在后台线程上。来源标记跟着 `PendingApproval.label` 走。
    """

    def __init__(
        self,
        broker: PermissionBroker,
        cancel: threading.Event | None,
        label: str,
    ) -> None:
        self._broker = broker
        self._cancel = cancel
        self._label = label

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        return self._broker.submit(request, self._label, self._cancel)

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        return self._broker.submit(request, self._label, self._cancel)

    def single_choice(self, question: str, options: list[str]) -> str:
        # 选择题没有"y/n 就能回答"的形态，后台问不了。主 agent 拿结论时自己问用户。
        raise NobodyToAskError(
            f"后台任务问不了选择题；把选项和取舍写进结论：{question}"
        )


def ask_permission(
    prompter: Prompter,
    request: PermissionRequest,
    grants: Grants,
    *,
    remember: bool = True,
) -> Answer:
    """问一次权限，并按用户的回答（可选地）记进 Grants。

    `runagent` 的前台路径和读键盘线程服务后台请求时共用这一份，免得两边的"什么时候记 Grants"
    慢慢走偏。`remember=False` 给后台服务端用：它要在拿到答案之后再检查任务是否已取消，
    已取消的任务不该留下授权。
    """
    if grants.covers(request):
        return Answer(True)
    if request.tool_name in EDIT_TOOLS:
        answer = prompter.confirm_edit(request)
    else:
        answer = prompter.confirm_tool(request)
    if remember and answer.allowed and answer.remember:
        grants.remember(request)
    return answer
