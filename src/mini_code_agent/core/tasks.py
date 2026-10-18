"""后台子 agent 的任务表。

一条 `agent` 调用跑满两分钟（或者模型主动要了 `run_in_background`）之后，`runagent` 会把
**剩下的循环体**包成一个工作函数交到这里，由一条 daemon 线程接着跑。前台那边立刻拿回一个
任务 id 继续干活，不必等它。

## 为什么是"交接"而不是"从第一秒就丢线程"

交接只发生在**两发请求之间**：此刻 `messages` 以一条完整的 tool_result（或最初的任务）
结尾，后台接着往下跑不会留下悬空的 tool_use。更要紧的是前台那两分钟里根本没有第二条线程——
权限提示、Ctrl+C、终端输出都还在主线程上，不存在两个线程抢 stdin 这种事。

## 结果怎么回到模型

`take_notices()` 给主线程取"已经结束、还没通知过"的那些，拼成一段话。主 agent 在**每一发
请求之前**问一次（见 `runagent` 的 `notices`），有就纯追加一条 user 消息——不碰前缀，缓存
照旧。结论太长就按任务 id 落进这次会话的 `tool-results/`，用现有的 `read_tool_result` 取，
不必为后台任务另开一个工具。

`set_on_finish` 那条回调是给交互循环用的：任务一结束就往它的事件队列里放一条，停在提示符上
的 `_chat` 因此会醒过来，让模型主动把结果告诉用户。

## 边界

- **不跨进程**：任务表在内存里，daemon 线程随进程退出被硬杀。只读任务没有副作用，所以这是
  可以接受的；`--resume` 之后历史里那些 id 是死的，`collect` 那类入口不存在，也不会有人去
  按 id 找。
- **后台不碰终端**：交接过去的 prompter 是 `BackgroundPrompter`，需要当场批准的调用一律拒绝。
  主线程可能正停在提示符上，任何读键盘的动作都会和它抢 stdin。
- 并发上限不在这里：`subagent` 在**主动要后台**时硬拦，自动交接不受限（它本来就在跑了）。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_code_agent.core.agent import CANCELLED_ERROR, AgentResult, Emit
from mini_code_agent.tools import spill
from mini_code_agent.tools.safety import MAX_TOOL_CHARS, truncate_text

# 过程输出最多留多少行给 `/tasks show`。后台任务可能跑几十步，全留着只是占内存；
# 用户要看的通常是"最后几步发生了什么"和结论。超了就丢最旧的。
MAX_LINES = 200

Work = Callable[[Emit, threading.Event], AgentResult]


def format_duration(seconds: float) -> str:
    """`42s` / `3m21s`。超过一小时就只报分钟——这种任务本来就不该有。"""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60:02d}s"


@dataclass
class TaskRecord:
    """一个后台任务现在的样子。

    `status`/`final`/`error`/`tally` 都在**同一次** `finish`/`fail` 里写完，然后才置
    `done_event`——外面看到事件被置位，就一定能看到完整的字段（Event 自带这层顺序）。
    """

    id: str
    agent_type: str
    description: str
    label: str
    cwd: Path
    spill_dir: Path | None
    started_at: float
    # 由 subagent 注入（`tally_calls`）。不在这里 import core.subagent：那会把任务表和一个
    # 它根本不需要认识的模块绑在一起。
    tally_fn: Callable[[list[dict[str, Any]]], str] | None = None
    status: str = "running"
    final: str | None = None
    error: str | None = None
    tally: str = ""
    lines: list[str] = field(default_factory=list)
    finished_at: float | None = None
    notified: bool = False

    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def emit(self, text: str, style: str) -> None:
        """后台那一半的输出去处。**只缓冲，不打印**——主线程可能正在跑别的一轮。"""
        with self.lock:
            self.lines.append(text)
            if len(self.lines) > MAX_LINES:
                del self.lines[: len(self.lines) - MAX_LINES]

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def snapshot(self) -> TaskRecord:
        """加锁复制一份，给 `/tasks` 和通知拼装用（别把锁带到渲染里去）。"""
        with self.lock:
            return TaskRecord(
                id=self.id,
                agent_type=self.agent_type,
                description=self.description,
                label=self.label,
                cwd=self.cwd,
                spill_dir=self.spill_dir,
                started_at=self.started_at,
                tally_fn=self.tally_fn,
                status=self.status,
                final=self.final,
                error=self.error,
                tally=self.tally,
                lines=list(self.lines),
                finished_at=self.finished_at,
                notified=self.notified,
            )

    def finish(self, result: AgentResult) -> None:
        with self.lock:
            self.finished_at = time.monotonic()
            self.final = result.final
            self.error = result.error
            if result.error == CANCELLED_ERROR:
                self.status = "cancelled"
            elif result.error:
                self.status = "failed"
            else:
                self.status = "done"
            if self.tally_fn is not None:
                self.tally = self.tally_fn(result.messages)

    def fail(self, exc: BaseException) -> None:
        with self.lock:
            self.finished_at = time.monotonic()
            if self.cancel_event.is_set():
                self.status = "cancelled"
                self.error = CANCELLED_ERROR
            else:
                self.status = "failed"
                self.error = f"{type(exc).__name__}: {exc}"

    def is_running(self) -> bool:
        return not self.done_event.is_set()


class TaskManager:
    """一张任务表。CLI 建一份，`agent` 工具和 `/tasks` 共用它。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        self._on_finish: Callable[[TaskRecord], None] | None = None

    def set_on_finish(self, callback: Callable[[TaskRecord], None] | None) -> None:
        """任务结束时调它一次（在锁外）。交互循环拿它往事件队列里放一条，唤醒提示符。"""
        self._on_finish = callback

    def start(
        self,
        work: Work,
        *,
        label: str,
        agent_type: str,
        description: str,
        cwd: Path,
        spill_dir: Path | None,
        tally: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> str:
        """起一条 daemon 线程跑 `work`，返回任务 id。

        id 是 `bg-` 加 12 位随机十六进制：满足 `spill._SAFE_ID`（能直接当落盘文件名），
        而且跨会话不会撞——`find_spilled` 是按名字在**所有**会话目录里搜的，短 id 会串。
        """
        task_id = f"bg-{uuid.uuid4().hex[:12]}"
        record = TaskRecord(
            id=task_id,
            agent_type=agent_type,
            description=description,
            label=label,
            cwd=cwd,
            spill_dir=spill_dir,
            started_at=time.monotonic(),
            tally_fn=tally,
        )
        with self._lock:
            self._tasks[task_id] = record

        def body() -> None:
            try:
                result = work(record.emit, record.cancel_event)
                record.finish(result)
            except BaseException as exc:  # noqa: BLE001 - 线程里的异常只能记在任务上
                record.fail(exc)
            finally:
                record.done_event.set()
                callback = self._on_finish
                if callback is not None:
                    callback(record)

        threading.Thread(target=body, name=f"task-{task_id}", daemon=True).start()
        return task_id

    def wait(self, task_id: str, timeout: float | None = None) -> bool:
        """等一个任务结束。**给测试用**——生产代码不等，等就失去后台的意义了。"""
        record = self.get(task_id)
        return record is not None and record.done_event.wait(timeout)

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[TaskRecord]:
        """按开始时间排好的一份快照。`/tasks` 和 `/status` 用。"""
        with self._lock:
            records = sorted(self._tasks.values(), key=lambda record: record.started_at)
        return [record.snapshot() for record in records]

    def running(self) -> int:
        with self._lock:
            records = list(self._tasks.values())
        return sum(1 for record in records if record.is_running())

    def cancel(self, task_id: str) -> bool:
        """请求取消。**不是立刻停**：后台会在下一次发请求之前看到这个标记。

        已经结束的任务什么也不做，返回 False——命令层据此说"它已经结束了"。
        """
        record = self.get(task_id)
        if record is None or not record.is_running():
            return False
        record.cancel_event.set()
        return True

    def take_notices(self) -> str | None:
        """取走"已经结束、还没通知过"的那些，拼成一段话；没有就返回 None。

        **取走即标记已通知**，所以同一个结果不会在主 agent 的一步里和空闲唤醒的汇报回合里
        说两遍。标记和挑选在同一把锁里，两个线程同时调也只会有一边拿到。
        """
        with self._lock:
            pending = [
                record
                for record in self._tasks.values()
                if record.done_event.is_set() and not record.notified
            ]
            for record in pending:
                record.notified = True
        if not pending:
            return None
        return "\n\n".join(_notice(record.snapshot()) for record in pending)


def _notice(record: TaskRecord) -> str:
    """一条完成通知的正文。**结论太长就落盘**，通知里给预览和 `read_tool_result` 的 id。"""
    head = f"【后台任务 {record.id}"
    elapsed = format_duration(record.elapsed())
    suffix = f"（{record.agent_type}，用时 {elapsed}）"

    if record.status == "cancelled":
        return (
            f"{head} 已取消】{suffix}\n"
            "取消之前它已经做过的事不会回流；需要的话重新派一个。"
        )
    if record.status == "failed":
        return f"{head} 没跑成】{suffix}\n{record.error}"
    if record.final is None:
        return (
            f"{head} 完成了，但没给出结论】{suffix}\n"
            "它可能把步数用满了。可以再派一次、把范围缩窄一点，或者自己来。"
        )
    tally = f"，{record.tally}" if record.tally else ""
    return f"{head} 完成】（{record.agent_type}{tally}，用时 {elapsed}）\n\n{_body(record)}"


def _body(record: TaskRecord) -> str:
    """结论正文。太长就落一份盘——用任务 id 当文件名，`read_tool_result` 能直接取。"""
    final = record.final or ""
    if len(final) <= MAX_TOOL_CHARS:
        return final
    if record.spill_dir is not None:
        saved = spill.save_into(record.spill_dir, "agent", record.id, final)
        if saved is not None:
            return spill.preview_message(
                final,
                name=saved.stem,
                where=spill.display_path(record.cwd, saved),
            )
    # 没地方落盘（测试里没给 spill_dir）：退回普通截断，至少不让几 MB 文本原样进上下文。
    return truncate_text(final, MAX_TOOL_CHARS)
