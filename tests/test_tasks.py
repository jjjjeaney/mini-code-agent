"""core/tasks.py 的单元测试：任务表、完成通知、取消、过程缓冲。

线程不靠 sleep：任务自己用 Event 卡住，测试用 `TaskManager.wait` 等它结束。
"""

from __future__ import annotations

import threading

from mini_code_agent.core.agent import CANCELLED_ERROR, AgentResult
from mini_code_agent.core.tasks import MAX_LINES, TaskManager, format_duration
from mini_code_agent.tools import spill
from mini_code_agent.tools.safety import MAX_TOOL_CHARS


def a_result(final: str | None = "结论", *, error: str | None = None) -> AgentResult:
    return AgentResult(final=final, trace=[], messages=[], error=error)


def start(manager: TaskManager, work, tmp_path, *, spill_dir=None) -> str:
    return manager.start(
        work,
        label="[explore] ",
        agent_type="explore",
        description="找压缩逻辑",
        cwd=tmp_path,
        spill_dir=spill_dir,
    )


class TestNotices:
    def test_a_finished_task_is_notified_exactly_once(self, tmp_path):
        manager = TaskManager()
        task_id = start(manager, lambda emit, cancel: a_result("找到三处"), tmp_path)
        assert manager.wait(task_id, timeout=2)

        notice = manager.take_notices()

        assert task_id in notice
        assert "完成" in notice
        assert "找到三处" in notice
        # 取走即标记：空闲唤醒和主 agent 的下一步可能同时来要，不能两边都说一遍。
        assert manager.take_notices() is None

    def test_a_failed_task_says_what_went_wrong(self, tmp_path):
        def boom(emit, cancel):
            raise RuntimeError("磁盘满了")

        manager = TaskManager()
        task_id = start(manager, boom, tmp_path)
        assert manager.wait(task_id, timeout=2)

        notice = manager.take_notices()

        assert "没跑成" in notice
        assert "RuntimeError: 磁盘满了" in notice

    def test_a_task_that_gives_no_conclusion_says_so(self, tmp_path):
        manager = TaskManager()
        task_id = start(manager, lambda emit, cancel: a_result(None), tmp_path)
        assert manager.wait(task_id, timeout=2)

        assert "没给出结论" in manager.take_notices()

    def test_a_cancelled_task_says_so(self, tmp_path):
        began = threading.Event()

        def work(emit, cancel):
            began.set()
            cancel.wait(2)
            return a_result(None, error=CANCELLED_ERROR)

        manager = TaskManager()
        task_id = start(manager, work, tmp_path)
        assert began.wait(2)

        assert manager.cancel(task_id) is True
        assert manager.wait(task_id, timeout=2)

        notice = manager.take_notices()
        assert "已取消" in notice
        assert "不会回流" in notice

    def test_cancelling_a_finished_task_is_a_no_op(self, tmp_path):
        manager = TaskManager()
        task_id = start(manager, lambda emit, cancel: a_result(), tmp_path)
        assert manager.wait(task_id, timeout=2)

        assert manager.cancel(task_id) is False

    def test_a_long_conclusion_lands_on_disk(self, tmp_path):
        """太长就按任务 id 落盘，通知里给 `read_tool_result` 的 id——不新增取回工具。"""
        spill_dir = tmp_path / ".agent" / "tool-results" / "s1"
        long_text = "x" * (MAX_TOOL_CHARS + 100)
        manager = TaskManager()
        task_id = start(
            manager, lambda emit, cancel: a_result(long_text), tmp_path, spill_dir=spill_dir
        )
        assert manager.wait(task_id, timeout=2)

        notice = manager.take_notices()

        assert "read_tool_result" in notice
        assert f'"{task_id}"' in notice
        assert (spill_dir / f"{task_id}.txt").read_text(encoding="utf-8") == long_text


class TestTheRecord:
    def test_the_id_can_be_used_as_a_file_name(self, tmp_path):
        manager = TaskManager()
        task_id = start(manager, lambda emit, cancel: a_result(), tmp_path)

        assert spill._SAFE_ID.fullmatch(task_id)

    def test_the_process_buffer_keeps_the_newest_lines(self, tmp_path):
        def chatty(emit, cancel):
            for index in range(MAX_LINES + 5):
                emit(f"line {index}", "dim")
            return a_result("ok")

        manager = TaskManager()
        task_id = start(manager, chatty, tmp_path)
        assert manager.wait(task_id, timeout=2)

        record = manager.get(task_id)

        assert len(record.lines) == MAX_LINES
        assert record.lines[-1] == f"line {MAX_LINES + 4}"
        assert record.lines[0] == "line 5"

    def test_running_counts_only_unfinished_tasks(self, tmp_path):
        manager = TaskManager()
        task_id = start(manager, lambda emit, cancel: a_result(), tmp_path)
        assert manager.wait(task_id, timeout=2)

        assert manager.running() == 0

    def test_on_finish_fires_once(self, tmp_path):
        seen = []
        manager = TaskManager()
        manager.set_on_finish(seen.append)

        task_id = start(manager, lambda emit, cancel: a_result(), tmp_path)
        assert manager.wait(task_id, timeout=2)

        assert [record.id for record in seen] == [task_id]

    def test_on_finish_fires_for_a_failure_too(self, tmp_path):
        seen = []
        manager = TaskManager()
        manager.set_on_finish(seen.append)

        def boom(emit, cancel):
            raise ValueError("坏了")

        task_id = start(manager, boom, tmp_path)
        assert manager.wait(task_id, timeout=2)

        assert len(seen) == 1
        assert seen[0].status == "failed"


class TestDuration:
    def test_seconds(self):
        assert format_duration(0) == "0s"
        assert format_duration(42.9) == "42s"

    def test_minutes(self):
        assert format_duration(81) == "1m21s"
