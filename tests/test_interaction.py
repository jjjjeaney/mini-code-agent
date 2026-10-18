"""interaction.py 的单元测试：授权请求队列、取消、关闭、提问规则。

线程只用来验证"submit 真的会等"；其余全部同步执行。
"""

from __future__ import annotations

import threading
import time

import pytest

from mini_code_agent.interaction import (
    BrokerPrompter,
    PermissionBroker,
    ask_permission,
)
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompts import Answer, NobodyToAskError


def a_request(tool: str = "write_file", path: str = "a.py") -> PermissionRequest:
    return PermissionRequest(tool_name=tool, tool_args={"path": path, "content": "x"})


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("等的东西没有出现")


def wait_for_item(broker: PermissionBroker):
    item = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        item = broker.take()
        if item is not None:
            return item
        time.sleep(0.005)
    raise AssertionError("请求没有入队")


class TestTheQueue:
    def test_submit_blocks_until_someone_answers(self):
        broker = PermissionBroker()
        result: list[Answer] = []
        thread = threading.Thread(
            target=lambda: result.append(broker.submit(a_request(), "[x] "))
        )
        thread.start()

        item = wait_for_item(broker)
        assert result == []  # 还等着
        broker.resolve(item, Answer(True))

        thread.join(2)
        assert result == [Answer(True)]

    def test_it_is_first_in_first_out(self):
        broker = PermissionBroker()
        first = broker.defer(a_request(), "[a] ")
        second = broker.defer(a_request(), "[b] ")

        assert broker.take() is first
        assert broker.take() is second
        assert broker.take() is None

    def test_pending_reports_the_labels(self):
        broker = PermissionBroker()
        broker.defer(a_request(), "[general-purpose] ")
        broker.defer(a_request(), "[fork] ")

        assert broker.pending() == ["[general-purpose] ", "[fork] "]

    def test_on_defer_fires_when_the_queue_goes_from_empty_to_nonempty(self):
        broker = PermissionBroker()
        calls: list[bool] = []
        broker.on_defer = lambda: calls.append(True)

        broker.defer(a_request(), "[a] ")
        broker.defer(a_request(), "[b] ")
        assert calls == [True]

        broker.take()
        broker.take()
        broker.defer(a_request(), "[c] ")
        assert calls == [True, True]


class TestCancellation:
    def test_cancelling_interrupts_a_waiting_submit(self):
        broker = PermissionBroker()
        cancel = threading.Event()
        result: list[Answer] = []
        thread = threading.Thread(
            target=lambda: result.append(broker.submit(a_request(), "[x] ", cancel))
        )
        thread.start()
        wait_until(lambda: broker.pending() == ["[x] "])

        cancel.set()

        thread.join(2)
        assert result[0].allowed is False
        assert "取消" in result[0].note
        assert broker.pending() == []

    def test_take_skips_a_request_whose_task_is_already_cancelled(self):
        broker = PermissionBroker()
        cancel = threading.Event()
        cancel.set()
        item = broker.defer(a_request(), "[x] ", cancel)

        assert broker.take() is None
        assert item.future.result(timeout=0.1).allowed is False

    def test_resolve_is_idempotent(self):
        broker = PermissionBroker()
        item = broker.defer(a_request(), "[x] ")

        broker.resolve(item, Answer(True))
        broker.resolve(item, Answer(False))

        assert item.future.result(timeout=0.1) == Answer(True)


class TestClosing:
    def test_close_rejects_the_queued_and_all_later_requests(self):
        broker = PermissionBroker()
        first = broker.defer(a_request(), "[a] ")
        second = broker.defer(a_request(), "[b] ")

        broker.close("会话结束了")

        assert first.future.result(timeout=0.1).allowed is False
        assert second.future.result(timeout=0.1).allowed is False
        assert broker.defer(a_request(), "[c] ") is None
        assert broker.submit(a_request(), "[d] ").allowed is False


class TestBrokerPrompter:
    def test_it_submits_and_waits_for_the_answer(self):
        broker = PermissionBroker()
        prompter = BrokerPrompter(broker, None, "[fork] ")
        result: list[Answer] = []
        thread = threading.Thread(
            target=lambda: result.append(prompter.confirm_edit(a_request()))
        )
        thread.start()

        item = wait_for_item(broker)
        assert item.label == "[fork] "
        broker.resolve(item, Answer(True))

        thread.join(2)
        assert result == [Answer(True)]

    def test_a_choice_question_cannot_be_asked_from_the_background(self):
        prompter = BrokerPrompter(PermissionBroker(), None, "[fork] ")

        with pytest.raises(NobodyToAskError):
            prompter.single_choice("选哪个", ["a", "b"])


class RecordingPrompter:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, PermissionRequest]] = []

    def confirm_tool(self, request):
        self.calls.append(("tool", request))
        return self.answer

    def confirm_edit(self, request):
        self.calls.append(("edit", request))
        return self.answer

    def single_choice(self, question, options):  # pragma: no cover
        raise AssertionError("不该走到")


class TestAskPermission:
    def test_edit_tools_go_through_confirm_edit(self):
        prompter = RecordingPrompter(Answer(True))

        ask_permission(prompter, a_request("write_file"), Grants())

        assert [kind for kind, _ in prompter.calls] == ["edit"]

    def test_other_tools_go_through_confirm_tool(self):
        prompter = RecordingPrompter(Answer(True))

        ask_permission(prompter, a_request("bash"), Grants())

        assert [kind for kind, _ in prompter.calls] == ["tool"]

    def test_a_grant_is_recorded_only_when_allowed_and_remembered(self):
        grants = Grants()
        request = a_request()

        ask_permission(RecordingPrompter(Answer(True, remember=True)), request, grants)
        assert grants.covers(request) is True

        other = a_request(path="b.py")
        ask_permission(RecordingPrompter(Answer(False, remember=True)), other, grants)
        assert grants.covers(other) is False

    def test_remember_false_never_records(self):
        grants = Grants()
        request = a_request()

        ask_permission(
            RecordingPrompter(Answer(True, remember=True)), request, grants, remember=False
        )

        assert grants.covers(request) is False

    def test_a_covered_request_never_reaches_the_prompter(self):
        grants = Grants()
        request = a_request()
        grants.remember(request)
        prompter = RecordingPrompter(Answer(False))

        assert ask_permission(prompter, request, grants) == Answer(True)
        assert prompter.calls == []
