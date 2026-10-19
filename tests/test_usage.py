"""每轮任务的 token 记账：累计、快照、忽略假 provider 的全 0 响应。"""

from __future__ import annotations

import threading

from mini_code_agent.providers.base import ModelResponse
from mini_code_agent.usage import UsageMeter, UsageTotals


def test_the_three_input_fields_add_up_without_overlap():
    totals = UsageTotals()

    totals.add(
        ModelResponse(
            input_tokens=10,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=5,
            output_tokens=20,
        )
    )

    assert totals.requests == 1
    assert totals.total_input == 105


def test_a_response_without_usage_is_not_counted():
    """假 provider 报的全是 0——那不是"这次真的没花钱"，别记进账里。"""
    meter = UsageMeter()

    meter.record(ModelResponse())
    meter.record(ModelResponse(text="好"))

    assert meter.snapshot().requests == 0


def test_usage_from_several_threads_is_accumulated():
    """后台任务在 daemon 线程里发请求，累加必须加锁。"""
    meter = UsageMeter()

    def work() -> None:
        for _ in range(100):
            meter.record(ModelResponse(input_tokens=1, output_tokens=1))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    totals = meter.snapshot()
    assert totals.requests == 400
    assert totals.input_tokens == 400
    assert totals.output_tokens == 400


def test_a_snapshot_does_not_change_after_it_was_taken():
    meter = UsageMeter()
    meter.record(ModelResponse(input_tokens=1))

    snapshot = meter.snapshot()
    meter.record(ModelResponse(input_tokens=1))

    assert snapshot.requests == 1
    assert snapshot.input_tokens == 1
    assert meter.snapshot().requests == 2
