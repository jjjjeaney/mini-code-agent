"""results.py 的单元测试：两个模式名，以及不认识的值不兜底。"""

from __future__ import annotations

import pytest

from mini_code_agent.results import DEFAULT_RESULTS, RESULTS_MODES, parse_results


def test_only_two_modes():
    assert RESULTS_MODES == ("full", "hidden")


def test_the_default_is_full():
    """默认必须是"照旧"：升级完不能让人以为工具没跑。"""
    assert DEFAULT_RESULTS == "full"


@pytest.mark.parametrize("value", RESULTS_MODES)
def test_each_mode_is_accepted(value):
    assert parse_results(value) == value


def test_case_and_surrounding_space_do_not_matter():
    assert parse_results("  Hidden ") == "hidden"


def test_an_unknown_mode_is_rejected_and_the_choices_are_listed():
    with pytest.raises(ValueError, match="不是结果显示模式") as exc:
        parse_results("brief")

    assert "full" in str(exc.value)
    assert "hidden" in str(exc.value)
