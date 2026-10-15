"""context_window.py 的单元测试：按 id 查窗口、查不到走默认档。全部本地执行。"""

from __future__ import annotations

import re
from pathlib import Path

from mini_code_agent import context_window
from mini_code_agent.context_window import (
    CONTEXT_WINDOWS,
    UNKNOWN_MODEL_WINDOW,
    window_for,
)


def test_a_known_model_gets_its_own_window():
    assert window_for("deepseek-flash") == 1_048_576


def test_an_unknown_model_falls_back():
    assert window_for("某个还没听说过的模型") == UNKNOWN_MODEL_WINDOW


def test_none_falls_back_too():
    """`_chat` 的 model 参数默认就是 None（测试里走得到）——那也当"不在表里"。"""
    assert window_for(None) == UNKNOWN_MODEL_WINDOW


def test_the_lookup_is_exact_not_a_prefix():
    """**不能按前缀猜。** `claude-sonnet-5` 和 `claude-sonnet-4-5-20250929` 差五倍，而
    `gpt-4o` / `gpt-4o-mini` 一样只是巧合。猜错的后果是阈值跟着错，那个数直接决定什么时候压。"""
    assert window_for("claude-sonnet-5") == 1_000_000
    assert window_for("claude-sonnet-4-5-20250929") == 200_000
    # 只是长得像的，一个字都不能含糊
    assert window_for("claude-sonnet-5-20250101") == UNKNOWN_MODEL_WINDOW


def test_deepseek_is_reachable_by_both_names():
    """项目的 `DEFAULT_MODEL` 用的是旧名 `deepseek-v4-flash`，而端点现在主推 `deepseek-flash`
    （旧名照收、账单一样）。漏了旧名的话，**这个项目自己**会在默认档（40 万）上跑。"""
    assert window_for("deepseek-v4-flash") == window_for("deepseek-flash")


def test_every_window_is_positive():
    """0 或负数会让阈值变成一个没意义的数——`cut_point` 那边专门有一支在防它。"""
    for model, window in CONTEXT_WINDOWS.items():
        assert window > 0, model


def test_the_default_is_positive():
    assert UNKNOWN_MODEL_WINDOW > 0


def test_no_model_is_listed_twice():
    """一张手维护的表里，重复的键会**静默地**取最后一个——字典本身看不出来这回事。

    所以去源码里数一遍键：数出来的比字典多，就说明有人写重了。
    """
    source = Path(context_window.__file__).read_text(encoding="utf-8")
    written = re.findall(r'^ {4}"([^"]+)":', source, flags=re.M)

    assert len(written) == len(CONTEXT_WINDOWS)
    assert len(set(written)) == len(written)
