"""console.py 的单元测试：方括号不被当成样式，以及输出编码的兜底。

Windows 上重定向输出时 Python 的 stdout 会用系统 ANSI 代码页（常见 GBK），而输出里有
`▶` / `→` / `✗`——不切到 UTF-8 就会在 Rich 的 legacy 渲染器里抛 UnicodeEncodeError。
"""

from __future__ import annotations

from mini_code_agent.console import _ensure_utf8, console


def test_an_array_in_the_content_survives(capsys):
    """工具结果里常有整行 JSON，`[1, 2]` 不能被解析成样式。"""
    console.print("out: [1, 2] end", markup=False)

    assert capsys.readouterr().out == "out: [1, 2] end\n"


def test_something_that_looks_like_a_tag_survives(capsys):
    console.print("text has [red] inside", markup=False)

    assert capsys.readouterr().out == "text has [red] inside\n"


def test_colors_are_dropped_when_the_output_is_not_a_terminal(capsys):
    """`mca ... > out.txt` 拿到的得是干净文本。"""
    console.print("坏了", style="red", markup=False)

    assert capsys.readouterr().out == "坏了\n"


class FakeStream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.calls: list[dict] = []

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_a_gbk_stream_is_switched_to_utf8():
    stream = FakeStream("gbk")

    _ensure_utf8(stream)

    assert stream.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_a_utf8_stream_is_left_alone():
    stream = FakeStream("utf-8")

    _ensure_utf8(stream)

    assert stream.calls == []


def test_a_cp65001_stream_is_left_alone():
    stream = FakeStream("cp65001")

    _ensure_utf8(stream)

    assert stream.calls == []


def test_a_stream_without_reconfigure_does_not_crash():
    class NoReconfigure:
        encoding = "gbk"

    _ensure_utf8(NoReconfigure())


def test_a_broken_reconfigure_does_not_crash():
    class Broken:
        encoding = "gbk"

        def reconfigure(self, **kwargs):
            raise ValueError("流已经关了")

    _ensure_utf8(Broken())


def test_none_is_a_no_op():
    _ensure_utf8(None)
