"""console.py 的单元测试：动态内容里的方括号别被 rich 当成样式标记吃掉。"""

from __future__ import annotations

from mini_code_agent.console import console


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
