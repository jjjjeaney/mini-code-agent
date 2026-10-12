"""斜杠命令的形状：一条命令长什么样、跑它的时候手边有什么。

和 tools/base.py 是一个套路的另一半：那边定义"模型能调什么"，这边定义"用户能在循环里
直接敲什么"。两边都只描述形状，不认识具体是哪几条。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rich.console import Console

# 一条命令跑完只有两种结果：办完了接着读下一行，或者该退出循环了。
CommandOutcome = Literal["handled", "stop"]

# 派发**一整行输入**的结果，比上面多一种"这一行不是命令"——那就照常发给模型。
Outcome = Literal["not_a_command", "handled", "stop"]


@dataclass
class ChatState:
    """交互循环里**会变**的运行参数。

    命令改的就是这里，循环每轮从这里现读一次。不变的东西不放进来：cwd、provider、
    registry 都是启动时定死的，命令碰不到、也不该碰。

    为什么要有这么个盒子：档位原来是 `_chat` 的一个参数，一路传进 runagent。参数是传值的，
    命令改了手边那份，外面那份不动；要让它改得动，就得有个两边都指着的对象。以后再有
    "能中途换的旋钮"，都加在这儿。
    """

    permission_mode: str


@dataclass
class CommandContext:
    """跑一条命令时摆在它手边的东西：往哪儿打印，以及能改的运行态。

    命令不碰 provider / registry / 会话落盘——它的活儿是**改下一轮怎么跑**，
    不是自己去跑一轮。
    """

    console: Console
    state: ChatState


@dataclass
class Command:
    """一条斜杠命令。

    name 是不带斜杠的那段（`/help` 的 name 是 `help`）。run 收到命令行剩下的那截原文，
    已经去掉首尾空白：**不预先拆参数、不认 --flag**——命令就这么几条，真要分参数自己
    split 就行，在这儿定一套解析规则反而多一层。
    """

    name: str
    run: Callable[[CommandContext, str], CommandOutcome]
    summary: str = ""
    usage: str = ""
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("命令名不能为空（斜杠后面那一段）")
        # 名字里带空白就永远查不到——分派时按空白切名字，切出来的只会是前半截。
        if not all(ch.isalnum() or ch in "_-" for ch in self.name):
            raise ValueError(f"命令名 {self.name!r} 只能是字母、数字、下划线和连字符")


def split_command(line: str) -> tuple[str, str]:
    """把一行命令拆成（名字, 参数）。**line 得是以 `/` 开头的那一行。**

    名字是斜杠后面到第一个空白为止的那段。剩下的整段都算参数，空白原样留着——
    `accept edits` 这种带空格的档位名，拆细了还得拼回去。
    """
    parts = line[1:].split(None, 1)
    if not parts:
        return "", ""
    return parts[0], parts[1].strip() if len(parts) > 1 else ""
