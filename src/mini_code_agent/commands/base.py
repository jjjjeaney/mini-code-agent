"""斜杠命令的形状：一条命令长什么样、跑它的时候手边有什么。

和 tools/base.py 是一个套路的另一半：那边定义"模型能调什么"，这边定义"用户能在循环里
直接敲什么"。两边都只描述形状，不认识具体是哪几条。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console

from mini_code_agent.permissions import Grants
from mini_code_agent.sessions import Session

# 一条命令跑完只有两种结果：办完了接着读下一行，或者该退出循环了。
CommandOutcome = Literal["handled", "stop"]

# 派发**一整行输入**的结果，比上面多一种"这一行不是命令"——那就照常发给模型。
Outcome = Literal["not_a_command", "handled", "stop"]


@dataclass
class ChatState:
    """交互循环里**会变**的东西。

    命令读写的就是这里，循环每轮从这儿现读一次。不变的东西不放进来：cwd、provider、
    registry 都是启动时定死的，命令碰不到、也不该碰。

    为什么要有这么个盒子：档位和 messages 原来是 `_chat` 的两个局部量，命令够不着——参数
    是传值的，改了手边那份外面不动。要让它俩改得动、也读得到，就得有个两边都指着的对象。
    以后再有"能中途换的旋钮"，都加在这儿。
    """

    permission_mode: str
    # 这一轮要发出去的那份历史（**内存里这份**，不是文件里那份：压过之后前缀换成了摘要）。
    messages: list[dict]


@dataclass
class CommandContext:
    """跑一条命令时摆在它手边的东西。

    分两类：`state` 和 `grants` 指的是**循环里那两个活着的对象**，命令改了它们，下一轮就按
    新的走；其余几个是这次运行的环境，命令只读。

    命令不碰 provider，也不碰工具注册表：注册表是给模型的，跟命令没关系；后端是 provider
    自己的实现——`/status` 连模型名都不印，就是为了不去扩 Provider 那个只声明了 complete
    的协议（见 builtin.make_status）。
    """

    console: Console
    state: ChatState
    # 这次运行的环境，只读
    cwd: Path
    allow_shell: bool
    session: Session
    # 本会话批过什么。指的是 _chat 里那个 Grants、不是副本——/grants clear 清的是真的那个。
    grants: Grants
    # 就地压一次历史（别管现在多长），返回该跟用户补的一句话；None = 没什么可说的。
    #
    # 是个**动作**、不是某个对象：闭包里已经抓好了 provider，命令层因此到现在都不认识后端。
    # 这里不塞 provider 是有意的——Provider 协议（providers/base.py）只声明了 complete 一个
    # 方法，为了 commands/ 方便就去扩它，等于让所有假 provider 跟着多背字段。
    compact: Callable[[], str | None]


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
