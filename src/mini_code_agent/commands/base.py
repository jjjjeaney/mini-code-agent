"""斜杠命令的形状：一条命令长什么样、跑它的时候手边有什么。

和 tools/base.py 是一个套路的另一半：那边定义"模型能调什么"，这边定义"用户能在循环里
直接敲什么"。两边都只描述形状，不认识具体是哪几条。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console

from mini_code_agent.permissions import Grants
from mini_code_agent.sessions import Session

if TYPE_CHECKING:
    # 只在标注里用。命令层只调任务表的几个公开方法，不需要它的实现。
    from mini_code_agent.core.tasks import TaskManager

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

    # fork 开没开（它是 `agent` 的 type 之一，见 core/subagent.py）。放这儿是因为 `/fork` 能
    # 中途换它，而换完下一轮就得按新的拼提示词——和档位同一个道理。
    #
    # **切它会重建缓存前缀**：提示词里有没有那段说明，属于请求开头的一部分。工具表不变
    # （type 枚举是静态的），但前缀照样从头算。所以命令那边会把代价说清楚，不是悄悄换掉。
    allow_fork: bool = False

    # 上一次**真请求**报的输入总量，以及那一发当时装了多少条 messages。压缩阈值拿这两个算
    # （见 summarizing.context_size）。
    #
    # **两个必须成对，而且必须来自同一发**——所以要改就一起改（下面那两个方法就是干这个的），
    # 别在外面单独赋值。`covered` 尤其不能拿 `AgentResult.messages` 的长度去顶：那份比发出去
    # 的长，响应是先拿到的、assistant 那条是之后才 append 的。
    #
    # `context_tokens` 是 None 表示"还没有过真请求"（会话第一轮、`--resume` 上来还没发过、
    # 或者刚压完），那时只能按字符粗估。
    context_tokens: int | None = None
    context_covered: int = 0

    def remember_context(self, tokens: int, covered: int) -> None:
        """记下锚点：上一次真请求报了多少 token、那一发装了多少条 messages。"""
        self.context_tokens = tokens
        self.context_covered = covered

    def forget_context(self) -> None:
        """锚点作废——消息被换掉了（压缩或者折叠），旧的数还指着换之前的历史。

        不作废的话下一轮会以为上下文还是那么大：够阈值、又刚好攒够 7 轮，就会再压一次，
        阈值被架空成"每 7 轮压一次"。清掉之后下一轮退回粗估，再下一轮就有新的真数。
        """
        self.context_tokens = None
        self.context_covered = 0

    def replaced(self, messages: list[dict]) -> bool:
        """换一份历史（压缩或折叠之后的），并把锚点作废。返回"真换了吗"。

        两个动作**必须成对**——换了历史不作废锚点，下一轮就会按换之前那个大小判断（见
        forget_context）。合成一个方法是因为压缩和折叠都要做同一件事，分开写迟早有一条
        漏掉，而漏掉是静默的。

        "换没换"看的是**对象身份**：两条路都是"没改动就原样返回传进来那份"，所以 `is`
        就是它们的信号（见 summarizing 和 folding 的模块说明）。
        """
        if messages is self.messages:
            return False
        self.messages = messages
        self.forget_context()
        return True


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
    # 这个模型的上下文窗口（token），压缩阈值和 `/status` 都要用它。一次运行里定死，所以放
    # 在这边（"这次运行的环境，只读"），不放 ChatState。
    #
    # 走这条路是为了不碰 provider：同一个约定见下面的注释。
    window: int
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

    # 后台子 agent 的任务表（见 core/tasks.py）。`/tasks` 看状态/取消、`/status` 报个数。
    # 默认 None：测试和一次性路径没有它，命令要能对着空表说话。
    tasks: TaskManager | None = None


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
