"""工具的格式定义。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReadFileState:
    # path -> (mtime_ns, size_bytes)。read-before-edit 要靠它判断"模型读过这个文件之后，
    # 文件在磁盘上是不是又被改过"。
    #
    # 存两个信号是因为 mtime 一个不够用：两次写落在同一个时钟刻度里，时间戳会**一模一样**
    # （实测这台机器上间隔 10ms 以内经常撞），那种时候只有长度还拦得住。
    entries: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def record(self, path: Path, size: int, mtime_ns: int) -> None:
        # size 和 mtime_ns 都由调用方量好传进来，这里不再自己去 stat。
        # 让这里 stat 的话，取到的是"记录那一刻"的状态，而内容属于更早的版本——
        # 两者一错开，读-before-改的守卫就会以为模型看到的就是磁盘上最新的东西，
        # 而它要防的恰恰就是这一类错位。
        #
        # size 是**字节数**（st_size），不是字符数：字符数得先把文件解码出来才知道，
        # 而且对非 ASCII 文件，同一份内容的字符数和磁盘上的字节数根本对不上，没法比。
        self.entries[path] = (mtime_ns, size)


class ToolWriteError(Exception):
    """改文件的工具出的错。子类区分是**哪一步**失败的。

    分这么细不是为了好看：模型看到 WriteFailedError 知道"原文件没被动过，可以重试"，
    看到 StaleReadError 知道"该重读一遍"。报错原文会被 dispatch 交给模型，说清是哪
    一步、原文件有没有被改动，比一句笼统的"写入失败"有用得多。
    """


class StaleReadError(ToolWriteError):
    """状态对不上：没读过这个文件，或者读过之后它又被改过。原文件没有被改动。"""


class BackupFailedError(ToolWriteError):
    """备份原文件那一步失败了。原文件没有被改动。"""


class WriteFailedError(ToolWriteError):
    """写临时文件那一步失败了。原文件没有被改动。"""


class ReplaceFailedError(ToolWriteError):
    """临时文件写好了，但顶替原文件那一步失败了。原文件没有被改动。"""


class ToolShellError(Exception):
    """跑命令这个工具出的错。子类区分是**哪一步**失败的。

    和"命令跑了但退出码非 0"要分开：那是命令的结果，交回模型让它自己看；
    下面是工具自己没跑起来，模型再怎么改命令也没用。
    """


class ShellDisabledError(ToolShellError):
    """这个 agent 没开命令执行能力（ToolContext.allow_shell 是 False）。"""


class DangerousCommandError(ToolShellError):
    """命令落在危险命令黑名单上。命令没有执行过。

    和 ShellDisabledError 都是"根本没跑"，但原因不同：那个是能力没开，这个是**判断它不该跑**。
    它是启发式（见 safety.py 的 DENY_PATTERNS），不是边界——但模型收到它就该换个做法，而不是
    改写成别的样子再试。
    """


class DockerUnavailableError(ToolShellError):
    """docker 用不了：命令没找到，或者 daemon 没起来。命令没有执行过。"""


class ContainerStartFailedError(ToolShellError):
    """容器没起来（镜像不存在、参数不对之类）。命令没有执行过。"""


@dataclass
class ToolContext:
    """调用工具时的运行环境。

    工具需要的外部信息都从这里拿，而不是去读全局状态——比如 cwd，
    路径校验就拿它当围栏的边界。

    cwd 是必填的：让工具在缺省时回落到进程当前目录，等于悄悄换了沙箱的边界。

    allow_shell 管的是能不能跑命令。默认关着——执行任意命令是这个 agent 里权限最大
    的一件事，不该因为"顺手"就默认打开。
    """

    cwd: Path
    read_state: ReadFileState=field(default_factory=ReadFileState)
    allow_shell: bool = False

    # 落盘的工具结果往哪儿放（见 tools/spill.py）。CLI 传的是**这次会话的目录**
    # （`.agent/tool-results/<会话 id>/`），这样删会话时能一起端掉。
    #
    # 缺省（None）落在项目下那个共用的目录里——测试、以及不经 CLI 直接调 runagent 的
    # 走的都是这条。传的是**一个目录**而不是会话 id：工具层不需要认识"会话"这个概念，
    # 它只需要知道往哪儿放。
    spill_dir: Path | None = None

    # 这次运行跑在哪个权限档下（四档见 permissions.py）。
    #
    # **它和 prompter 不是一类东西，这正是它能在 ToolContext 里的原因。** prompter 是
    # "能力"——工具能不能自己问用户，所以被挡在工具层外面（prompts.py 那条架构说明，
    # 以及 test_prompt.py 里钉它的用例）。档位是"环境"：跟 cwd、allow_shell 一样，
    # 回答的是"工具此刻跑在什么条件下"。permissions.decide_permission 的签名也没动，
    # 它照旧只收那个字符串——那条例外写在那边的 docstring 里。
    #
    # 谁要用：**子 agent**。它必须继承主 agent 当前的档位，不然 plan 档下派一个
    # general-purpose 子 agent 就能改文件，那是 plan 档的一个洞。而档位会被
    # `/permission_mode` 中途改掉，所以不能靠"建工具那一刻"的值——只能每轮现读，
    # 这就是它必须走 ctx 而不是闭包捕获的原因。
    #
    # 默认取 "default" 落在**会问**的那一侧（decide_permission 对写文件给的是 ask，
    # 不是放行），漏传的后果是多问一句而不是少问一句。而且这个默认跟着 runagent 自己的
    # `permission_mode: str = "default"` 走，没有另开一条路——生产代码里 ToolContext
    # 只有 core/agent.py 一处构造，那里永远传的是真值。
    permission_mode: str = "default"


def _always_available(_: ToolContext) -> bool:
    """默认：一直能用。绝大多数工具都是这样。"""
    return True


@dataclass
class Tool:
    """一个可以被模型调用的工具。

    params 是 JSON Schema，会原样成为请求里的 input_schema，留空表示这个工具不收参数。
    fn 收到两样东西：模型给的参数字典，和一个 ToolContext。返回值由 ToolRegistry
    包成 ToolResult 交回模型。
    """

    name: str
    fn: Callable[[dict[str, Any], ToolContext], Any]
    description: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    # 这个工具**此刻**能不能用。不给就是一直能用。
    #
    # 为什么要这么个东西：广告一个做不到的能力，代价不只是难看——模型会试一次、吃一条报错、
    # 而且**花掉一步**（max_steps 数的是模型调用次数）。"没有"就别发。
    #
    # **它只管"发不发"，不管"放不放行"**。闸门（permissions.py）照旧管后者，而且必须照旧：
    # 模型在这条对话里见过这个 schema（上一步的请求里就有），照样调得出来——所以"不发"只能
    # 是加法，替代不了闸门。
    #
    # 判的也只有"存不存在"这一类（没开 shell、没配 key、一份技能/记忆都没有）。**权限不归
    # 它管**：那是闸门的事，而且藏起来会连"为什么不行"那句话一起藏掉。
    available: Callable[[ToolContext], bool] = _always_available

    # 结果太长时**落盘**，上下文里只留一段预览（见 tools/spill.py）。不给就是不落盘，
    # 超了照旧截断。
    #
    # 给谁开、为什么：只给那些**输出取不回来**的工具开。命令有副作用、网页下一刻会变，
    # 截掉了就是真没了，重跑一遍未必等价。反过来，read_file 就算读满 50000 字符也不该
    # 落盘——原文件本来就在盘上，start_line / end_line 能读回来，而且**读了会记
    # read_state**（改成读落盘的副本就断了这条，模型改原件时会被"你还没读过它"拦住）。
    spill: bool = False

    # **只从外界取信息**：不写盘、不跑命令、不向用户输出、不改长期记忆。
    #
    # 判的是"这个动作会不会改变什么"，**不是**"它属于哪一类"——所以 `echo` 不算只读，
    # 它往终端打印，那是输出。
    #
    # 谁在用：子 agent 里那两个只读类型（explore / plan）的工具池按它筛，见
    # core/subagent.py。以后做工具并行时，"谁可以并行"基本也是这份名单。
    #
    # **默认 False，而且绝不能反过来。** 忘了给一个新工具标上的后果，是它被排除出只读池
    # （少一个工具可用），不是它被放进 plan 子 agent 的池子里（那才是灾难）。
    # tests/test_tools.py 里有两条例出整份名单的用例，加新工具时会红、逼人做这个决定
    # ——那是这个默认值的安全网，别把它们删了。
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("工具名不能为空")
        if not callable(self.fn):
            raise TypeError(f"工具 {self.name!r} 的 fn 必须可调用")

    def to_anthropic_schema(self) -> dict[str, Any]:
        """转成 Messages API 的 tools 条目。description 为空时不带这个键，免得发 null 过去。"""
        schema: dict[str, Any] = {"name": self.name}
        if self.description:
            schema["description"] = self.description
        schema["input_schema"] = self.params or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return schema


@dataclass
class ToolResult:
    """一次工具调用的结果。

    id 取自发起这次调用的 ToolCall.id，回填时模型靠它把结果和调用配对，不能错。
    name 只是本地元信息，给日志和 trace 看，协议里用不到。
    result 保留工具返回的原值，序列化推迟到出网的那一刻，这样 trace 里还是结构化的数据。
    """

    id: str
    name: str
    result: Any = None
    is_error: bool = False

    def to_anthropic_block(self) -> dict[str, Any]:
        """转成可以直接回填进 messages 的 tool_result block。"""
        content = self.result
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": content,
        }
        if self.is_error:
            block["is_error"] = True
        return block
