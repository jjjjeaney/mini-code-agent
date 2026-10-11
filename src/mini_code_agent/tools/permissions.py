"""权限模式：用户指定的那一档，以及判定要用的两个类型。

三种模式管的都是"改文件的工具要不要一个个问"：

- `accept edit`：默认同意。模型改文件不再逐个确认——给"我知道它在改什么、别老打断我"
  的时候用
- `default`：要用户确认。每次改文件都停下来问一句
- `plan`：默认拒绝。计划模式——只读，先让模型把打算怎么做讲清楚，再切出去执行

bash 不在这三档里：**不论什么模式都要用户确认**。一条命令会干什么，用户事先看不出来
（沙箱挡的是它碰不到宿主机，不是它做的事对不对），所以它不跟着模式走。

这些是**默认**答案：模式定的是"不问的时候怎么办"，具体到某一次调用还可能被别的条件
改成要问（比如路径在不在围栏里）。这个模块管两件事：把用户敲的档位解析成规范值，
以及判定要用的请求/结果两个类型；判定逻辑本身在 decide_permission 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    # 只在标注里用到，但这里必须写成真 import 的对立面：本模块住在 tools 包里，而
    # tools/__init__ 迟早会 import 它（判定要接在派发那一步），模块层互相 import 就成环了。
    from mini_code_agent.tools import ToolContext


# 三个档位。参数标注、parse_permission_mode 的返回值、报错里列出的可选值都指向这一处，
# 以后想加第四档只改这里。
PermissionMode = Literal["accept edit", "default", "plan"]

PERMISSION_MODES: tuple[PermissionMode, ...] = get_args(PermissionMode)


# 会改磁盘的那几个工具。模式判定（哪一档管它们）和"该给用户看改动还是看参数"都按这一份
# 名单分——两处各写一份迟早会漂。
EDIT_TOOLS = ("write_file", "edit_file", "replace_lines")


@dataclass
class PermissionRequest:
    tool_name: str
    tool_args: dict


@dataclass
class PermissionDecision:
    decision: Literal["allow", "deny", "ask"]
    message:str

def parse_permission_mode(value: str) -> PermissionMode:
    """把用户敲的那一档解析成规范值。

    `accept edit` 中间有个空格，命令行上敲起来别扭，所以这几种写法都算同一个：
    `accept edit`、`accept_edit`、`accept-edit`、`acceptedit`——比的是"只留下字母数字、
    并转成小写"之后的样子，因此大小写和空格对数都不计较。

    看不懂的词直接报错，并把可选值列出来。**不比默认值兜底**：兜底的后果是用户以为
    切到了某一档、实际在另一档上跑，那种误解不该由默认值来消化。
    """
    if not isinstance(value, str):
        raise ValueError(f"权限模式得是字符串，收到的是 {type(value).__name__}：{value!r}")
    squashed = "".join(ch for ch in value.lower() if ch.isalnum())
    for mode in PERMISSION_MODES:
        if squashed == mode.replace(" ", ""):
            return mode
    options = "、".join(f"{mode!r}" for mode in PERMISSION_MODES)
    raise ValueError(f"{value!r} 不是权限模式。可选：{options}")


def decide_permission(
    request: PermissionRequest, ctx: ToolContext, permission_mode: str
) -> PermissionDecision:
    """判一次调用该不该放行：allow / deny / ask。

    - bash **不论哪一档都要问用户**。一条命令会干什么，用户事先看不出来；档位管的是
      "改文件那类动作用不用一个个问"，管不到它
    - 其余的工具一律放行——读文件、列目录、搜索本来就只受路径闸门管，逐个问没人受得了
    - 改文件的那几个（EDIT_TOOLS）看档位：`accept edit` 放行、`default` 问、`plan` 拒绝

    "ask" 不是答案，是"这事得问用户"；真的去问、以及用户说不行之后怎么办，在 harness 那层。

    ctx 现在用不上，但判定迟早要看上下文（比如这个文件读过没有、shell 有没有开），签名
    先留在这儿，免得以后每个调用点都要改一遍。
    """
    mode = parse_permission_mode(permission_mode)

    if request.tool_name == "bash":
        return PermissionDecision("ask", "跑命令要用户确认")
    if request.tool_name not in EDIT_TOOLS:
        return PermissionDecision("allow", "")
    if mode == "accept edit":
        return PermissionDecision("allow", "")
    if mode == "plan":
        return PermissionDecision("deny", "plan 模式只读，不给改文件")
    return PermissionDecision("ask", "改文件要用户确认")
