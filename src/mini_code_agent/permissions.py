"""权限模式：用户指定的那一档，以及判定要用的两个类型。

三种模式管的都是"改文件的工具要不要一个个问"：

- `accept edit`：默认同意。模型改文件不再逐个确认——给"我知道它在改什么、别老打断我"
  的时候用
- `default`：要用户确认。每次改文件都停下来问一句
- `plan`：默认拒绝。计划模式——只读，先让模型把打算怎么做讲清楚，再切出去执行

bash 不在这三档里：**不论什么模式都要用户确认**（除非压根没开 shell——那时先弹一句询问、
答完才发现跑不了，是白问）。一条命令会干什么，用户事先看不出来（沙箱挡的是它碰不到
宿主机，不是它做的事对不对），所以它不跟着模式走。

这些是**默认**答案：模式定的是"不问的时候怎么办"，具体到某一次调用还可能被别的条件
改成要问（比如路径在不在围栏里）。这个模块管两件事：把用户敲的档位解析成规范值，
以及判定要用的请求/结果两个类型；判定逻辑本身在 decide_permission 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

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
    request: PermissionRequest, allow_shell: bool, permission_mode: str
) -> PermissionDecision:
    """判一次调用该不该放行：allow / deny / ask。

    - bash 得开了 shell 才谈得上问。没开的话它本来就会报"没开命令执行"，没必要先弹一句
      询问、再让人发现跑不了——直接放行，把那句报错留给工具自己去说
    - bash **不论哪一档都要问用户**。一条命令会干什么，用户事先看不出来；档位管的是
      "改文件那类动作用不用一个个问"，管不到它
    - 其余的工具一律放行——读文件、列目录、搜索本来就只受路径闸门管，逐个问没人受得了
    - 改文件的那几个（EDIT_TOOLS）看档位：`accept edit` 放行、`default` 问、`plan` 拒绝

    "ask" 不是答案，是"这事得问用户"；真的去问、以及用户说不行之后怎么办，在 harness 那层。

    参数都是权限自己的东西（工具名、shell 开没开、档位），不认识 ToolContext 那种工具层的
    环境对象——权限不是工具的属性。
    """
    mode = parse_permission_mode(permission_mode)

    if request.tool_name == "bash":
        if not allow_shell:
            return PermissionDecision("allow", "")
        return PermissionDecision("ask", "跑命令要用户确认")
    if request.tool_name not in EDIT_TOOLS:
        return PermissionDecision("allow", "")
    if mode == "accept edit":
        return PermissionDecision("allow", "")
    if mode == "plan":
        return PermissionDecision("deny", "plan 模式只读，不给改文件")
    return PermissionDecision("ask", "改文件要用户确认")
