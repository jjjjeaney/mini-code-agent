"""问用户这件事的接口：谁来问、问什么，以及没人可问的时候怎么办。

这里**只有协议**，没有一行终端代码——打印提示、读键盘、渲染 diff 都在 CLI 那层的
prompting.py 里，那一层才 import typer。

它不住在 tools 包里，问用户也不是工具的能力：确认是 harness 在调工具**之前**发的，以后
那个 ask_user 工具要问也是拿这个协议问。谁都只认这个协议，于是测试塞一个假实现就能把
"问用户"那条路径整个跑起来。

四种问法分两类：

- `confirm_command` / `confirm_tool` / `confirm_edit` 是**守卫**：策略判了"要问"才轮到
  它们，答案只有是/否。问不问由调用方（harness）决定，不由被问的东西决定——模型跳不过去
- `single_choice` 是**模型主动**想问用户：在若干选项里挑一个

"该不该问"不在这儿判（那是 decide_permission 的事）。这儿只管怎么问、拿到答案之后
怎么答。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # 只在标注里用到，就不真 import 了：两个模块同在上层，标注没必要结成运行时依赖。
    from mini_code_agent.permissions import PermissionRequest


class NobodyToAskError(Exception):
    """没人可问，而这件事**没有**缺省答案。

    交给 registry.dispatch 变成模型看得见的 tool_result——"用户不在"是该让模型知道的事，
    不是该让整个会话崩掉的事。
    """


class Prompter(Protocol):
    """把四个问题摆给用户的方式。

    实现者只管渲染和读答案：怎么把一条命令、一处改动、一组选项讲清楚，是它的活；
    "该不该问"不是。
    """

    def confirm_command(self, command: str) -> bool:
        """让用户决定跑不跑这条命令。给的就是那条命令本身。"""
        ...

    def confirm_tool(self, request: PermissionRequest) -> bool:
        """让用户决定放不放行这次工具调用。不认得类别的那些走这一条。"""
        ...

    def confirm_edit(self, request: PermissionRequest) -> bool:
        """让用户看一眼改动，决定改不改。和上面那条分开，是因为这里该渲染的是 diff，
        不是参数本身。"""
        ...

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        """把问题摆给用户，让他在若干选项里挑一个，返回挑中的那一个。

        返回的是选项原文（用户自己敲了别的，就返回他敲的那个）。
        """
        ...


class NullPrompter:
    """没人可问的时候用它。非交互的场景——测试、一次性跑一条指令——缺省就是它。

    守卫类的三个答**拒绝**：没有安全的"问"法，但有安全的默认答案。挡下来，模型收到
    一句"没人能批准这次调用"，换个不动文件的办法重来即可。这里绝不能等输入：等的是
    一个永远不会来的答案。

    `single_choice` 只能抛错：一道让用户做选择的问题没有缺省答案，替他挑一个就是替他
    做了决定。
    """

    def confirm_command(self, command: str) -> bool:
        return False

    def confirm_tool(self, request: PermissionRequest) -> bool:
        return False

    def confirm_edit(self, request: PermissionRequest) -> bool:
        return False

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        raise NobodyToAskError(
            f"没人可问，这个选择做不了：{question}（可选：{'、'.join(options)}）"
        )
