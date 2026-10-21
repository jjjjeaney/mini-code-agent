"""问用户这件事的接口：谁来问、问什么，以及没人可问的时候怎么办。

这里**只有协议**，没有一行终端代码——打印提示、读键盘、渲染 diff 都在 CLI 那层的
prompting.py 里，那一层才 import typer。

它不住在 tools 包里，问用户也不是工具的能力：确认是 harness 在调工具**之前**发的，以后
那个 ask_user 工具要问也是拿这个协议问。谁都只认这个协议，于是测试塞一个假实现就能把
"问用户"那条路径整个跑起来。

三种问法分两类：

- `confirm_tool` / `confirm_edit` 是**守卫**：策略判了"要问"才轮到它们。问不问由调用方
  （harness）决定，不由被问的东西决定——模型跳不过去。答案见 `Answer`：除了准 / 不准，
  改文件那一档还能答"本会话这个文件别再问了"
- `single_choice` 是**模型主动**想问用户：在若干选项里挑一个

"该不该问"不在这儿判（那是 decide_permission 的事）。这儿只管怎么问、拿到答案之后
怎么答。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # 只在标注里用到，就不真 import 了：两个模块同在上层，标注没必要结成运行时依赖。
    from mini_code_agent.permissions import PermissionRequest


class NobodyToAskError(Exception):
    """没人可问，而这件事**没有**缺省答案。

    交给 registry.dispatch 变成模型看得见的 tool_result——"用户不在"是该让模型知道的事，
    不是该让整个会话崩掉的事。
    """


@dataclass
class Answer:
    """用户对一次"要不要放行"的回答。

    remember 只在 allowed 为真时有意义：用户答"本会话这个别再问了"。记多久、记成什么粒度
    是调用方的事（见 permissions.py 的 Grants），answer 这边只负责把话说清楚。

    note 是用户补的那句话——拒绝的时候尤其有用（"改 bar.py 而不是这个"）。它会被拼进回给
    模型的结果里，比一句干巴巴的"用户拒绝了"有用得多。
    """

    allowed: bool
    remember: bool = False
    note: str = ""


class Prompter(Protocol):
    """把四个问题摆给用户的方式。

    实现者只管渲染和读答案：怎么把一条命令、一处改动、一组选项讲清楚，是它的活；
    "该不该问"不是。
    """

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        """让用户决定放不放行这次工具调用。

        **bash 也走这一条**：渲染该给的是那行命令本身、不是那串参数，但"命令藏在 args 的
        哪个键上"是实现自己的事，harness 一路递的是整个 `request`。所以没单为它开一个
        方法——开了也只是多一处要对齐的东西。

        bash 那一档**不给"别再问了"**：命令之间差异太大，记住其中一条没什么意义，而它又是
        唯一没有 diff 可看的调用，逐条点头是对的。
        """
        ...

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        """让用户看一眼改动，决定改不改。和上面那条分开，是因为这里该渲染的是 diff，
        不是参数本身。

        只有这一档提供 `remember`：改动是同一条路径上的重复劳动，用户说一句"这个文件
        别再问"很自然。
        """
        ...

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        """把问题摆给用户，让他在若干选项里挑一个，返回挑中的那一个。

        返回的是选项原文（用户自己敲了别的，就返回他敲的那个）。
        """
        ...


class NullPrompter:
    """没人可问的时候用它。非交互的场景——测试、没有注入 prompter 的调用——缺省就是它。

    守卫类的两个答**拒绝**：没有安全的"问"法，但有安全的默认答案。挡下来，模型收到
    一句"没人能批准这次调用"，换个不动文件的办法重来即可。这里绝不能等输入：等的是
    一个永远不会来的答案。

    `single_choice` 只能抛错：一道让用户做选择的问题没有缺省答案，替他挑一个就是替他
    做了决定。
    """

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        return Answer(False)

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        return Answer(False)

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        raise NobodyToAskError(
            f"没人可问，这个选择做不了：{question}（可选：{'、'.join(options)}）"
        )


class BackgroundPrompter:
    """没有 broker 时后台子 agent 用的兜底：**不碰终端，需要当场批准的一律拒绝**。

    交互会话里有 `interaction.PermissionBroker`，那边用 `BrokerPrompter` 把请求排队、由读键盘
    线程去问；这个类只剩没注入 broker 的调用方（主要是测试）在用了。

    为什么不是 `NullPrompter`：用户需要知道"这不是没人，是这时候问不了"。后台任务跑的
    时候主循环可能在提示符上、也可能在跑别的一轮，任何读键盘的动作都会和它抢 stdin。
    所以这里连问都不问，直接把"把改动写进结论"这条出路交给模型——主 agent 拿到结论后
    可以自己来做，或者下一轮再问用户。

    `Grants` 里本会话已经批过的文件走不到这儿（`_gate` 先查 `covers`），所以后台任务仍然
    能改用户已经放行的那些文件。
    """

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        return Answer(False, note=self._note(request))

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        return Answer(False, note=self._note(request))

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        raise NobodyToAskError(f"后台任务没法问用户，这个选择做不了：{question}")

    @staticmethod
    def _note(request: PermissionRequest) -> str:
        return (
            f"这是后台任务，没人能当场批准 {request.tool_name}。"
            "把要改什么、为什么写进结论，让主 agent 来做，或者等用户回来再派一次。"
        )
