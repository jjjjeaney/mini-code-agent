"""LLM 后端的抽象接口，以及后端无关的响应类型。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


class ProviderError(Exception):
    """调模型这件事失败了，而且**这一轮已经没法继续**。

    provider 层负责把后端的异常翻译成这个——**消息写成人能照着做的话**（是 key 的问题、是网络、
    还是服务端），因为上游（agent、CLI）不认识 anthropic 那些类名，换后端时也不该改。

    **该不该重试看类型、不看消息文本**：可以再试的是 `TransientProviderError`，其余的都是
    "再试也没用"（401 再试一百次还是 401，只会把"去查 .env"这句话推迟几秒）。

    上层拿它要做的事就一件：把消息给用户看，然后把**这一轮**收掉——不是把会话收掉。
    """


class TransientProviderError(ProviderError):
    """**可以再试一次**的那种：连不上、超时、限流、对面 5xx。

    和父类的区别就是"该不该重试"，所以**类型就是那条信息**——别把它塞进消息文本里让上层去认字
    （认字会漂，类型不会）。

    这个集合是照抄 SDK 的 `_should_retry` 定的（408 / 409 / 429 / ≥500，加上连接类错误），
    **不要再发明第二套**：两套标准迟早对不上。还有一点要记住——走到这儿的，都是 **SDK 自己
    那 2 次退避重试也没救回来的**，所以上面只该再试很有限的一两次，而且等的时间要明显更长
    （比它那 0.5 + 1 秒长），否则就是把它干过的事再干一遍。
    """


@dataclass
class ModelResponse:
    """一次请求归一化之后的结果。

    assistant_content 是后端原始的 content blocks，续接对话时要原样回填，
    因为里面除了文字还有 tool_use（以及 DeepSeek 可能返回的 thinking）。
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str | None = None

    # 这一发的用量。默认 0 是为了让假 provider（测试里那一堆）不用管它，真后端一定会填。
    #
    # **输入那三个字段不重叠**：input_tokens 是没走缓存的那部分，cache_read 是命中的，
    # cache_creation 是往缓存里新写的——**三个加起来**才是这次请求的输入。这是 Anthropic 的
    # usage 语义。打的后端（DeepSeek）是自动前缀缓存：请求里不用传任何东西，命中多少只看
    # cache_read（所以 cache_creation 常态是 0，那一段是留给"哪天真的手动写缓存"的）。
    #
    # output_tokens 是生成的那部分。算钱的话它通常比谁都重——单价一般是输入的好几倍。
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0

    # **没有 server_tool_use**：SDK 里那不是 token，是"服务端工具跑了几次"
    # （ServerToolUsage 就两个字段：web_search_requests / web_fetch_requests）。而且这个后端
    # 不提供服务端工具——本项目的 web_search / web_fetch 是**客户端**工具，在 tools/web.py 里，
    # 它们花的 token 混在正常的输入输出里。哪天用上真 Anthropic 的 server tool 再说。


class Provider(Protocol):
    """agent 需要的后端能力：内部回合用 `complete`，前台终端回合用 `stream`。

    **两个方法的分工不是可选项**：用户能看见的那几个回合（交互、后台任务完成后的
    汇报）一律走 `stream`，没有"关掉流式"的开关；后台续跑、子 agent、压缩摘要不面向终端，
    走 `complete`——它们和前台共用终端，实时打字会互相插字。
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse: ...
