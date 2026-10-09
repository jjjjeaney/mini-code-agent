"""工具的格式定义。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReadFileState:
    # path -> (mtime_ns, char_count)。Day 4 的 read-before-edit 要靠它判断
    # "模型读过这个文件之后，文件在磁盘上是不是又被改过"。今天先只做记录。
    entries: dict[Path, tuple[int, int]] = field(default_factory=dict)

    def record(self, path: Path, content: str, mtime_ns: int) -> None:
        # mtime_ns 由调用方在**读之前**量好传进来，这里不再自己去 stat。
        # 让这里 stat 的话，取到的是"记录那一刻"的时间戳，而 content 属于更早的版本——
        # 两者一错开，读-before-改的守卫就会以为模型看到的就是磁盘上最新的东西，
        # 而它要防的恰恰就是这一类错位。
        self.entries[path] = (mtime_ns, len(content))


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


@dataclass
class ToolContext:
    """调用工具时的运行环境。

    工具需要的外部信息都从这里拿，而不是去读全局状态——比如 cwd，
    路径校验就拿它当围栏的边界。

    cwd 是必填的：让工具在缺省时回落到进程当前目录，等于悄悄换了沙箱的边界。
    """

    cwd: Path
    read_state: ReadFileState=field(default_factory=ReadFileState)


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
