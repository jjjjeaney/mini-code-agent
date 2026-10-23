"""命令执行的沙箱模式：docker（一次性容器）或 host（直接跑在宿主机上）。

这个模块只认"模式名字"，不依赖其它内部模块——CLI、命令、shell 工具、系统提示和权限
预览都从这里取同一份定义，避免五处各写一遍字符串。
"""

from __future__ import annotations

SANDBOXES: tuple[str, ...] = ("docker", "host")
DEFAULT_SANDBOX = "docker"

SANDBOX_GLOSS: dict[str, str] = {
    "docker": "一次性容器，跑完就删（默认）",
    "host": "直接跑在宿主机上，没有隔离",
}


def parse_sandbox(value: str) -> str:
    """把用户给的值解析成规范模式。

    大小写和首尾空白不计；只认 docker / host。**不兜底**：猜错的后果是用户以为切到了
    host，实际还在 docker 上跑，和权限模式那边是同一条理由。
    """
    if not isinstance(value, str):
        raise ValueError(f"沙箱模式得是字符串，收到的是 {type(value).__name__}：{value!r}")
    normalized = value.strip().lower()
    if normalized in SANDBOXES:
        return normalized
    options = "、".join(f"{name!r}" for name in SANDBOXES)
    raise ValueError(f"{value!r} 不是沙箱模式。可选：{options}")
