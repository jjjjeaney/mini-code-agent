"""工具结果在终端上打不打。

只有两种：`full` 完整打印每次工具调用和它的结果（默认，和以前一样）；`hidden` 只打调用，
结果一个字符都不打——没有摘要、没有字符数。`hidden` 下**出错的结果仍然完整打印**：把
失败原因藏起来，用户会以为工具跑成了。

**它只影响屏幕。** 发给模型的结果、落盘的文件、会话记录、折叠过的历史都原样不动；它也
不进系统提示，所以对缓存和花费没有任何影响。

和 sandbox.py 一样是个叶子模块：CLI、命令和 core 都从这里取，字符串不散落。
"""

RESULTS_MODES = ("full", "hidden")
DEFAULT_RESULTS = "full"

RESULTS_GLOSS: dict[str, str] = {
    "full": "完整打印工具调用的结果",
    "hidden": "只打工具调用，出错的结果仍会完整打印",
}


def parse_results(value: str) -> str:
    """归一化并校验。大小写和首尾空白不计，只认 `RESULTS_MODES`；未知值直接报错，不兜底。"""
    text = value.strip().lower()
    if text not in RESULTS_MODES:
        raise ValueError(f"{value!r} 不是结果显示模式，可选：{' / '.join(RESULTS_MODES)}")
    return text
