"""终端输出的那个 Console，全项目共用一份。

几处配置是按踩过的坑定的：

- **soft_wrap=True**：不按终端宽度折行。工具参数和结果里常有整行 JSON、很长的路径，
  rich 默认会折，折完就分不清哪是内容本身、哪是换行。
- **highlight=False**：关掉 rich 自动给数字、字符串上色那个功能。这种日志式的输出里它只是噪音。
- 打印**动态内容**（工具参数、结果、用户那句话）时要带 `markup=False`。那些字符串里到处
  是方括号——JSON 数组 `[1, 2]`、代码里的下标、用户自己打的——rich 会把它们当样式标记：
  实测 `out: [1, 2] end` 会被渲染成彩色数字，`text has [red] inside` 里的 `[red]` 直接
  消失。`[bold]` 那种方括号写法留给**固定文案**用（比如标题），动态内容一律走 style= 参数。
- 输出不是终端时（`mca ... > out.txt`）rich 自己会把颜色去掉，不用自己判 TTY。

- **非 UTF-8 的输出流在启动时切成 UTF-8**（见 `_ensure_utf8`）。Windows 上重定向输出时
  Python 的 stdout 用的是系统 ANSI 代码页（常见是 GBK），而输出里全是 `▶` / `→` / `✗`——
  rich 的 legacy Windows 渲染器往 GBK 流里写会直接抛 UnicodeEncodeError。控制台交互时
  Python 走的是 Unicode 控制台 API，本来没事；出问题的就是管道/文件/旧控制台那条路。

要单色输出就改成 `Console(soft_wrap=True, highlight=False, no_color=True)`——但注意那时
颜色全没了（实测 no_color 只保留粗体、斜体、dim），红色也不例外。
"""

import sys

from rich.console import Console


def _ensure_utf8(stream) -> None:
    """把输出流从本地编码切到 UTF-8。改不动就算了——不能因为一行日志把程序带崩。

    `errors="replace"` 是兜底：万一以后有 UTF-8 也编码不了的东西，宁可显示一个替换字符，
    也不要抛异常。
    """
    if stream is None:
        return
    encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if not encoding or encoding in ("utf8", "cp65001"):
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


_ensure_utf8(sys.stdout)
_ensure_utf8(sys.stderr)

console = Console(soft_wrap=True, highlight=False)
