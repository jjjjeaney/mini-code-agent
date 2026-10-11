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

要单色输出就改成 `Console(soft_wrap=True, highlight=False, no_color=True)`——但注意那时
颜色全没了（实测 no_color 只保留粗体、斜体、dim），红色也不例外。
"""

from rich.console import Console

console = Console(soft_wrap=True, highlight=False)
