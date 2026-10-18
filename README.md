# mini-code-agent

一个最小可用的 coding agent，走 Anthropic Messages API（默认指向 DeepSeek 的兼容端点）。

## 准备环境

```bash
uv sync
```

## 配置

复制 `.env.example` 为 `.env`，填入自己的凭证：

```bash
cp .env.example .env
```

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 接口密钥 |
| `DEEPSEEK_ANTHROPIC_BASE_URL` | Anthropic 兼容端点地址 |
| `TAVILY_API_KEY` | Tavily 联网密钥，`web_search` 和 `web_fetch` 两个工具用；不配的话这两个都不可用 |

## 使用

```bash
uv run mca "用一句话解释什么是闭包"
```

`--model` 换模型；不写就用 `providers/anthropic.py` 里的 `DEFAULT_MODEL`（`deepseek-v4-flash`）。上下文窗口和压缩阈值会跟着模型名走。

```bash
uv run mca --model deepseek-v4-pro "用一句话解释什么是闭包"
```

`--api-key` / `--base-url` 可以临时覆盖 `.env` 里的 `DEEPSEEK_API_KEY` / `DEEPSEEK_ANTHROPIC_BASE_URL`。**密钥写在命令行上会留在 shell 历史和进程列表里**，日常还是用 `.env`；这两个参数主要给脚本或临时切换端点用。

不带 prompt 就进交互模式，能连着聊：每条 prompt 接着上一条。以 `/` 开头的由**程序**处理、不发给模型——`/help` 列出有哪些（`/status`、`/sessions`、`/tasks`、`/compact`、`/grants`、`/permission_mode`、`/fork`、`/exit`）。`/fork` 是 `--fork` 那个开关的运行时版本：`/fork on` / `/fork off`，下一轮起生效（它会换掉提示词里那段说明，也就是重建前缀缓存）。

```bash
uv run mca
```

会话按目录存在 `.agent/sessions/` 里。`--list` 列出有过哪些（连同各自的 id），`--resume <id>` 接上某一次；`--continue` 接最近一次**说过话**的。

工具输出太长时（`bash` 的命令输出、`web_fetch` 取回的网页），完整内容落在 `.agent/tool-results/<会话 id>/` 里，上下文里只留一段预览——需要的时候模型用 `read_tool_result` 把它取回来。这样截断不再是不可逆的：长日志的**尾巴**（测试失败清单就在那儿）也留得住。按会话分目录是为了让 `--delete` 能把它们连同会话一起删掉（删之前会告诉你一共有几个）。缓存冷了之后折叠旧工具结果时，重跑拿不回来的那几类（`bash`、`web_fetch`、`web_search`）也会先按工具调用 id 存一份进来，同样用 `read_tool_result` 取。

`--fork` 打开 `agent` 的 `fork` 类型：`agent(type="fork", ...)` 继承当前这份对话（同一份系统提示、同一张工具表、除最后一条工具调用之外的全部消息），所以背景不用重讲，缓存也直接命中。适合"必须带着现在这段上下文才做得对"的事——按现有方向独立试另一种做法，或者复用已经读过的文件。反过来，一段自足的任务描述就能说清的，用 `general-purpose` 更划算。它和其它类型同级，也能 `run_in_background`；它改过的文件，主 agent 之后改之前会先被要求重读。

同一批里如果有多个**相邻的只读工具调用**（读文件、`grep`、联网取页这些），它们会并行执行，最多同时 4 个。写文件、跑命令、终端输出和子 agent 是屏障：被它们隔开的只读调用不会跨过去并行，结果的顺序始终和模型给出的顺序一致。

子 agent 默认也是同步的，但**跑满两分钟会自动转到后台**：工具调用立刻返回一个任务 id，子 agent 在后台接着跑，主 agent 可以去干别的。模型也可以直接要 `run_in_background=true`，一开始就不等。结果会在主 agent 下一发请求之前以一条通知自动进上下文（太长就按任务 id 落盘，用 `read_tool_result` 取）；如果主 agent 已经闲着，完成事件会叫醒循环，让模型自己开口告诉你。后台任务需要新的权限批准时会把请求排进队列：读键盘线程在你回到提示符时问（你正停在 `>` 上时，下一次回车之后才问）。`ask_user` 那道选择题后台问不了，要拍板的事会写进结论。`/tasks` 能看在跑什么、跑了多久、几个在等授权，也能取消；进程退出时后台任务会被硬杀。

```bash
uv run mca --list
```

`--delete <id>` 删掉某一次。删之前会先把要删的那次摆出来（多少条消息、最后说了什么）问一遍——会话文件是那次对话唯一的记录，没有回收站也没有备份，所以默认要问；脚本里用 `--yes` 跳过。

```bash
uv run mca --delete 20261014-153000-0001a3f2
```

## 测试

`tests/test_anthropic_provider.py` 是真实接口冒烟测试，会实际发起请求并产生费用；`.env` 里缺少任一变量时整组跳过。`tests/test_agent.py` 和 `tests/test_tools.py` 用假 provider 驱动，完全不联网。

```bash
uv run pytest
```

## 目录结构

```
src/mini_code_agent/
├── cli.py          # 命令行入口（typer）
├── core/           # agent 主循环与对话状态
├── providers/      # LLM 后端适配层，base.py 定义 Protocol
├── tools/          # 工具协议与实现
├── memory/         # 跨会话长期记忆
├── skills/         # 技能：一份一份"照着做"的步骤
├── sessions/       # 单次会话的 transcript 持久化
└── commands/       # 斜杠命令（CLI 表现层）
```

依赖方向自上而下：`cli` / `commands` → `core` → `tools` → `providers`，没有反向依赖。
