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

不带 prompt 就进交互模式，能连着聊：每条 prompt 接着上一条。以 `/` 开头的由**程序**处理、不发给模型——`/help` 列出有哪些（`/status`、`/sessions`、`/compact`、`/grants`、`/permission_mode`、`/exit`）。

```bash
uv run mca
```

会话按目录存在 `.agent/sessions/` 里。`--list` 列出有过哪些（连同各自的 id），`--resume <id>` 接上某一次；`--continue` 接最近一次**说过话**的。

工具输出太长时（`bash` 的命令输出、`web_fetch` 取回的网页），完整内容落在 `.agent/tool-results/` 里，上下文里只留一段预览——需要的时候模型用 `read_tool_result` 把它取回来。这样截断不再是不可逆的：长日志的**尾巴**（测试失败清单就在那儿）也留得住。

```bash
uv run mca --list
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
