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

## 使用

```bash
uv run mini-code-agent "用一句话解释什么是闭包"
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
├── sessions/       # 单次会话的 transcript 持久化
└── commands/       # 斜杠命令（CLI 表现层）
```

依赖方向自上而下：`cli` / `commands` → `core` → `tools` → `providers`，没有反向依赖。
