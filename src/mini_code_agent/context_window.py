"""模型的上下文窗口：`模型 id → 能吃多少 token`。

**只按 id 精确匹配，不做前缀或模糊匹配。** 表里 `claude-sonnet-4-5-20250929` 是 200k 而
`claude-sonnet-5` 是 1M，差五倍；`gpt-4o` 和 `gpt-4o-mini` 一样只是巧合。按前缀猜会把这种
事猜错，而猜错的后果是阈值跟着错——那个数直接决定"什么时候压"。

这张表是给人维护的：加一个模型就往下面加一行，引擎不用动。
"""

from __future__ import annotations

# 表里没有的模型按这个算。
#
# **这是个偏大的默认值**（有意的）：宁可少压几次，也不要在窗口其实更大的模型上过早压掉
# ——压缩是有损的、不可逆的，而"没压"只是让历史长一点。
#
# 代价要知道：默认值**大于**真实窗口时，阈值也跟着大，自动压缩可能永远不触发，而请求超窗是
# **硬报错**（后端 400，那一轮直接收掉），不像压缩那样只是丢点细节。真碰上了，把那个模型的
# id 加进表里就完事。
UNKNOWN_MODEL_WINDOW = 400_000

CONTEXT_WINDOWS: dict[str, int] = {
    # DeepSeek —— 这里和别的两处不同：项目里 `providers/anthropic.py` 的 DEFAULT_MODEL 用的是
    # **旧名** `deepseek-v4-flash`，而端点现在主推的名字是 `deepseek-flash`（旧名照收，账单
    # 一样）。两个都写上：漏了旧名的话，这个项目自己会在 400k 那个默认档上跑。
    "deepseek-flash": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    # OpenAI
    "gpt-6-astra": 1_048_576,
    "gpt-6.1-sol": 1_048_576,
    "gpt-6-luna": 1_048_576,
    "gpt-5.5": 1_050_000,
    "gpt-5.5-pro": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.3-codex": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.2-codex": 400_000,
    "gpt-5.2-pro": 400_000,
    "gpt-5": 400_000,
    "gpt-5-pro": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3": 200_000,
    "o3-pro": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "o3-deep-research": 200_000,
    "o4-mini-deep-research": 200_000,
    # Anthropic
    "claude-opus-5-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-5-5": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-fable-5-1": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-mythos-5-1": 1_000_000,
    "claude-mythos-5": 1_000_000,
    # Google
    "gemini-4-argon": 1_048_576,
    "gemini-3.8-flash": 1_048_576,
    "gemini-3.6-flash": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.5-flash-lite": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    # 通义千问
    "qwen3.7-plus": 1_000_000,
    "qwen3.6-plus": 1_000_000,
    "qwen3.5-plus": 1_000_000,
    "qwen3.5-flash": 1_000_000,
    "qwen3-max-2026-01-23": 262_144,
    "qwen3-coder-plus": 1_000_000,
    "qwen3-coder-next": 262_144,
    "qwen-plus": 1_000_000,
    "qwen-max": 32_768,
    "qwen-long": 10_000_000,
    "qwen-coder-plus": 131_072,
    # 智谱 / 月之暗面
    "glm-5.2": 1_048_576,
    "glm-5.1": 204_800,
    "glm-5": 204_800,
    "glm-4.7": 204_800,
    "glm-4.6": 204_800,
    "glm-4.5": 131_072,
    "kimi-k3": 1_048_576,
    "kimi-latest": 1_048_576,
    "kimi-k2.7-code": 262_144,
    "kimi-k2.6": 262_144,
    "kimi-k2.5": 262_144,
}


def window_for(model: str | None) -> int:
    """这个模型的上下文窗口（token）。表里没有就用 UNKNOWN_MODEL_WINDOW。

    `model` 可以是 None（`_chat` 的默认值，测试里就会走到）——那也当作"不在表里"。
    """
    if not model:
        return UNKNOWN_MODEL_WINDOW
    return CONTEXT_WINDOWS.get(model, UNKNOWN_MODEL_WINDOW)
