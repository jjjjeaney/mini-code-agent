"""AnthropicProvider 冒烟测试。

这些用例会真实调用 DeepSeek 的 Anthropic 兼容接口，需要 .env 中配置好
DEEPSEEK_API_KEY 与 DEEPSEEK_ANTHROPIC_BASE_URL，否则整个用例集会被跳过。

运行方式（项目根目录下）：
    uv run pytest -v
"""

import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

from mini_code_agent.providers.anthropic import AnthropicProvider
from mini_code_agent.providers.base import ProviderError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class AnthropicProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未设置 DEEPSEEK_API_KEY，跳过真实接口调用")
        if not os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL"):
            raise unittest.SkipTest("未设置 DEEPSEEK_ANTHROPIC_BASE_URL，跳过真实接口调用")
        cls.provider = AnthropicProvider(max_tokens=512)

    def test_env_and_client_configuration(self):
        """构造器应该用 .env 里的 key / base_url，并保存 max_tokens。"""
        self.assertEqual(self.provider.max_tokens, 512)
        self.assertEqual(self.provider.model, "deepseek-v4-flash")
        self.assertEqual(
            str(self.provider.client.base_url).rstrip("/"),
            os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL").rstrip("/"),
        )

    def test_simple_completion(self):
        """最基础的调用：单轮 user 消息，应返回非空文本。"""
        response = self.provider.complete(
            messages=[{"role": "user", "content": "用一句话介绍你自己。"}],
        )
        print(f"\n[simple] stop_reason={response.stop_reason}")
        print(f"[simple] text={response.text!r}")

        self.assertTrue(response.text and response.text.strip(), "响应中没有任何文本内容")
        self.assertIsNone(response.tool_calls)
        self.assertIn(response.stop_reason, {"end_turn", "max_tokens", "stop_sequence"})

    def test_assistant_content_is_replayable(self):
        """assistant_content 要能原样回填进下一轮请求。"""
        response = self.provider.complete(
            messages=[{"role": "user", "content": "说一句「你好」。"}],
        )
        self.assertTrue(response.assistant_content)

        replay = [
            {"role": "user", "content": "说一句「你好」。"},
            {"role": "assistant", "content": response.assistant_content},
            {"role": "user", "content": "你刚才说了什么？"},
        ]
        follow_up = self.provider.complete(messages=replay)
        self.assertTrue(follow_up.text and follow_up.text.strip(), "回填后没有拿到回复")

    def test_system_prompt_is_applied(self):
        """带上 system 参数时，模型的输出应遵守该约束。"""
        response = self.provider.complete(
            messages=[{"role": "user", "content": "2 + 3 等于多少？"}],
            system="你是一个计算器。只输出阿拉伯数字，不要任何其他字符或标点。",
        )
        text = response.text or ""
        print(f"\n[system] text={text!r}")

        self.assertTrue(text.strip(), "响应中没有任何文本内容")
        self.assertIn("5", text)

    def test_tools_are_forwarded(self):
        """传入 tools 时请求不应报错，且模型应发起 tool_use。"""
        tools = [
            {
                "name": "get_weather",
                "description": "查询指定城市的当前天气",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名"}},
                    "required": ["city"],
                },
            }
        ]
        response = self.provider.complete(
            messages=[{"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"}],
            tools=tools,
        )
        print(f"\n[tools] stop_reason={response.stop_reason}")
        print(f"[tools] tool_calls={response.tool_calls}")

        self.assertTrue(response.tool_calls, "期望模型发起 tool_use，但没有拿到")
        call = response.tool_calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertIn("city", call.arguments)
        self.assertTrue(call.id, "tool_use 块应该带 id，否则没法回填 tool_result")

    def test_multi_turn_messages(self):
        """多轮对话应该被正确拼接并延续上下文。"""
        response = self.provider.complete(
            messages=[
                {"role": "user", "content": "记住一个数字：42。"},
                {"role": "assistant", "content": "好的，我记住了 42。"},
                {"role": "user", "content": "我刚才让你记的数字是多少？只回答数字。"},
            ],
        )
        text = response.text or ""
        print(f"\n[multi-turn] text={text!r}")

        self.assertIn("42", text)

    def test_invalid_credentials_raise(self):
        """错误凭证要抛**项目自己的**异常，带着一句能照着做的话。

        不是静默返回空响应，也不是把 anthropic 的类名甩给上层——翻译就在 provider 这一层做，
        所以上面（agent、CLI）永远不用认识后端的异常类名。
        """
        import anthropic

        bad = AnthropicProvider(max_tokens=32)
        bad.client = anthropic.Anthropic(api_key="sk-invalid-key", base_url=bad.client.base_url)
        with self.assertRaises(ProviderError) as caught:
            bad.complete(messages=[{"role": "user", "content": "hi"}])
        self.assertIn("DEEPSEEK_API_KEY", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
