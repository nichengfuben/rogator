from __future__ import annotations

import unittest

from upstream.zen.openai_chat import build_chat_payload, normalize_model_name, normalize_tools
from upstream.zen.chat_stream import SseLineAssembler, has_valid_sse_event, parse_openai_sse_data
from upstream.zen.proxy import (
    load_dynamic_proxy_pool,
    load_static_pool_from_config,
    merge_proxy_pools,
    normalize_proxy_url,
)


class TestZenPayload(unittest.TestCase):
    def test_normalize_model_strips_local(self) -> None:
        self.assertEqual(normalize_model_name("mimo-v2.5-free-local"), "mimo-v2.5-free")

    def test_normalize_tools_openai_and_anthropic(self) -> None:
        tools = normalize_tools([
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"name": "b", "input_schema": {"type": "object"}},
        ])
        assert tools is not None
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["function"]["name"], "a")
        self.assertEqual(tools[1]["function"]["name"], "b")

    def test_build_payload_thinking(self) -> None:
        payload = build_chat_payload(
            [{"role": "user", "content": "hi"}],
            "mimo-v2.5-free",
            thinking=True,
        )
        self.assertTrue(payload["thinking"])
        self.assertTrue(payload["stream"])


class TestZenSse(unittest.TestCase):
    def test_parse_answer_and_thinking(self) -> None:
        events = parse_openai_sse_data(
            '{"choices":[{"delta":{"reasoning_content":"t","content":"a"}}]}'
        )
        types = [e["type"] for e in events]
        self.assertEqual(types, ["thinking", "answer"])
        self.assertEqual(events[0]["content"], "t")
        self.assertEqual(events[1]["content"], "a")

    def test_parse_usage(self) -> None:
        events = parse_openai_sse_data(
            '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}'
        )
        self.assertEqual(events[0]["type"], "usage")
        self.assertEqual(events[0]["data"]["prompt_tokens"], 1)

    def test_assembler_multiline_data(self) -> None:
        asm = SseLineAssembler()
        self.assertIsNone(asm.feed_line("data: {\"a\":1}"))
        payload = asm.feed_line("")
        self.assertEqual(payload, '{"a":1}')

    def test_has_valid_sse_event(self) -> None:
        self.assertTrue(has_valid_sse_event(b"data: {\"x\":1}\n\n"))
        self.assertFalse(has_valid_sse_event(b"data: [DONE]\n\n"))


class TestZenModule(unittest.TestCase):
    def test_create_client_and_caps(self) -> None:
        from upstream import zen

        self.assertEqual(zen.NAME, "zen")
        self.assertTrue(zen.CAPABILITIES.get("chat"))
        client = zen.create_client(None)
        models = client.load_models_cache()
        self.assertIn("mimo-v2.5-free", models)
        self.assertGreaterEqual(client.node_manager.pool_size, 1)


class TestZenProxy(unittest.TestCase):
    def test_normalize_proxy(self) -> None:
        self.assertIsNone(normalize_proxy_url(""))
        self.assertIsNone(normalize_proxy_url("direct"))
        self.assertEqual(normalize_proxy_url("127.0.0.1:7890"), "http://127.0.0.1:7890")
        self.assertEqual(
            normalize_proxy_url("http://1.2.3.4:8080"), "http://1.2.3.4:8080",
        )

    def test_merge_static_before_dynamic(self) -> None:
        static = load_static_pool_from_config(["", "127.0.0.1:7890"])
        merged = merge_proxy_pools(static, ["http://127.0.0.1:7890", "http://9.9.9.9:1"])
        self.assertEqual(merged[0], None)
        self.assertEqual(merged[1], "http://127.0.0.1:7890")
        self.assertEqual(merged[2], "http://9.9.9.9:1")

    def test_load_dynamic_missing_file(self) -> None:
        self.assertEqual(load_dynamic_proxy_pool("nonexistent_proxy_pool.json"), [])


if __name__ == "__main__":
    unittest.main()
