from __future__ import annotations

"""Anthropic 端点与 OpenAI 共享的 entml 流式/非流式 tool 解析一致性测试。"""

import json
import unittest

from echotools import FncallStreamParser, get_protocol

from handlers.anthro import (
    _anthropic_event_bytes,
    _build_anthropic_protocol_options,
    _message_delta_event,
    _message_start_event,
    _message_stop_event,
    _normalize_anthropic_messages,
    _normalize_anthropic_tools,
    _tool_use_block_events,
)
from handlers.openai import _parse_tool_calls, protocol_thinking_level
from server.formats import convert_to_anthropic
from state import AppState


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SAMPLE_RESPONSE = (
    "好的，我来查天气。\n"
    '<entml:invoke name="get_weather">\n'
    '<entml:parameter name="city">上海</entml:parameter>\n'
    '<entml:parameter name="unit">c</entml:parameter>\n'
    "</entml:invoke>"
)

THINKING_RESPONSE = (
    "<entml:thinking>\n需要查上海天气\n</entml:thinking>\n"
    "马上查询。\n"
    '<entml:invoke name="get_weather">\n'
    '<entml:parameter name="city">上海</entml:parameter>\n'
    "</entml:invoke>"
)


def _collect_stream_tool_calls(text: str, *, chunk_size: int = 3) -> list[dict]:
    """模拟 handler 流式路径：feed 返回值 + finalize 后 get_ready。"""
    protocol = get_protocol("entml")
    parser = FncallStreamParser(protocol=protocol, tools=TOOLS)
    emitted: list[dict] = []
    for i in range(0, len(text), chunk_size):
        emitted.extend(parser.feed(text[i : i + chunk_size]))
    parser.finalize()
    emitted.extend(parser.get_ready_tool_calls())
    return emitted


class TestAnthropicToolStream(unittest.TestCase):
    def setUp(self) -> None:
        self.state = AppState.__new__(AppState)
        self.state.protocol = get_protocol("entml")

    def test_feed_return_not_double_consumed(self) -> None:
        """feed() 返回的 ready 不应再被 get_ready_tool_calls 重复消费。"""
        protocol = get_protocol("entml")
        parser = FncallStreamParser(protocol=protocol, tools=TOOLS)
        ready = parser.feed(SAMPLE_RESPONSE)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]["function"]["name"], "get_weather")
        self.assertEqual(parser.get_ready_tool_calls(), [])

    def test_stream_incremental_matches_batch_parse(self) -> None:
        stream_calls = _collect_stream_tool_calls(SAMPLE_RESPONSE, chunk_size=5)
        _, batch_calls = _parse_tool_calls(self.state, SAMPLE_RESPONSE, TOOLS)
        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(
            stream_calls[0]["function"]["name"],
            batch_calls[0]["function"]["name"],
        )
        self.assertEqual(
            json.loads(stream_calls[0]["function"]["arguments"]),
            json.loads(batch_calls[0]["function"]["arguments"]),
        )

    def test_stream_invoke_emits_input_json_delta_incrementally(self) -> None:
        protocol = get_protocol("entml")
        parser = FncallStreamParser(protocol=protocol, tools=TOOLS)
        deltas: list[tuple[str, str]] = []
        text = (
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">上海</entml:parameter>\n'
            "</entml:invoke>"
        )
        for i in range(0, len(text), 4):
            parser.feed(text[i : i + 4])
            chunk = parser.consume_stream_delta()
            if chunk:
                deltas.append(chunk)
        self.assertTrue(deltas)
        names = {d[0] for d in deltas}
        self.assertEqual(names, {"get_weather"})
        merged = "".join(d[1] for d in deltas)
        self.assertIn("上海", merged)

    def test_post_stream_remaining_after_streamed_tool(self) -> None:
        """input_json_delta 已发完时，post_stream 不应再重复整段 tool_use。"""
        all_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
            }
        ]
        pending_tc_count = 1
        remaining = all_tool_calls[pending_tc_count:]
        self.assertEqual(remaining, [])
        protocol = get_protocol("entml")
        parser = FncallStreamParser(protocol=protocol, tools=TOOLS)
        emitted: list[dict] = []
        for i in range(0, len(THINKING_RESPONSE), 4):
            emitted.extend(parser.feed(THINKING_RESPONSE[i : i + 4]))
        self.assertIn("需要查上海天气", parser.partial_thinking)
        parser.finalize()
        emitted.extend(parser.get_ready_tool_calls())
        self.assertEqual(len(emitted), 1)
        clean = parser.partial_text
        self.assertIn("马上查询", clean)
        self.assertNotIn("entml:invoke", clean)

    def test_convert_to_anthropic_tool_use(self) -> None:
        _, tool_calls = _parse_tool_calls(self.state, SAMPLE_RESPONSE, TOOLS)
        openai_resp = {
            "id": "msg_test",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "好的，我来查天气。",
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        anth = convert_to_anthropic(openai_resp)
        self.assertEqual(anth["stop_reason"], "tool_use")
        blocks = anth["content"]
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertTrue(tool_blocks[0]["id"].startswith("toolu_"))
        self.assertEqual(tool_blocks[0]["name"], "get_weather")
        self.assertEqual(tool_blocks[0]["input"]["city"], "上海")

    def test_normalize_anthropic_messages_roundtrip(self) -> None:
        anth_msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": "你好"}],
        }, {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "嗯"},
                {"type": "text", "text": "你好呀"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "search_web",
                    "input": {"query": "test"},
                },
            ],
        }]
        normalized = _normalize_anthropic_messages(anth_msgs)
        self.assertEqual(normalized[-1]["role"], "assistant")
        self.assertEqual(normalized[-1]["reasoning"], "嗯")
        self.assertEqual(len(normalized[-1]["tool_calls"]), 1)
        self.assertEqual(
            normalized[-1]["tool_calls"][0]["function"]["name"],
            "search_web",
        )

    def test_normalize_anthropic_tools(self) -> None:
        anth_tools = [{
            "name": "get_weather",
            "description": "weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }]
        converted = _normalize_anthropic_tools(anth_tools)
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "get_weather")


class TestAnthropicProtocolOptions(unittest.TestCase):
    def test_default_effort_high(self) -> None:
        opts = _build_anthropic_protocol_options({})
        self.assertEqual(protocol_thinking_level(opts), "high")

    def test_output_config_effort(self) -> None:
        opts = _build_anthropic_protocol_options({"output_config": {"effort": "xhigh"}})
        self.assertEqual(opts["thinking_level"], "xhigh")

    def test_thinking_disabled(self) -> None:
        opts = _build_anthropic_protocol_options({
            "output_config": {"effort": "max"},
            "thinking": {"type": "disabled"},
        })
        self.assertEqual(opts["thinking_level"], "none")

    def test_thinking_adaptive_uses_effort(self) -> None:
        opts = _build_anthropic_protocol_options({
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        })
        self.assertEqual(opts["thinking_level"], "low")

    def test_thinking_enabled_budget_and_effort(self) -> None:
        opts = _build_anthropic_protocol_options({
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {"effort": "high"},
        })
        self.assertEqual(opts["thinking_level"], "high")
        self.assertEqual(opts["max_thinking_length"], 8000)

    def test_ignores_openai_reasoning_effort(self) -> None:
        opts = _build_anthropic_protocol_options({"reasoning_effort": "xhigh"})
        self.assertEqual(opts["thinking_level"], "high")

    def test_invalid_effort_raises(self) -> None:
        with self.assertRaises(ValueError):
            _build_anthropic_protocol_options({"output_config": {"effort": "turbo"}})

    def test_invalid_thinking_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            _build_anthropic_protocol_options({"thinking": {"type": "on"}})


class TestAnthropicStreamFormat(unittest.TestCase):
    """SSE 事件结构与 mock.py AnthropicBuilder 对齐。"""

    def test_message_start_shape(self) -> None:
        ev = _message_start_event("claude-test", "msg-1")
        self.assertEqual(ev["type"], "message_start")
        self.assertEqual(ev["message"]["role"], "assistant")
        self.assertEqual(ev["message"]["content"], [])

    def test_tool_use_block_events_match_mock(self) -> None:
        events = _tool_use_block_events(1, "toolu_abc123", "Glob", {"pattern": "**/*.json", "path": "/project"})
        self.assertEqual(events[0]["type"], "content_block_start")
        self.assertEqual(events[0]["index"], 1)
        block = events[0]["content_block"]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "toolu_abc123")
        self.assertEqual(block["name"], "Glob")
        self.assertEqual(block["input"], {})

        deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.assertTrue(deltas)
        for d in deltas:
            self.assertEqual(d["index"], 1)
            self.assertEqual(d["delta"]["type"], "input_json_delta")
        merged = "".join(d["delta"]["partial_json"] for d in deltas)
        self.assertEqual(json.loads(merged), {"pattern": "**/*.json", "path": "/project"})

        self.assertEqual(events[-1], {"type": "content_block_stop", "index": 1})

    def test_sse_bytes_event_line(self) -> None:
        ev = _message_stop_event()
        raw = _anthropic_event_bytes(ev).decode("utf-8")
        self.assertTrue(raw.startswith("event: message_stop\n"))
        self.assertIn('"type": "message_stop"', raw)

    def test_finish_events(self) -> None:
        delta = _message_delta_event("tool_use")
        self.assertEqual(delta["delta"]["stop_reason"], "tool_use")
        self.assertEqual(delta["delta"]["stop_sequence"], None)
        self.assertIn("output_tokens", delta["usage"])

    def test_finish_stop_reason_when_only_streamed_count(self) -> None:
        """已流式发出 tool_use 但 finalize 无列表时，仍应 tool_use（对齐 mock 收尾）。"""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from handlers.anthro import _send_anthropic_finish

        resp = MagicMock()
        resp.write = AsyncMock()
        disconnected = [False]

        async def run() -> None:
            await _send_anthropic_finish(
                resp, [], disconnected, streamed_tool_count=2,
            )

        asyncio.run(run())
        written = b"".join(c.args[0] for c in resp.write.call_args_list)
        self.assertIn(b'"stop_reason": "tool_use"', written)
        self.assertIn(b"message_stop", written)


if __name__ == "__main__":
    unittest.main()
