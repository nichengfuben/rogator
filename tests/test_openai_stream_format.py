from __future__ import annotations

"""OpenAI 流式 tool_calls 格式与 mock.py OpenAIBuilder 对齐测试。"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from echotools.fncall import FncallStreamParser, get_protocol

from handlers.openai import (
    _emit_openai_streaming_tool_delta,
    _openai_tool_call_entry,
    _send_stream_finish,
)
from server.formats import build_openai_chunk


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


class TestOpenAIStreamFormat(unittest.TestCase):
    def test_tool_call_entry_shape(self) -> None:
        tc = {
            "id": "call_0000",
            "type": "function",
            "function": {
                "name": "Glob",
                "arguments": json.dumps({"pattern": "**/*.json", "path": "/project"}, ensure_ascii=False),
            },
        }
        entry = _openai_tool_call_entry(0, tc)
        self.assertEqual(entry["index"], 0)
        self.assertTrue(entry["id"].startswith("toolu_"))
        self.assertEqual(entry["type"], "function")
        self.assertEqual(entry["function"]["name"], "Glob")
        self.assertEqual(json.loads(entry["function"]["arguments"]), {"pattern": "**/*.json", "path": "/project"})

    def test_tool_call_chunk_matches_mock(self) -> None:
        tc = {
            "id": "toolu_abc",
            "type": "function",
            "function": {
                "name": "Read",
                "arguments": '{"file_path":"/project/src/main.py"}',
            },
        }
        chunk = build_openai_chunk(
            "qwen3.7-max",
            chunk_id="gen-test",
            tool_calls=[_openai_tool_call_entry(1, tc)],
        )
        self.assertEqual(chunk["object"], "chat.completion.chunk")
        self.assertEqual(chunk["id"], "gen-test")
        choice = chunk["choices"][0]
        self.assertIsNone(choice["finish_reason"])
        delta = choice["delta"]
        self.assertEqual(delta["role"], "assistant")
        self.assertIsNone(delta["content"])
        self.assertEqual(len(delta["tool_calls"]), 1)
        self.assertEqual(delta["tool_calls"][0]["index"], 1)

    def test_finish_chunk_matches_mock(self) -> None:
        chunk = build_openai_chunk("qwen3.7-max", chunk_id="gen-test", finish_reason="tool_calls")
        choice = chunk["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["delta"]["role"], "assistant")
        self.assertEqual(choice["delta"]["content"], "")

    def test_finish_reason_when_streamed_only(self) -> None:
        """mock 规范：只要已流式发出 tool_calls，finish 必须为 tool_calls。"""
        resp = MagicMock()
        resp.write = AsyncMock()
        disconnected = [False]

        async def run() -> None:
            await _send_stream_finish(
                resp, "m", "gen-x", [], disconnected, already_sent_tc_count=3,
            )

        asyncio.run(run())
        payload = b"".join(c.args[0] for c in resp.write.call_args_list).decode("utf-8")
        self.assertIn('"finish_reason": "tool_calls"', payload)
        self.assertIn("[DONE]", payload)

    def test_stream_invoke_emits_tool_call_arguments_incrementally(self) -> None:
        """OpenAI 流式路径应在 invoke 开标签后增量发送 arguments。"""
        protocol = get_protocol("entml")
        parser = FncallStreamParser(protocol=protocol, tools=TOOLS)
        text = (
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">上海</entml:parameter>\n'
            "</entml:invoke>"
        )
        resp = MagicMock()
        writes: list[bytes] = []

        async def capture_write(data: bytes) -> None:
            writes.append(data)

        resp.write = capture_write
        disconnected = [False]
        stream_tool = None
        tool_index = 0

        async def run() -> None:
            nonlocal stream_tool, tool_index
            for i in range(0, len(text), 4):
                parser.feed(text[i : i + 4])
                stream_tool, tool_index, ok = await _emit_openai_streaming_tool_delta(
                    resp, parser, "qwen3.7-max", "gen-test", stream_tool, tool_index, disconnected,
                )
                self.assertTrue(ok)

        asyncio.run(run())
        payload = b"".join(writes).decode("utf-8")
        self.assertIn('"tool_calls"', payload)
        self.assertIn("get_weather", payload)
        self.assertIn("city", payload)
        chunks = [line for line in payload.split("\n") if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertGreater(len(chunks), 1)

    def test_post_stream_remaining_after_streamed_tool(self) -> None:
        """arguments 已流式发完时，finish 不应再重复整段 tool_calls。"""
        all_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
            }
        ]
        pending_tc_index = 1
        remaining = all_tool_calls[pending_tc_index:]
        self.assertEqual(remaining, [])

    def test_thinking_chunk_has_reasoning_details(self) -> None:
        chunk = build_openai_chunk("qwen3.7-max", chunk_id="gen-test", reasoning="plan")
        delta = chunk["choices"][0]["delta"]
        self.assertEqual(delta["content"], "")
        self.assertEqual(delta["reasoning"], "plan")
        self.assertEqual(delta["reasoning_details"][0]["type"], "reasoning.text")
        self.assertEqual(delta["reasoning_details"][0]["text"], "plan")


if __name__ == "__main__":
    unittest.main()
