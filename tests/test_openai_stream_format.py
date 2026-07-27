from __future__ import annotations

"""OpenAI 流式 tool_calls 格式与 mock.py OpenAIBuilder 对齐测试。"""

import json
import unittest

from handlers.openai import _openai_tool_call_entry
from server.formats import build_openai_chunk


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

    def test_thinking_chunk_has_reasoning_details(self) -> None:
        chunk = build_openai_chunk("qwen3.7-max", chunk_id="gen-test", reasoning="plan")
        delta = chunk["choices"][0]["delta"]
        self.assertEqual(delta["content"], "")
        self.assertEqual(delta["reasoning"], "plan")
        self.assertEqual(delta["reasoning_details"][0]["type"], "reasoning.text")
        self.assertEqual(delta["reasoning_details"][0]["text"], "plan")


if __name__ == "__main__":
    unittest.main()
