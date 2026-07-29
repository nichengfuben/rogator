from __future__ import annotations

"""count_tokens 预检估算测试（正式对话 usage 见 test_upstream_usage）。"""

import json
import unittest

from server.formats import UpstreamUsageTracker
from server.model.token_estimate import (
    estimate_anthropic_request_input_tokens,
    estimate_openai_request_input_tokens,
    estimate_tokens_from_char_count,
)


class TestTokenEstimate(unittest.TestCase):
    def test_char_count_div_three(self) -> None:
        self.assertEqual(estimate_tokens_from_char_count(0), 0)
        self.assertEqual(estimate_tokens_from_char_count(9), 3)
        self.assertEqual(estimate_tokens_from_char_count(10), 3)

    def test_anthropic_count_tokens_includes_system_tools_and_blocks(self) -> None:
        body = {
            "model": "claude-opus-4-6",
            "system": [{"type": "text", "text": "abc"}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        }
        without_tools = estimate_anthropic_request_input_tokens({
            "system": body["system"],
            "messages": body["messages"],
        })
        with_tools = estimate_anthropic_request_input_tokens(body)
        self.assertGreater(with_tools, without_tools)
        self.assertEqual(with_tools, estimate_tokens_from_char_count(
            len("abc")
            + len(json.dumps(body["messages"][0]["content"], ensure_ascii=False))
            + len(json.dumps(body["tools"], ensure_ascii=False))
        ))

    def test_openai_estimate_messages_and_tools(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "123456"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        }
        raw_len = 6 + len(json.dumps(body["tools"], ensure_ascii=False))
        self.assertEqual(estimate_openai_request_input_tokens(body), raw_len // 3)


class TestMessageUsageUsesUpstreamInput(unittest.TestCase):
    def test_openai_usage_prompt_from_upstream(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 9999, "output_tokens": 42},
        })
        usage = tracker.openai_stream_usage()
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage["prompt_tokens"], 9999)
        self.assertEqual(usage["completion_tokens"], 42)

    def test_anthropic_message_start_both_from_upstream(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 500, "output_tokens": 7},
        })
        self.assertEqual(tracker.anthropic_message_start_usage, {
            "input_tokens": 500,
            "output_tokens": 7,
        })


if __name__ == "__main__":
    unittest.main()
