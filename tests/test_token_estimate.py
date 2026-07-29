from __future__ import annotations

"""count_tokens 与 input/output token 策略测试。"""

import json
import unittest

from server.formats import UpstreamUsageTracker
from server.token_estimate import (
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


class TestClientUsageMerge(unittest.TestCase):
    def test_output_from_upstream_input_from_estimate(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 9999, "output_tokens": 42},
        })
        usage = tracker.client_openai_usage(estimated_input_tokens=100)
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 42)
        self.assertEqual(usage["total_tokens"], 142)

    def test_anthropic_message_start_uses_estimate_and_upstream_output(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 500, "output_tokens": 7},
        })
        start = tracker.anthropic_message_start_usage_for(estimated_input_tokens=50)
        self.assertEqual(start, {"input_tokens": 50, "output_tokens": 7})

    def test_message_delta_output_only_from_upstream(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({"type": "usage", "data": {"output_tokens": 15}})
        self.assertEqual(tracker.anthropic_message_delta_usage, {"output_tokens": 15})


if __name__ == "__main__":
    unittest.main()
