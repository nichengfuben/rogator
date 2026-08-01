from __future__ import annotations

"""上游 token 用量解析与 OAI / Anthropic 格式输出测试。"""

import json
import unittest

from upstream.qwen.chat.sse import parse_sse_event
from server.formats import (
    UpstreamUsageTracker,
    build_openai_chunk,
    build_openai_response,
    convert_to_anthropic,
    normalize_upstream_usage,
    should_emit_anthropic_message_start,
)


class TestNormalizeUpstreamUsage(unittest.TestCase):
    def test_qwen_input_output_fields(self) -> None:
        usage = normalize_upstream_usage({
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        })
        self.assertEqual(usage, {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
        })

    def test_openai_field_names_passthrough(self) -> None:
        usage = normalize_upstream_usage({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        })
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 20)

    def test_total_derived_when_missing(self) -> None:
        usage = normalize_upstream_usage({"input_tokens": 5, "output_tokens": 7})
        self.assertEqual(usage["total_tokens"], 12)

    def test_empty_returns_zeros(self) -> None:
        self.assertEqual(normalize_upstream_usage(None), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })


class TestUpstreamUsageTracker(unittest.TestCase):
    def test_stream_estimate_before_upstream(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.set_estimated_input_from_prompt_chars(664_430)
        self.assertEqual(tracker.anthropic_message_start_usage, {
            "input_tokens": 166_107,
            "output_tokens": 0,
        })
        tracker.add_output_chars(1200)
        self.assertEqual(tracker.anthropic_message_delta_usage, {
            "input_tokens": 166_107,
            "output_tokens": 300,
        })
        usage = tracker.openai_stream_usage()
        assert usage is not None
        self.assertEqual(usage["prompt_tokens"], 166_107)
        self.assertEqual(usage["completion_tokens"], 300)

    def test_upstream_overrides_stream_estimate_at_finish(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.set_estimated_input_from_prompt_chars(1000)
        tracker.add_output_chars(400)
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 185251, "output_tokens": 300},
        })
        self.assertEqual(tracker.anthropic_message_start_usage["input_tokens"], 185251)
        self.assertEqual(tracker.anthropic_message_delta_usage, {
            "input_tokens": 185251,
            "output_tokens": 300,
        })
        assert tracker.openai_stream_usage() is not None
        self.assertEqual(tracker.openai_stream_usage()["prompt_tokens"], 185251)

    def test_ingest_usage_event(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({"type": "usage", "data": {"input_tokens": 3, "output_tokens": 4}})
        self.assertEqual(tracker.openai_usage["prompt_tokens"], 3)
        self.assertEqual(tracker.openai_usage["completion_tokens"], 4)
        self.assertTrue(tracker.has_usage)

    def test_ingest_attached_usage_on_answer(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "answer",
            "content": "hi",
            "usage": {"input_tokens": 8, "output_tokens": 2},
        })
        self.assertEqual(tracker.anthropic_message_start_usage, {
            "input_tokens": 8,
            "output_tokens": 0,
        })
        self.assertEqual(tracker.anthropic_message_delta_usage, {
            "input_tokens": 8,
            "output_tokens": 2,
        })

    def test_anthropic_message_start_includes_cache_read(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {
                "input_tokens": 124392,
                "output_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 122496},
            },
        })
        self.assertEqual(tracker.anthropic_message_start_usage["input_tokens"], 124392)
        self.assertEqual(tracker.anthropic_message_start_usage["output_tokens"], 0)
        self.assertEqual(
            tracker.anthropic_message_start_usage["cache_read_input_tokens"],
            122496,
        )

    def test_cumulative_last_snapshot_wins(self) -> None:
        tracker = UpstreamUsageTracker()
        for output in (3, 5, 9, 338):
            tracker.ingest_event({
                "type": "thinking",
                "content": "x",
                "usage": {"input_tokens": 336, "output_tokens": output, "total_tokens": 336 + output},
            })
        self.assertEqual(tracker.openai_usage["prompt_tokens"], 336)
        self.assertEqual(tracker.openai_usage["completion_tokens"], 338)
        self.assertEqual(tracker.openai_usage["total_tokens"], 674)

    def test_cached_tokens_in_openai_usage(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 42},
            },
        })
        self.assertEqual(tracker.openai_usage["prompt_tokens_details"], {"cached_tokens": 42})
        self.assertEqual(tracker.last_raw_usage["input_tokens"], 100)

    def test_log_qwen_upstream_usage_emits_debug_json(self) -> None:
        from unittest.mock import patch

        from server.formats.usage import log_qwen_upstream_usage

        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
        })
        with patch("server.formats.usage.logger") as log:
            log_qwen_upstream_usage("req-test", tracker)
        log.debug.assert_called_once()
        msg = log.debug.call_args[0][2]
        self.assertIn("input_tokens", msg)
        self.assertIn("9", msg)


class TestAnthropicMessageStartTiming(unittest.TestCase):
    def test_skip_response_created(self) -> None:
        ev = {"type": "response_created", "response_id": "r1"}
        self.assertFalse(should_emit_anthropic_message_start(ev, False))

    def test_emit_on_first_usage(self) -> None:
        ev = {"type": "usage", "data": {"input_tokens": 10, "output_tokens": 1}}
        self.assertTrue(should_emit_anthropic_message_start(ev, False))

    def test_emit_on_thinking_without_usage(self) -> None:
        ev = {"type": "thinking", "content": "plan"}
        self.assertFalse(should_emit_anthropic_message_start(ev, False))

    def test_no_double_start(self) -> None:
        ev = {"type": "answer", "content": "hi", "usage": {"input_tokens": 1, "output_tokens": 2}}
        self.assertFalse(should_emit_anthropic_message_start(ev, True))


    def test_message_delta_includes_cumulative_input_output(self) -> None:
        tracker = UpstreamUsageTracker()
        tracker.ingest_event({
            "type": "usage",
            "data": {"input_tokens": 100, "output_tokens": 42},
        })
        self.assertEqual(tracker.anthropic_message_delta_usage, {
            "input_tokens": 100,
            "output_tokens": 42,
        })


class TestUsageInResponses(unittest.TestCase):
    def test_openai_non_stream_response(self) -> None:
        resp = build_openai_response(
            "qwen3.7-max",
            "hello",
            usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        )
        self.assertEqual(resp["usage"]["prompt_tokens"], 11)
        self.assertEqual(resp["usage"]["completion_tokens"], 22)

    def test_openai_stream_finish_chunk(self) -> None:
        chunk = build_openai_chunk(
            "qwen3.7-max",
            chunk_id="gen-test",
            finish_reason="stop",
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 9,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        )
        self.assertEqual(chunk["usage"]["total_tokens"], 14)
        self.assertEqual(chunk["usage"]["prompt_tokens_details"]["cached_tokens"], 2)

    def test_convert_to_anthropic_usage(self) -> None:
        openai_resp = build_openai_response(
            "qwen3.7-max",
            "hello",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        anth = convert_to_anthropic(openai_resp)
        self.assertEqual(anth["usage"], {"input_tokens": 100, "output_tokens": 50})


class TestParseSseUsage(unittest.TestCase):
    def test_usage_only_event(self) -> None:
        payload = json.dumps({"usage": {"input_tokens": 20, "output_tokens": 10}})
        event = parse_sse_event(payload)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "usage")
        self.assertEqual(event["data"]["input_tokens"], 20)

    def test_usage_attached_to_answer_delta(self) -> None:
        payload = json.dumps({
            "choices": [{
                "delta": {"phase": "answer", "content": "x", "status": "typing"},
            }],
            "usage": {"input_tokens": 30, "output_tokens": 1},
        })
        event = parse_sse_event(payload)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "answer")
        self.assertEqual(event["usage"]["input_tokens"], 30)

    def test_empty_content_usage_tail(self) -> None:
        payload = json.dumps({
            "choices": [{
                "delta": {"phase": "answer", "content": "", "status": "typing"},
            }],
            "usage": {"input_tokens": 327, "output_tokens": 11, "total_tokens": 338},
        })
        event = parse_sse_event(payload)
        assert event is not None
        self.assertEqual(event["type"], "usage")
        self.assertEqual(event["data"]["output_tokens"], 11)


if __name__ == "__main__":
    unittest.main()
