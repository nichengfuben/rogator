from __future__ import annotations

import time
import unittest

from server.formats.messages import build_chat_payload, build_qwen_message


class TestQwenPayload(unittest.TestCase):
    def test_completion_timestamp_seconds(self) -> None:
        msg = build_qwen_message("hello", "qwen3.7-max")
        payload = build_chat_payload("chat-1", "qwen3.7-max", msg)
        ts = payload["timestamp"]
        self.assertGreaterEqual(ts, int(time.time()) - 2)
        self.assertLess(ts, int(time.time()) + 2)
        self.assertLess(ts, 10_000_000_000)

    def test_no_stream_options_by_default(self) -> None:
        msg = build_qwen_message("hello", "qwen3.7-max")
        payload = build_chat_payload("chat-1", "qwen3.7-max", msg)
        self.assertNotIn("stream_options", payload)

    def test_stream_options_when_include_usage(self) -> None:
        msg = build_qwen_message("hello", "qwen3.7-max")
        payload = build_chat_payload("chat-1", "qwen3.7-max", msg, include_usage=True)
        self.assertEqual(payload.get("stream_options"), {"include_usage": True})

    def test_auto_search_left_off(self) -> None:
        # 不碰上游原生搜索 / MCP；保持关闭
        msg = build_qwen_message("hello", "qwen3.7-max")
        self.assertFalse(msg["feature_config"]["auto_search"])
        self.assertEqual(msg["chat_type"], "t2t")

    def test_files_preserved_when_attached(self) -> None:
        files = [{"id": "f1", "name": "a.txt", "type": "file"}]
        msg = build_qwen_message("tail", "qwen3.7-max", files)
        self.assertEqual(msg["content"], "tail")
        self.assertEqual(msg["files"], files)
        self.assertFalse(msg["feature_config"]["auto_search"])


if __name__ == "__main__":
    unittest.main()
