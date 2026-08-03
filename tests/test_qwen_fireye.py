from __future__ import annotations

"""fireyejs Node runner integration tests (skip if node/jsdom unavailable)."""

import unittest

from upstream.qwen.auth.crypto import get_baxia_tokens, reset_baxia_runtime
from upstream.qwen.auth.fy_bridge import fireye_available, request_fireye_tokens, reset_fireye_worker


@unittest.skipUnless(fireye_available(), "fireye runner not available")
class TestFireyeRunner(unittest.TestCase):
    def setUp(self) -> None:
        reset_baxia_runtime()

    def tearDown(self) -> None:
        reset_fireye_worker()
        reset_baxia_runtime()

    def test_runner_returns_231_ua(self) -> None:
        data = request_fireye_tokens("https://chat.qwen.ai/api/v2/chats/new")
        self.assertTrue(data.get("ok"))
        ua = str(data.get("bxUa") or "")
        self.assertTrue(ua.startswith("231!"), ua[:40])
        self.assertGreater(len(ua), 100)

    def test_get_baxia_tokens_uses_fireye_ua(self) -> None:
        a = get_baxia_tokens(req_url="https://chat.qwen.ai/api/v2/chat/completions")
        b = get_baxia_tokens(req_url="https://chat.qwen.ai/api/v2/chats/new")
        self.assertTrue(a["bxUa"].startswith("231!"))
        self.assertTrue(b["bxUa"].startswith("231!"))
        self.assertNotEqual(a["bxUa"], b["bxUa"])
        self.assertEqual(a["bxUmidToken"], b["bxUmidToken"])


if __name__ == "__main__":
    unittest.main()
