from __future__ import annotations

"""纯 Python fireye 模块测试（231! bx-ua 形态与 HAR 布局约束）。"""

import base64
import unittest

from upstream.qwen.auth.crypto import get_baxia_tokens, reset_baxia_runtime
from upstream.qwen.auth.fireye import get_fy_token, reset_session
from upstream.qwen.auth.fireye.codec import expected_magic, unwrap_token


class TestFireyePython(unittest.TestCase):
    def setUp(self) -> None:
        reset_baxia_runtime()
        reset_session()

    def tearDown(self) -> None:
        reset_baxia_runtime()
        reset_session()

    def test_fy_token_231_prefix(self) -> None:
        ua = get_fy_token("https://chat.qwen.ai/api/v2/chats/new")
        self.assertTrue(ua.startswith("231!"), ua[:40])
        self.assertGreater(len(ua), 100)

    def test_binary_layout_magic_byte(self) -> None:
        ua = get_fy_token("https://chat.qwen.ai/api/v2/chat/completions")
        raw = unwrap_token(ua)
        self.assertGreaterEqual(len(raw), 1050)
        self.assertEqual(raw[5], expected_magic())

    def test_session_stable_blocks(self) -> None:
        url = "https://chat.qwen.ai/api/v2/chats?page=1"
        a = unwrap_token(get_fy_token(url))
        b = unwrap_token(get_fy_token(url))
        self.assertEqual(a[12:28], b[12:28])
        self.assertNotEqual(a[:12], b[:12])

    def test_chat_context_changes_block(self) -> None:
        base = "https://chat.qwen.ai/api/v2/chats?page=1"
        chat = (
            "https://chat.qwen.ai/api/v2/chat/completions"
            "?chat_id=f07fc0a2-f718-4076-8f7d-56834a8013bb"
        )
        blk_base = unwrap_token(get_fy_token(base))[12:28]
        blk_chat = unwrap_token(get_fy_token(chat))[12:28]
        self.assertNotEqual(blk_base, blk_chat)
        ctx_base = unwrap_token(get_fy_token(base))[28:44]
        ctx_chat = unwrap_token(get_fy_token(chat))[28:44]
        self.assertNotEqual(ctx_base, ctx_chat)

    def test_get_baxia_tokens_integration(self) -> None:
        a = get_baxia_tokens(req_url="https://chat.qwen.ai/api/v2/chat/completions")
        b = get_baxia_tokens(req_url="https://chat.qwen.ai/api/v2/chats/new")
        self.assertTrue(a["bxUa"].startswith("231!"))
        self.assertTrue(b["bxUa"].startswith("231!"))
        self.assertNotEqual(a["bxUa"], b["bxUa"])
        self.assertEqual(a["bxUmidToken"], b["bxUmidToken"])
        raw = base64.b64decode(a["bxUa"][4:])
        self.assertEqual(raw[5], expected_magic())


if __name__ == "__main__":
    unittest.main()
