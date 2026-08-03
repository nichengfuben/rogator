from __future__ import annotations

"""对齐 FE jschunk：Baxia token 生成时机与注入范围。"""

import unittest

from upstream.qwen.auth.crypto import (
    build_headers,
    ensure_baxia_runtime,
    get_baxia_tokens,
    path_needs_baxia_ua,
    reset_baxia_runtime,
    resolve_baxia_mode,
    validate_bxumidtoken,
)


class TestBaxiaPathGate(unittest.TestCase):
    def test_protected_paths(self) -> None:
        self.assertTrue(path_needs_baxia_ua("/api/v2/chats/new"))
        self.assertTrue(path_needs_baxia_ua("/api/v2/chat/completions"))
        self.assertTrue(path_needs_baxia_ua("/api/v2/files/getstsToken"))
        self.assertFalse(path_needs_baxia_ua("/api/v2/users/status"))
        self.assertFalse(path_needs_baxia_ua("/api/v2/library/list"))
        self.assertFalse(path_needs_baxia_ua("/api/v2/projects/"))

    def test_resolve_mode(self) -> None:
        self.assertEqual(resolve_baxia_mode("/api/v2/chats/new"), "full")
        self.assertEqual(resolve_baxia_mode("/api/v2/users/status"), "version")
        self.assertEqual(resolve_baxia_mode("", explicit="none"), "none")
        self.assertEqual(resolve_baxia_mode(""), "full")


class TestBaxiaSessionTiming(unittest.TestCase):
    def setUp(self) -> None:
        reset_baxia_runtime()

    def tearDown(self) -> None:
        reset_baxia_runtime()

    def test_umid_stable_ua_rotates(self) -> None:
        a = get_baxia_tokens()
        b = get_baxia_tokens()
        self.assertEqual(a["bxUmidToken"], b["bxUmidToken"])
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertNotEqual(a["bxUa"], b["bxUa"])
        self.assertTrue(validate_bxumidtoken(a["bxUmidToken"]))
        self.assertEqual(len(a["bxUmidToken"]), 68)

    def test_reset_rotates_session(self) -> None:
        first = ensure_baxia_runtime()
        reset_baxia_runtime()
        second = ensure_baxia_runtime()
        self.assertNotEqual(first[1], second[1])

    def test_header_modes(self) -> None:
        full = build_headers("tok", api_path="/api/v2/chats/new")
        self.assertIn("bx-ua", full)
        self.assertIn("bx-umidtoken", full)
        self.assertEqual(full["bx-v"], "2.5.37")
        version = build_headers("tok", baxia="version")
        self.assertIn("bx-v", version)
        self.assertNotIn("bx-ua", version)
        self.assertNotIn("bx-umidtoken", version)
        none = build_headers("tok", baxia="none")
        self.assertNotIn("bx-v", none)
        self.assertNotIn("bx-ua", none)


if __name__ == "__main__":
    unittest.main()
