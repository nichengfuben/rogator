from __future__ import annotations

"""Qwen web Cookie jar 单元测试（对齐 main.js JO / 抓包字段）。"""

import unittest
from types import SimpleNamespace

from upstream.qwen.auth.http import (
    absorb_response_cookies,
    build_cookie_string,
    generate_cookies,
    merge_session_cookies,
    sync_cookie_store,
)
from upstream.qwen.auth.crypto import build_headers


class TestWebCookies(unittest.TestCase):
    def test_fe_defaults_from_jschunk_and_capture(self) -> None:
        jar = merge_session_cookies("tok-abc", user_id="uid-9")
        self.assertEqual(jar["token"], "tok-abc")
        self.assertEqual(jar["qwen-theme"], "dark")
        self.assertEqual(jar["qwen-locale"], "zh-CN")
        self.assertEqual(jar["qwen-thinking_mode"], "Fast")
        self.assertEqual(jar["xlly_s"], "1")
        self.assertEqual(jar["x-ap"], "cn-hongkong")
        self.assertEqual(jar["cnaui"], "uid-9")
        self.assertEqual(jar["aui"], "uid-9")
        self.assertEqual(len(jar["cna"]), 24)
        self.assertNotIn("fingerprint", jar)

    def test_extra_overrides_and_meta_stripped(self) -> None:
        jar = merge_session_cookies(
            "tok",
            {"fingerprint": "x", "cna": "fixed-cna-value-0000001", "sca": "deadbeef"},
            user_id="u1",
        )
        self.assertEqual(jar["cna"], "fixed-cna-value-0000001")
        self.assertEqual(jar["sca"], "deadbeef")
        self.assertNotIn("fingerprint", jar)
        header = build_cookie_string(jar)
        self.assertIn("token=tok", header)
        self.assertNotIn("fingerprint=", header)

    def test_merge_preserves_cna_on_second_call(self) -> None:
        store: dict = {}
        first = merge_session_cookies("tok", store, user_id="u1")
        sync_cookie_store(store, first)
        second = merge_session_cookies("tok", store, user_id="u1")
        self.assertEqual(first["cna"], second["cna"])
        self.assertEqual(first["sca"], second["sca"])

    def test_absorb_set_cookie(self) -> None:
        jar: dict = {}
        resp = SimpleNamespace(
            cookies={
                "acw_tc": SimpleNamespace(value="acw-1"),
                "tfstk": SimpleNamespace(value="tf-1"),
                "ignore_me": SimpleNamespace(value="x"),
            }
        )
        absorb_response_cookies(jar, resp)
        self.assertEqual(jar["acw_tc"], "acw-1")
        self.assertEqual(jar["tfstk"], "tf-1")
        self.assertNotIn("ignore_me", jar)

    def test_completions_cookie_header_has_fe_keys(self) -> None:
        cookies = generate_cookies(user_id="uid-2", thinking_mode="Thinking")
        headers = build_headers(
            "tok",
            include_sse=True,
            api_path="/api/v2/chat/completions",
            cookies=cookies,
        )
        cookie = headers.get("Cookie", "")
        for key in (
            "token=",
            "qwen-theme=",
            "qwen-locale=",
            "qwen-thinking_mode=Thinking",
            "xlly_s=1",
            "cnaui=uid-2",
        ):
            self.assertIn(key, cookie)
        self.assertNotIn("Authorization", headers)


if __name__ == "__main__":
    unittest.main()
