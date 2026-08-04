from __future__ import annotations

"""QwenClient 账号级 cookie jar 与单次 chat 绑定测试。"""

import unittest
from unittest.mock import MagicMock

from upstream.qwen.auth.http import merge_session_cookies, sync_cookie_store
from upstream.qwen.client import QwenClient


class TestQwenCookieStore(unittest.TestCase):
    def test_begin_chat_cookies_stable_within_account(self) -> None:
        client = QwenClient(MagicMock())
        session = MagicMock(token="tok-a", user_id="uid-1", username="user_a")
        first = client.begin_chat_cookies(session)
        second = client.begin_chat_cookies(session)
        self.assertEqual(first["cna"], second["cna"])
        self.assertEqual(first["sca"], second["sca"])

    def test_begin_chat_cookies_isolated_between_accounts(self) -> None:
        client = QwenClient(MagicMock())
        sa = MagicMock(token="tok-a", user_id="uid-1", username="user_a")
        sb = MagicMock(token="tok-b", user_id="uid-2", username="user_b")
        ca = client.begin_chat_cookies(sa)
        cb = client.begin_chat_cookies(sb)
        self.assertNotEqual(ca["cna"], cb["cna"])

    def test_binding_snapshot_not_mutated_by_other_account(self) -> None:
        client = QwenClient(MagicMock())
        sa = MagicMock(token="tok-a", user_id="uid-1", username="user_a")
        sb = MagicMock(token="tok-b", user_id="uid-2", username="user_b")
        binding = client.begin_chat_cookies(sa)
        cna_a = binding["cna"]
        client.begin_chat_cookies(sb)
        self.assertEqual(binding["cna"], cna_a)

    def test_sync_cookie_store_roundtrip(self) -> None:
        store: dict = {}
        merged = merge_session_cookies("tok", store, user_id="u1")
        sync_cookie_store(store, merged)
        again = merge_session_cookies("tok", store, user_id="u1")
        self.assertEqual(merged["cna"], again["cna"])


if __name__ == "__main__":
    unittest.main()
