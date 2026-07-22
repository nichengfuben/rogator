from __future__ import annotations

"""session 过期与清理测试。"""

import time
import unittest
from unittest.mock import MagicMock

from accounts import Account
from server.formats import TOKEN_EXPIRE_SECONDS
from server.session_store import QwenSession, clean_expired
from server.qwen_client import QwenClient


def _session(age_seconds: float, valid: bool = True) -> QwenSession:
    return QwenSession(
        account=Account(username=f"user{int(age_seconds)}@test.com", password="pw"),
        token="token",
        user_id="uid",
        login_time=time.time() - age_seconds,
        is_valid=valid,
    )


class TestSessionExpiry(unittest.TestCase):
    def test_is_expired_by_login_age(self) -> None:
        fresh = _session(TOKEN_EXPIRE_SECONDS - 60)
        expired = _session(TOKEN_EXPIRE_SECONDS + 1)
        self.assertFalse(fresh.is_expired())
        self.assertTrue(expired.is_expired())

    def test_clean_expired_removes_old_and_invalid(self) -> None:
        sessions = [
            _session(100),
            _session(TOKEN_EXPIRE_SECONDS + 10),
            _session(200, valid=False),
        ]
        kept, removed = clean_expired(sessions)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 2)

    def test_prune_expired_on_get_valid_session(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [
            _session(TOKEN_EXPIRE_SECONDS + 5),
            _session(60),
        ]
        client._current_index = 0

        import asyncio
        session = asyncio.run(client.get_valid_session())

        self.assertEqual(len(client._sessions), 1)
        self.assertIsNotNone(session)
        self.assertFalse(session.is_expired())

    def test_current_session_hides_expired(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [_session(TOKEN_EXPIRE_SECONDS + 1)]
        client._current_index = 0
        self.assertIsNone(client.current_session)


if __name__ == "__main__":
    unittest.main()
