from __future__ import annotations

"""换号逻辑与 prelogin 维护测试。"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from accounts import Account
from server.client.qwen_client import QwenClient
from server.client.session_store import QwenSession, valid_session_count, SessionStoreMeta
from tests.test_session_cleanup import _make_jwt


def _session(name: str, valid: bool = True) -> QwenSession:
    return QwenSession(
        account=Account(username=f"{name}@test.com", password="pw"),
        token=_make_jwt(time.time() + 3600),
        user_id=name,
        login_time=time.time(),
        is_valid=valid,
    )


class TestSwitchAccount(unittest.IsolatedAsyncioTestCase):
    async def test_switch_excludes_failed_username(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [
            _session("a"),
            _session("b"),
        ]
        client._current_index = 0
        client._save_meta = MagicMock(return_value=[])

        with patch("server.client.qwen_client.random.choice", return_value=1):
            new = await client.switch_to_next(exclude_username="a@test.com")
        self.assertIsNotNone(new)
        self.assertEqual(new.username, "b@test.com")
        self.assertEqual(client._current_index, 1)

    async def test_block_account_skips_relogin(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = []
        client._blocked_accounts = {"only@test.com": time.time() + 3600}
        client._save_meta = MagicMock(return_value=[])

        with patch("server.client.account.ACCOUNTS", [Account(username="only@test.com", password="pw")]):
            picked = client._pick_account_for_login()
        self.assertIsNone(picked)


class TestPrelogin(unittest.IsolatedAsyncioTestCase):
    async def test_prelogin_fills_to_target(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("server.client.qwen_client.load_session_store", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = []
        client._prelogin_target = 2
        client._ensure_cleanup = AsyncMock()
        client._pick_account_for_login = MagicMock(side_effect=[
            Account(username="a@test.com", password="pw"),
            Account(username="b@test.com", password="pw"),
            None,
        ])

        async def _fake_login(account: Account) -> QwenSession:
            qs = _session(account.username.split("@")[0])
            client._sessions.append(qs)
            return qs

        client.login_account = AsyncMock(side_effect=_fake_login)

        stub_accounts = [
            Account(username="a@test.com", password="pw"),
            Account(username="b@test.com", password="pw"),
        ]
        with patch("server.client.account.ACCOUNTS", stub_accounts):
            await client.prelogin_accounts(2)
        self.assertEqual(valid_session_count(client._sessions), 2)


if __name__ == "__main__":
    unittest.main()
