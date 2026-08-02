from __future__ import annotations

"""换号逻辑与 prelogin 维护测试。"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.qwen.account import Account
from upstream.qwen.client import QwenClient
from upstream.qwen.chat.store import QwenSession, valid_session_count, SessionStoreMeta
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

        new = await client.switch_to_next(exclude_username="a@test.com")
        self.assertIsNotNone(new)
        self.assertEqual(new.username, "b@test.com")
        self.assertEqual(client._current_index, 1)

    async def test_block_account_skips_relogin(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = []
        client._blocked_accounts = {"only@test.com": time.time() + 3600}
        client._save_meta = MagicMock(return_value=[])

        with patch("core.session.pool.accounts_for_upstream", return_value=[Account(username="only@test.com", password="pw")]):
            picked = client._pick_account_for_login()
        self.assertIsNone(picked)


class TestPreloginLoadAware(unittest.IsolatedAsyncioTestCase):
    async def test_idle_replenish_only_bootstrap(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = []
        client._prelogin_target = 32
        client._login_interval = 0.0
        client._ensure_cleanup = AsyncMock()
        client._pick_account_for_login = MagicMock(side_effect=[
            Account(username="a@test.com", password="pw"),
            Account(username="b@test.com", password="pw"),
        ])

        async def _fake_login(account: Account) -> QwenSession:
            qs = _session(account.username.split("@")[0])
            client._sessions.append(qs)
            return qs

        client.login_account = AsyncMock(side_effect=_fake_login)

        with patch("core.session.pool.accounts_for_upstream", return_value=[
            Account(username="a@test.com", password="pw"),
            Account(username="b@test.com", password="pw"),
        ]):
            await client.replenish_sessions()
        self.assertEqual(valid_session_count(client._sessions), 1)
        self.assertEqual(client.login_account.await_count, 1)

    async def test_replenish_scales_with_inflight(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = [_session("warm")]
        client._prelogin_target = 32
        client._login_interval = 0.0
        client._ensure_cleanup = AsyncMock()
        client._inflight = {"warm@test.com": 10}
        accounts = [
            Account(username=f"u{i}@test.com", password="pw")
            for i in range(20)
        ]
        client._pick_account_for_login = MagicMock(side_effect=accounts)

        async def _fake_login(account: Account) -> QwenSession:
            qs = _session(account.username.split("@")[0])
            client._sessions.append(qs)
            return qs

        client.login_account = AsyncMock(side_effect=_fake_login)

        with patch("core.session.pool.accounts_for_upstream", return_value=accounts):
            with patch.object(client, "_prelogin_cap", return_value=16):
                with patch.object(client, "_prelogin_headroom", return_value=8):
                    await client.replenish_sessions()
        # demand=10, headroom=8 → target=18, cap=16, valid=1 → need 15
        self.assertEqual(client.login_account.await_count, 15)

    async def test_urgent_replenish_skips_interval(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = []
        client._prelogin_target = 32
        client._login_interval = 999.0
        client._ensure_cleanup = AsyncMock()
        client._inflight = {"busy@test.com": 3}
        accounts = [
            Account(username=f"u{i}@test.com", password="pw")
            for i in range(8)
        ]
        client._pick_account_for_login = MagicMock(side_effect=accounts + [None])

        async def _fake_login(account: Account) -> QwenSession:
            qs = _session(account.username.split("@")[0])
            client._sessions.append(qs)
            return qs

        client.login_account = AsyncMock(side_effect=_fake_login)

        with patch("core.session.pool.accounts_for_upstream", return_value=accounts):
            with patch("core.session.pool.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
                with patch.object(client, "_prelogin_cap", return_value=16):
                    with patch.object(client, "_prelogin_headroom", return_value=2):
                        await client.replenish_sessions()
        self.assertGreaterEqual(client.login_account.await_count, 2)
        sleep_mock.assert_not_called()

    async def test_signal_replenish_urgent_with_existing_sessions(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = [_session("warm")]
        client._prelogin_target = 32
        client._login_interval = 999.0
        client._ensure_cleanup = AsyncMock()
        client.signal_replenish()
        accounts = [
            Account(username=f"u{i}@test.com", password="pw")
            for i in range(5)
        ]
        client._pick_account_for_login = MagicMock(side_effect=accounts)

        async def _fake_login(account: Account) -> QwenSession:
            qs = _session(account.username.split("@")[0])
            client._sessions.append(qs)
            return qs

        client.login_account = AsyncMock(side_effect=_fake_login)

        with patch("core.session.pool.accounts_for_upstream", return_value=accounts):
            with patch("core.session.pool.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
                with patch.object(client, "_prelogin_cap", return_value=16):
                    with patch.object(client, "_prelogin_headroom", return_value=4):
                        await client.replenish_sessions()
        self.assertGreaterEqual(client.login_account.await_count, 2)
        sleep_mock.assert_not_called()


class TestPrelogin(unittest.IsolatedAsyncioTestCase):
    async def test_prelogin_fills_to_target(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = []
        client._prelogin_target = 2
        client._login_interval = 0.0
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
        with patch("core.session.pool.accounts_for_upstream", return_value=stub_accounts):
            await client.prelogin_accounts(2)
        self.assertEqual(valid_session_count(client._sessions), 2)


if __name__ == "__main__":
    unittest.main()
