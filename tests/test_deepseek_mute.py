from __future__ import annotations

"""DeepSeek 上游 mute 检测与 24h 登录屏蔽。"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.session.accounts import Account
from core.session.store import (
    MUTE_LOGIN_BLOCK_SECONDS,
    SessionStoreMeta,
    is_account_mute_blocked,
    load_upstream_sessions,
    save_upstream_sessions,
)
from upstream.deepseek.client import DeepSeekClient
from upstream.deepseek.lib.adapter.helpers.biz_error import (
    DeepSeekUserMutedError,
    DeepSeekWafChallengeError,
    parse_biz_error_from_line,
    raise_if_user_muted,
    raise_if_waf_challenge,
)


class TestBizErrorParse(unittest.TestCase):
    def test_detect_user_muted_line(self) -> None:
        line = (
            '{"code":0,"msg":"","data":{"biz_code":5,'
            '"biz_msg":"user is muted","biz_data":{"is_muted":1,"mute_until":1785423511.027}}}'
        )
        biz = parse_biz_error_from_line(line)
        self.assertIsNotNone(biz)
        assert biz is not None
        self.assertEqual(biz["biz_code"], 5)
        self.assertIn("muted", biz["biz_msg"].lower())

    def test_raise_if_user_muted(self) -> None:
        line = '{"code":0,"data":{"biz_code":5,"biz_msg":"user is muted"}}'
        with self.assertRaises(DeepSeekUserMutedError):
            raise_if_user_muted(line)

    def test_raise_if_waf_challenge(self) -> None:
        with self.assertRaises(DeepSeekWafChallengeError):
            raise_if_waf_challenge(202, {"x-amzn-waf-action": "challenge"})
        raise_if_waf_challenge(202, {"x-amzn-waf-action": "allow"})
        raise_if_waf_challenge(200, {"x-amzn-waf-action": "challenge"})


class TestMutedAccountsPersist(unittest.TestCase):
    def test_save_and_load_muted_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "deepseek.json"
            with patch("core.session.store.PROJECT_ROOT", root):
                with patch("core.session.store.sessions_file", lambda up: path):
                    with patch("core.session.store._maybe_migrate_upstream_sessions"):
                        now = time.time()
                        save_upstream_sessions(
                            "deepseek",
                            [],
                            muted_accounts={"a@test.com": now},
                        )
                        _, meta = load_upstream_sessions("deepseek")
                    self.assertIn("a@test.com", meta.muted_accounts)

    def test_mute_block_expires_after_24h(self) -> None:
        now = time.time()
        muted = {"u@test.com": now - MUTE_LOGIN_BLOCK_SECONDS - 1}
        self.assertFalse(is_account_mute_blocked(muted, "u@test.com", now=now))
        fresh = {"u@test.com": now - 100}
        self.assertTrue(is_account_mute_blocked(fresh, "u@test.com", now=now))


class TestSessionPoolMute(unittest.IsolatedAsyncioTestCase):
    async def test_handle_account_muted_removes_session_and_blocks_login(self) -> None:
        with patch("core.session.pool.load_upstream_sessions", return_value=([], SessionStoreMeta())):
            client = DeepSeekClient(MagicMock())
        client._save_meta = MagicMock(return_value=[])

        with patch("core.session.pool.accounts_for_upstream", return_value=[Account(username="m@test.com", password="pw")]):
            client.handle_account_muted("m@test.com", mute_at=time.time())
            self.assertIn("m@test.com", client._muted_accounts)
            picked = client._pick_account_for_login()
        self.assertIsNone(picked)

    async def test_replenish_skips_when_all_muted(self) -> None:
        empty_meta = SessionStoreMeta(muted_accounts={"only@test.com": time.time()})
        with patch("core.session.pool.load_upstream_sessions", return_value=([], empty_meta)):
            client = DeepSeekClient(MagicMock())
        client._prelogin_target = 2
        client._login_interval = 0.0
        client._ensure_cleanup = AsyncMock()
        client.login_account = AsyncMock()

        with patch("core.session.pool.accounts_for_upstream", return_value=[Account(username="only@test.com", password="pw")]):
            await client.replenish_sessions(2)
        client.login_account.assert_not_called()

    async def test_perform_login_waf_challenge_mutes_account(self) -> None:
        with patch("core.session.pool.load_upstream_sessions", return_value=([], SessionStoreMeta())):
            client = DeepSeekClient(MagicMock())
        client._save_meta = MagicMock(return_value=[])
        client._http = MagicMock(closed=False)
        client._ensure_ready = AsyncMock(return_value=MagicMock(_session=MagicMock(), _hif_managers={}))

        with patch(
            "upstream.deepseek.client.login",
            AsyncMock(side_effect=DeepSeekWafChallengeError()),
        ):
            ps = await client._perform_login(Account(username="waf@test.com", password="pw"))

        self.assertIsNone(ps)
        self.assertIn("waf@test.com", client._muted_accounts)


if __name__ == "__main__":
    unittest.main()
