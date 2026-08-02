from __future__ import annotations

"""DeepSeek 登录与 session 过期策略测试。"""

import asyncio
import base64
import json
import time
import unittest

import aiohttp

from core.session.accounts import Account, accounts_for_upstream
from core.session.store import PlatformSession, _deepseek_session_ttl, _jwt_exp
from upstream.deepseek.lib.user.userapi import login


def _make_jwt(exp: float) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(exp)}, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


class TestDeepSeekSessionToken(unittest.TestCase):
    def test_opaque_token_not_jwt(self) -> None:
        token = "jUt37lNFVvDQQueC2Jhimfmwy38yZ2w8mA4yesSm" + "x" * 24
        self.assertIsNone(_jwt_exp(token))
        self.assertEqual(len(token.split(".")), 1)

    def test_deepseek_ttl_defaults_one_hour(self) -> None:
        self.assertEqual(_deepseek_session_ttl(), 3600.0)

    def test_deepseek_opaque_expires_by_login_time_ttl(self) -> None:
        now = time.time()
        ttl = _deepseek_session_ttl()
        fresh = PlatformSession(
            account=Account(username="u@test.com", password="pw"),
            token="opaque-not-jwt-token",
            user_id="1",
            upstream="deepseek",
            login_time=now,
        )
        stale = PlatformSession(
            account=Account(username="u@test.com", password="pw"),
            token="opaque-not-jwt-token",
            user_id="1",
            upstream="deepseek",
            login_time=now - ttl - 60,
        )
        self.assertFalse(fresh.is_expired())
        self.assertTrue(stale.is_expired())

    def test_deepseek_jwt_uses_exp_minus_30(self) -> None:
        now = time.time()
        fresh = PlatformSession(
            account=Account(username="u@test.com", password="pw"),
            token=_make_jwt(now + 3600),
            user_id="1",
            upstream="deepseek",
        )
        near = PlatformSession(
            account=Account(username="u@test.com", password="pw"),
            token=_make_jwt(now + 20),
            user_id="1",
            upstream="deepseek",
        )
        self.assertFalse(fresh.is_expired())
        self.assertTrue(near.is_expired())


class TestDeepSeekLoginIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_live_login_token_is_opaque_not_jwt(self) -> None:
        pool = accounts_for_upstream("deepseek")
        if not pool:
            self.skipTest("no deepseek accounts configured")
        account = pool[0]
        async with aiohttp.ClientSession() as session:
            token, user_id, _did = await login(session, account.username, account.password)
        self.assertTrue(token)
        self.assertTrue(user_id)
        self.assertIsNone(_jwt_exp(token))
        self.assertEqual(len(token.split(".")), 1)
        ps = PlatformSession(
            account=account,
            token=token,
            user_id=user_id,
            upstream="deepseek",
        )
        self.assertFalse(ps.is_expired())


if __name__ == "__main__":
    unittest.main()
