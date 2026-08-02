from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from server.formats import BaxiaSmBlockedError, TokenExpiredError, UpstreamWafBlockedError
from core.session.accounts import Account
from upstream.qwen.chat.chat import iter_sse_events, raise_sse_inline_error
from upstream.qwen.chat.store import QwenSession


_BAXIA_SM = (
    '{"ret":["FAIL_SYS_USER_VALIDATE","RGV587_ERROR::SM::哎哟喂,被挤爆啦,请稍后重试"],'
    '"data":{"url":"https://chat.qwen.ai:443//api/v2/chat/completions/_____tmd_____/punish'
    '?x5secdata=abc&x5step=2&action=captcha&pureCaptcha="}}'
)


class TestQwenSseBaxiaError(unittest.TestCase):
    def test_raise_sse_inline_error_baxia_sm(self) -> None:
        client = MagicMock()
        session = QwenSession(
            account=Account(username="u@test.com", password="pw"),
            token="tok",
            user_id="u",
            login_time=0.0,
        )
        with self.assertRaises(BaxiaSmBlockedError) as ctx:
            raise_sse_inline_error(client, session, _BAXIA_SM)
        self.assertIn("RGV587", str(ctx.exception))
        client._invalidate_session.assert_not_called()

    def test_raise_sse_inline_error_success_false(self) -> None:
        client = MagicMock()
        client._invalidate_session = MagicMock()
        session = QwenSession(
            account=Account(username="u@test.com", password="pw"),
            token="tok",
            user_id="u",
            login_time=0.0,
        )
        line = '{"success":false,"message":"Token expired"}'
        with self.assertRaises(TokenExpiredError):
            raise_sse_inline_error(client, session, line)

    def test_raise_sse_inline_error_upstream_rate_limited(self) -> None:
        client = MagicMock()
        client._invalidate_session = MagicMock()
        session = QwenSession(
            account=Account(username="u@test.com", password="pw"),
            token="tok",
            user_id="u",
            login_time=0.0,
        )
        line = '{"success":false,"data":{"code":"RateLimited","details":"too many"}}'
        with self.assertRaises(TokenExpiredError) as ctx:
            raise_sse_inline_error(client, session, line)
        self.assertIn("Rate limited", str(ctx.exception))

    def test_raise_sse_inline_error_baxia_fail_sys_with_punish(self) -> None:
        client = MagicMock()
        session = QwenSession(
            account=Account(username="u@test.com", password="pw"),
            token="tok",
            user_id="u",
            login_time=0.0,
        )
        line = '{"ret":["FAIL_SYS"],"data":{"url":"https://x/punish?action=deny"}}'
        with self.assertRaises(BaxiaSmBlockedError):
            raise_sse_inline_error(client, session, line)


class TestIterSseBaxiaError(unittest.IsolatedAsyncioTestCase):
    async def test_iter_sse_events_raises_on_baxia_body(self) -> None:
        client = MagicMock()
        session = MagicMock(username="u@test.com")

        async def _body():
            yield _BAXIA_SM.encode("utf-8")

        resp = MagicMock()
        resp.content = _body()
        with self.assertRaises(BaxiaSmBlockedError):
            async for _ in iter_sse_events(client, resp, session):
                pass


if __name__ == "__main__":
    unittest.main()
