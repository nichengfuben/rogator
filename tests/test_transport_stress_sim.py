from __future__ import annotations

"""高强度模拟：不触达上游，验证 transport 腐化后 create_chat / 登录可恢复。"""

import asyncio
import importlib
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp.client_exceptions import ClientConnectorError

from core.session.accounts import Account
from core.session.store import PlatformSession
from core.transport.http import (
    _POOL_CONNECT_TIMEOUT,
    close_shared_connector,
    get_upstream_ssl_context,
    make_connector,
    reset_upstream_transport,
    upstream_timeout,
)
from core.transport.owned import HttpTransportMixin
from server.formats import LOGIN_TIMEOUT, UpstreamTimeoutError
from server.retry import run_with_session_retry, stream_with_session_retry
from upstream.qwen.account import QwenLoginMixin
from upstream.qwen.chat.chat import create_chat_for_session
from upstream.qwen.chat.store import QwenSession
from upstream.qwen.completion_stream import _post_chat_sse


def _qwen_session() -> QwenSession:
    return QwenSession(
        account=Account(username="u@test.com", password="pw"),
        token="tok",
        user_id="uid",
        upstream="qwen",
    )


def _mock_json_response(payload: Dict[str, Any], *, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value="")
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@dataclass
class PostScript:
    outcomes: List[Any] = field(default_factory=list)
    calls: int = 0
    timeouts_used: List[aiohttp.ClientTimeout] = field(default_factory=list)

    def build_request(self):
        def _request(*_args, **kwargs):
            self.calls += 1
            timeout = kwargs.get("timeout")
            if timeout is not None:
                self.timeouts_used.append(timeout)
            if not self.outcomes:
                return _mock_json_response(
                    {"success": True, "data": {"id": "fallback"}}
                )
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return _request


class TransportProbe(HttpTransportMixin):
    def __init__(self, script: PostScript) -> None:
        self._script = script
        self._last_used_proxy_enabled: bool = False
        self._init_http_transport()
        self.reset_count = 0

    def _get_proxy_kwarg(self) -> None:
        return None

    def _ensure_http_unlocked(self) -> aiohttp.ClientSession:
        if self._http is None or bool(getattr(self._http, "closed", True)):
            session = MagicMock(spec=aiohttp.ClientSession)
            session.closed = False
            session.close = AsyncMock()
            handler = self._script.build_request()
            session.request = handler
            session.post = handler
            self._http = session
        return self._http

    async def reset_http_transport(self) -> None:
        await super().reset_http_transport()
        self.reset_count += 1

    def _invalidate_session(self, _session: QwenSession) -> None:
        pass


class LoginProbe(QwenLoginMixin, TransportProbe):
    UPSTREAM_NAME = "qwen"

    def __init__(self, script: PostScript) -> None:
        TransportProbe.__init__(self, script)


class TestCreateChatTransportSim(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    async def test_create_chat_timeout_uses_connect_bound(self) -> None:
        script = PostScript(
            outcomes=[asyncio.TimeoutError()],
        )
        probe = TransportProbe(script)
        session = _qwen_session()
        with self.assertRaises(UpstreamTimeoutError):
            await create_chat_for_session(probe, session, "qwen3.7-max")
        self.assertGreaterEqual(script.calls, 1)
        tm = script.timeouts_used[0]
        self.assertEqual(tm.connect, _POOL_CONNECT_TIMEOUT)
        self.assertEqual(tm.total, 15.0)

    async def test_create_chat_succeeds_without_network(self) -> None:
        script = PostScript(
            outcomes=[
                _mock_json_response({"success": True, "data": {"id": "chat-xyz"}})
            ],
        )
        probe = TransportProbe(script)
        session = _qwen_session()
        chat_id = await create_chat_for_session(probe, session, "qwen3.7-max")
        self.assertEqual(chat_id, "chat-xyz")
        self.assertEqual(script.calls, 1)

    async def test_session_retry_resets_transport_on_create_chat_timeout(self) -> None:
        script = PostScript(
            outcomes=[
                asyncio.TimeoutError(),
                _mock_json_response({"success": True, "data": {"id": "chat-retry"}}),
            ],
        )
        probe = TransportProbe(script)
        session = _qwen_session()
        state = MagicMock()
        state.is_shutting_down = False
        calls = {"n": 0}

        async def _run() -> str:
            calls["n"] += 1
            return await create_chat_for_session(probe, session, "qwen3.7-max")

        chat_id = await run_with_session_retry("req-cc", state, _run, client=probe)
        self.assertEqual(chat_id, "chat-retry")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(probe.reset_count, 1)
        self.assertEqual(script.calls, 2)


class TestLoginTransportSim(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    async def test_login_timeout_resets_transport(self) -> None:
        script = PostScript(outcomes=[asyncio.TimeoutError()])
        probe = LoginProbe(script)
        account = Account(username="login@test.com", password="secret")
        result = await probe._perform_login(account)
        self.assertIsNone(result)
        self.assertEqual(probe.reset_count, 1)

    async def test_login_uses_connect_bound_timeout(self) -> None:
        script = PostScript(
            outcomes=[
                _mock_json_response(
                    {"success": True, "data": {"access_token": "access-tok"}},
                ),
            ],
        )
        probe = LoginProbe(script)
        account = Account(username="ok@test.com", password="secret")
        with patch(
            "upstream.qwen.account.fetch_user_id",
            new_callable=AsyncMock,
            return_value="uid-1",
        ), patch(
            "upstream.qwen.chat.upload.upstream_api.warmup_session",
            new_callable=AsyncMock,
        ):
            ps = await probe._perform_login(account)
        self.assertIsNotNone(ps)
        self.assertEqual(script.calls, 1)
        tm = script.timeouts_used[0]
        self.assertEqual(tm.connect, _POOL_CONNECT_TIMEOUT)
        self.assertEqual(tm.total, LOGIN_TIMEOUT)

    async def test_login_recovers_on_second_attempt_after_manual_reset(self) -> None:
        script = PostScript(
            outcomes=[
                asyncio.TimeoutError(),
                _mock_json_response(
                    {"success": True, "data": {"access_token": "tok2"}},
                ),
            ],
        )
        probe = LoginProbe(script)
        account = Account(username="retry@test.com", password="secret")
        first = await probe._perform_login(account)
        self.assertIsNone(first)
        self.assertEqual(probe.reset_count, 1)
        with patch(
            "upstream.qwen.account.fetch_user_id",
            new_callable=AsyncMock,
            return_value="uid-2",
        ):
            second = await probe._perform_login(account)
        self.assertIsNotNone(second)
        self.assertEqual(second.token, "tok2")


class TestTransportStressSim(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    async def test_connector_reset_under_concurrency(self) -> None:
        async def _worker(worker_id: int) -> int:
            session = aiohttp.ClientSession(
                connector=make_connector(), connector_owner=False
            )
            try:
                if worker_id % 3 == 0:
                    await reset_upstream_transport(session)
                    return worker_id
                return -1
            finally:
                if not session.closed:
                    await session.close()

        results = await asyncio.gather(*[_worker(i) for i in range(24)])
        reset_workers = [r for r in results if r >= 0]
        self.assertEqual(len(reset_workers), 8)
        conn = make_connector()
        self.assertFalse(conn.closed)

    async def test_many_reset_cycles_keep_ssl_context_valid(self) -> None:
        for _ in range(32):
            session = aiohttp.ClientSession(
                connector=make_connector(), connector_owner=False
            )
            ctx = get_upstream_ssl_context()
            self.assertFalse(ctx.check_hostname)
            await reset_upstream_transport(session)
        ctx_after = get_upstream_ssl_context()
        self.assertFalse(ctx_after.check_hostname)
        self.assertFalse(make_connector().closed)

    async def test_timeout_error_not_treated_as_connection_error(self) -> None:
        from server.formats import as_upstream_connection_error

        self.assertIsNone(as_upstream_connection_error(asyncio.TimeoutError()))

    async def test_session_closed_mapped_as_connection_error(self) -> None:
        from server.formats import as_upstream_connection_error

        err = as_upstream_connection_error(
            RuntimeError("Session is closed"),
            upstream="deepseek",
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertEqual(err.upstream, "deepseek")
        self.assertIn("Session is closed", err.message)

    async def test_stale_session_attribute_error_mapped_as_connection_error(
        self,
    ) -> None:
        from server.formats import as_upstream_connection_error

        err = as_upstream_connection_error(
            AttributeError(
                "'NoneType' object has no attribute '_timeout_ceil_threshold'"
            ),
            upstream="qwen",
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertEqual(err.upstream, "qwen")
        self.assertIn("_timeout_ceil_threshold", err.message)

    async def test_aiohttp_connector_assertion_mapped_as_connection_error(self) -> None:
        from server.formats import as_upstream_connection_error
        from server.formats.errors import _traceback_touches_aiohttp_client

        self.assertFalse(_traceback_touches_aiohttp_client(AssertionError()))
        with patch(
            "server.formats.errors._traceback_touches_aiohttp_client",
            return_value=True,
        ):
            err = as_upstream_connection_error(AssertionError(), upstream="qwen")
        self.assertIsNotNone(err)
        assert err is not None
        self.assertEqual(err.upstream, "qwen")

    async def test_post_chat_sse_retries_on_stale_session(self) -> None:
        script = PostScript(
            outcomes=[
                RuntimeError("Session is closed"),
                _mock_json_response({"success": True, "data": {"id": "ignored"}}),
            ],
        )
        probe = TransportProbe(script)
        session = _qwen_session()

        async def _fake_iter(*_args, **_kwargs):
            yield {"type": "answer", "content": "ok"}
            return

        with patch(
            "upstream.qwen.completion_stream._iter_qwen_sse_or_reconnect",
            side_effect=lambda *_a, **_k: _fake_iter(),
        ), patch(
            "upstream.qwen.completion_stream.handle_chat_error",
            new=AsyncMock(),
        ):
            events = [
                evt
                async for evt in _post_chat_sse(
                    probe,
                    session,
                    "chat-1",
                    {},
                    {},
                    [],
                )
            ]
        self.assertEqual(len(events), 1)
        self.assertEqual(probe.reset_count, 1)
        self.assertEqual(script.calls, 2)

    async def test_closing_one_unowned_session_keeps_peer_alive(self) -> None:
        from server.retry.http_client import client_session

        s1 = client_session()
        s2 = client_session()
        self.assertIs(s1.connector, s2.connector)
        await reset_upstream_transport(s1)
        self.assertFalse(s2.closed)
        self.assertFalse(make_connector().closed)
        await s2.close()

    async def test_connection_error_retry_resets_transport(self) -> None:
        script = PostScript(
            outcomes=[
                ClientConnectorError(MagicMock(), OSError("stale keep-alive")),
                _mock_json_response({"success": True, "data": {"id": "after-reset"}}),
            ],
        )
        probe = TransportProbe(script)
        session = _qwen_session()
        chat_id = await create_chat_for_session(probe, session, "qwen3.7-max")
        self.assertEqual(chat_id, "after-reset")
        self.assertEqual(probe.reset_count, 1)
        self.assertEqual(script.calls, 2)

    async def test_stream_retry_resets_transport_on_timeout(self) -> None:
        probe = TransportProbe(PostScript())
        state = MagicMock()
        state.is_shutting_down = False
        attempts = {"n": 0}

        async def make_stream():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise UpstreamTimeoutError("Create chat timed out after 15s")
            yield {"type": "answer", "content": "hi"}

        events: List[Dict[str, Any]] = []
        async for event in stream_with_session_retry(
            "req-stream",
            state,
            make_stream,
            client=probe,
        ):
            events.append(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(probe.reset_count, 1)


class TestCrossVersionImports(unittest.TestCase):
    """Py3.8–3.14：变更模块可 import / compile。"""

    _MODULES = (
        "core.transport.http",
        "core.transport.owned",
        "core.transport.conn_retry",
        "core.transport",
        "server.retry.http_client",
        "server.retry.session_retry",
        "upstream.qwen.client",
        "upstream.qwen.account",
        "upstream.qwen.chat.chat",
    )

    def test_modules_import(self) -> None:
        for name in self._MODULES:
            mod = importlib.import_module(name)
            self.assertIsNotNone(mod)

    def test_upstream_timeout_py38_signature(self) -> None:
        tm = upstream_timeout(15.0)
        self.assertEqual(tm.total, 15.0)
        self.assertEqual(tm.connect, 10.0)

    def test_ssl_op_no_ticket_available(self) -> None:
        import ssl as ssl_mod

        ctx = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_CLIENT)
        ctx.options |= ssl_mod.OP_NO_TICKET
        self.assertTrue(ctx.options & ssl_mod.OP_NO_TICKET)


if __name__ == "__main__":
    unittest.main()
