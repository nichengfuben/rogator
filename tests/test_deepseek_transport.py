from __future__ import annotations

"""DeepSeek transport 重建与 HIF rebind。"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.session.store import SessionStoreMeta
from core.transport.http import close_shared_connector
from server.retry.http_client import client_session
from upstream.deepseek.client import DeepSeekClient
from upstream.deepseek.lib.adapter.client import DeepseekClient
from upstream.deepseek.lib.adapter.helpers.pmtutil import Account as DsAccount


class TestDeepSeekTransport(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    async def test_rebind_http_session_updates_hif_managers(self) -> None:
        inner = DeepseekClient()
        old = client_session()
        await inner.init_immediate(
            old, accounts=[DsAccount(username="a@test.com", password="pw")],
        )
        mgr = inner._hif_managers["a@test.com"]  # noqa: SLF001
        self.assertIs(mgr._session, old)  # noqa: SLF001
        new = client_session()
        inner.rebind_http_session(new)
        self.assertIs(inner._session, new)  # noqa: SLF001
        self.assertIs(mgr._session, new)  # noqa: SLF001
        await old.close()
        await new.close()

    async def test_reset_http_transport_rebinds_without_startup_done(self) -> None:
        with patch(
            "core.session.pool.load_upstream_sessions",
            return_value=([], SessionStoreMeta()),
        ):
            client = DeepSeekClient()
        http = client_session()
        inner = DeepseekClient()
        await inner.init_immediate(
            http, accounts=[DsAccount(username="b@test.com", password="pw")],
        )
        client._http = http
        client._inner = inner
        client._startup_done = False
        mgr = inner._hif_managers["b@test.com"]  # noqa: SLF001
        old_http = http
        await client.reset_http_transport()
        self.assertIsNotNone(client._http)
        self.assertFalse(client._http.closed)
        self.assertIsNot(client._http, old_http)
        self.assertIs(inner._session, client._http)  # noqa: SLF001
        self.assertIs(mgr._session, client._http)  # noqa: SLF001
        self.assertFalse(old_http.closed)
        await client.shutdown()

    async def test_login_retries_after_session_closed(self) -> None:
        with patch(
            "core.session.pool.load_upstream_sessions",
            return_value=([], SessionStoreMeta()),
        ):
            client = DeepSeekClient()
        account = MagicMock()
        account.username = "c@test.com"
        account.password = "pw"
        calls = {"n": 0}

        async def _fake_login_once(_account):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Session is closed")
            return MagicMock(token="tok", user_id="uid")

        client._login_once = _fake_login_once  # type: ignore[method-assign]
        client.reset_http_transport = AsyncMock()  # type: ignore[method-assign]
        result = await client._perform_login(account)
        self.assertIsNotNone(result)
        self.assertEqual(calls["n"], 2)
        client.reset_http_transport.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
