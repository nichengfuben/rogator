from __future__ import annotations

import asyncio
import ssl
import unittest
from unittest.mock import patch

import aiohttp

from core.transport.http import (
    close_shared_connector,
    get_upstream_ssl_context,
    make_connector,
    reset_upstream_transport,
    upstream_timeout,
)
from core.transport.owned import HttpTransportMixin
from server.retry.http_client import client_session


class _OwnedClient(HttpTransportMixin):
    def __init__(self) -> None:
        self._init_http_transport()


class TestUpstreamTransport(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    def test_ssl_context_disables_session_ticket(self) -> None:
        ctx = get_upstream_ssl_context()
        self.assertFalse(ctx.check_hostname)
        self.assertEqual(ctx.verify_mode, ssl.CERT_NONE)
        self.assertTrue(ctx.options & ssl.OP_NO_TICKET)

    async def test_make_connector_is_singleton(self) -> None:
        first = make_connector()
        second = make_connector()
        self.assertIs(first, second)

    def test_upstream_timeout_sets_connect(self) -> None:
        tm = upstream_timeout(600.0)
        self.assertEqual(tm.total, 600.0)
        self.assertEqual(tm.connect, 10.0)
        self.assertEqual(tm.sock_connect, 10.0)
        self.assertEqual(tm.sock_read, 600.0)

    async def test_reset_upstream_transport_keeps_shared_connector(self) -> None:
        session = client_session()
        other = client_session()
        first_conn = make_connector()
        await reset_upstream_transport(session)
        second_conn = make_connector()
        self.assertIs(first_conn, second_conn)
        self.assertFalse(first_conn.closed)
        self.assertFalse(other.closed)
        await other.close()
        await close_shared_connector()

    async def test_client_session_uses_shared_connector(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            session = client_session()
            try:
                self.assertIs(session.connector, make_connector())
            finally:
                await session.close()
                await close_shared_connector()

    async def test_mixin_reset_keeps_peer_session_alive(self) -> None:
        a = _OwnedClient()
        b = _OwnedClient()
        sa = await a.ensure_http_session()
        sb = await b.ensure_http_session()
        self.assertIs(sa.connector, sb.connector)
        await a.reset_http_transport()
        self.assertTrue(sa.closed)
        self.assertFalse(sb.closed)
        self.assertFalse(make_connector().closed)
        await b.close_http_transport()

    async def test_mixin_reset_closes_orphan_session(self) -> None:
        client = _OwnedClient()
        session = await client.ensure_http_session()
        self.assertFalse(session.closed)
        await client.reset_http_transport()
        self.assertTrue(session.closed)
        self.assertIsNone(client._http)


if __name__ == "__main__":
    unittest.main()
