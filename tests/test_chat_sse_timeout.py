from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from server.client.chat import iter_sse_events
from server.formats import UpstreamTimeoutError


class TestIterSseEventsTimeout(unittest.IsolatedAsyncioTestCase):
    async def test_sse_read_timeout_raises_upstream_timeout(self) -> None:
        client = MagicMock()
        session = MagicMock()

        async def _timeout_iter():
            yield b"data: {}\n"
            raise asyncio.TimeoutError()

        resp = MagicMock()
        resp.content = _timeout_iter()

        with self.assertRaises(UpstreamTimeoutError):
            async for _ in iter_sse_events(client, resp, session):
                pass


if __name__ == "__main__":
    unittest.main()
