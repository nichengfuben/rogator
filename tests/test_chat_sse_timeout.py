from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from upstream.qwen.chat.chat import iter_sse_events
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

    async def test_single_chunk_multiple_sse_lines(self) -> None:
        client = MagicMock()
        session = MagicMock()
        chunk = (
            'data: {"response.created":{"response_id":"rid-1"}}\n\n'
            'data: {"choices":[{"delta":{"content":"hi","phase":"answer","status":"typing"}}],'
            '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}\n\n'
        )

        async def _one_shot():
            yield chunk.encode("utf-8")

        resp = MagicMock()
        resp.content = _one_shot()
        events = [e async for e in iter_sse_events(client, resp, session)]
        types = [e.get("type") for e in events]
        self.assertEqual(types, ["response_created", "answer"])
        self.assertIn("usage", events[1])


if __name__ == "__main__":
    unittest.main()
