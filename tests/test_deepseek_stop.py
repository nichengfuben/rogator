from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.deepseek.lib.adapter.client import DeepseekClient


class TestDeepSeekStopStream(unittest.IsolatedAsyncioTestCase):
    async def test_stop_upstream_generation_calls_sessapi(self) -> None:
        client = DeepseekClient()
        client._session = MagicMock()
        with patch(
            "upstream.deepseek.lib.session.sessapi.stop_stream",
            new_callable=AsyncMock,
            return_value=True,
        ) as stop:
            ok = await client.stop_upstream_generation("tok", "sess-1", 99)
        self.assertTrue(ok)
        stop.assert_awaited_once_with(client._session, "tok", "sess-1", "99")

    async def test_abort_on_cancel_skips_without_message_id(self) -> None:
        client = DeepseekClient()
        client.stop_upstream_generation = AsyncMock(return_value=True)
        parser = MagicMock(message_id=None)
        await client._abort_upstream_on_cancel("tok", "sess-1", parser)
        client.stop_upstream_generation.assert_not_awaited()

    async def test_abort_on_cancel_invokes_stop(self) -> None:
        client = DeepseekClient()
        client.stop_upstream_generation = AsyncMock(return_value=True)
        parser = MagicMock(message_id=42)
        await client._abort_upstream_on_cancel("tok", "sess-1", parser)
        client.stop_upstream_generation.assert_awaited_once_with("tok", "sess-1", 42)

    async def test_do_complete_aborts_on_cancel(self) -> None:
        client = DeepseekClient()
        client._abort_upstream_on_cancel = AsyncMock()

        async def _boom(*_a, **_k):
            yield "partial"
            raise asyncio.CancelledError()

        with patch.object(client, "_stream_and_continue", _boom), patch(
            "upstream.deepseek.lib.adapter.client.prepare_full_request",
            new_callable=AsyncMock,
            return_value=(
                {"token": "tok", "prompt": "hi"},
                "sess-9",
                "leim",
                "dliq",
                {},
                MagicMock(message_id=7),
            ),
        ):
            gen = client._do_complete(MagicMock(), [], "deepseek-v4-pro", True)
            with self.assertRaises(asyncio.CancelledError):
                async for _ in gen:
                    pass

        client._abort_upstream_on_cancel.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
