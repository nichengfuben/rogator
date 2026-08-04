from __future__ import annotations

"""anyio run_blocking / async headers 基础测试。"""

import asyncio
import unittest

from core.transport.blocking import fireye_limiter, io_limiter, run_blocking
from upstream.qwen.auth.crypto import build_headers, build_headers_async, reset_baxia_runtime
from upstream.qwen.auth.fireye import reset_session


class TestConcurrencyBlocking(unittest.TestCase):
    def test_run_blocking_offloads_sync_fn(self) -> None:
        async def _run() -> int:
            return await run_blocking(lambda: 42, limiter=io_limiter())

        self.assertEqual(asyncio.run(_run()), 42)

    def test_fireye_async_headers_match_sync_shape(self) -> None:
        reset_baxia_runtime()
        reset_session()

        async def _async_ua() -> str:
            headers = await build_headers_async(
                "token",
                api_path="/api/v2/chat/completions",
                chat_id="f07fc0a2-f718-4076-8f7d-56834a8013bb",
            )
            return headers.get("bx-ua", "")

        sync_headers = build_headers(
            "token",
            api_path="/api/v2/chat/completions",
            chat_id="f07fc0a2-f718-4076-8f7d-56834a8013bb",
        )
        async_ua = asyncio.run(_async_ua())
        self.assertTrue(sync_headers["bx-ua"].startswith("231!"))
        self.assertTrue(async_ua.startswith("231!"))
        self.assertIn("bx-umidtoken", sync_headers)

    def test_fireye_async_does_not_block_event_loop(self) -> None:
        reset_baxia_runtime()
        reset_session()

        async def _probe() -> None:
            async def ticker() -> str:
                await asyncio.sleep(0.01)
                return "ok"

            tick = asyncio.create_task(ticker())
            await build_headers_async(
                "token",
                api_path="/api/v2/chat/completions",
            )
            self.assertEqual(await asyncio.wait_for(tick, timeout=0.5), "ok")

        asyncio.run(_probe())

    def test_fireye_limiter_singleton(self) -> None:
        self.assertIs(fireye_limiter(), fireye_limiter())


if __name__ == "__main__":
    unittest.main()
