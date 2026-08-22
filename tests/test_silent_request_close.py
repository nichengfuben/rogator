"""silent_request 即使中途异常也必须关闭 ClientSession（Unclosed client session 回归）。"""
from __future__ import annotations

import asyncio

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_silent_request_closes_session_on_request_error():
    from upstream.qwen.auth.report.core import silent_request

    closed: list[bool] = []
    orig_init = aiohttp.ClientSession.__init__

    def tracking_init(self, *a, **k):
        orig_init(self, *a, **k)
        orig_close = self.close

        async def close_wrap():
            closed.append(True)
            await orig_close()

        self.close = close_wrap  # type: ignore[method-assign]

    import upstream.qwen.auth.report.core as core

    class _BoomSession(aiohttp.ClientSession):
        def request(self, *a, **k):
            raise RuntimeError("boom")

    # 让 silent_request 内部创建的 session 变成会炸的版本，并通过 tracking_init 观察 close
    orig_cls = aiohttp.ClientSession

    class _TrackingSession(_BoomSession):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _orig = self.close

            async def _close():
                closed.append(True)
                await _orig()

            self.close = _close  # type: ignore[method-assign]

    aiohttp.ClientSession = _TrackingSession  # type: ignore[assignment]
    try:
        await silent_request(None, "POST", "https://example.invalid/x", json_body={})
    finally:
        aiohttp.ClientSession = orig_cls  # type: ignore[assignment]

    assert closed == [True]


@pytest.mark.asyncio
async def test_silent_request_closes_session_when_cancelled():
    from upstream.qwen.auth.report.core import silent_request

    closed: list[bool] = []
    orig_cls = aiohttp.ClientSession

    class _HangSession(aiohttp.ClientSession):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _orig = self.close

            async def _close():
                closed.append(True)
                await _orig()

            self.close = _close  # type: ignore[method-assign]

        def request(self, *a, **k):
            class _Ctx:
                async def __aenter__(self):
                    try:
                        await asyncio.sleep(60)
                    finally:
                        raise asyncio.CancelledError

                async def __aexit__(self, *exc):
                    return False

            return _Ctx()

    aiohttp.ClientSession = _HangSession  # type: ignore[assignment]
    try:
        task = asyncio.create_task(silent_request(None, "GET", "https://example.invalid/x"))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # 预期：取消传播
        except Exception:
            pass
        # 给 close() 里的 await 一个调度机会
        await asyncio.sleep(0.1)
    finally:
        aiohttp.ClientSession = orig_cls  # type: ignore[assignment]

    assert closed == [True]
