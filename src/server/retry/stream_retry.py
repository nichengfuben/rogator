from __future__ import annotations

"""流式换号重试：失败则整段重开。"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from server.formats import (
    BaxiaSmBlockedError,
    PayloadTooLargeError,
    TokenExpiredError,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamWafBlockedError,
    UpstreamChatNotFoundError,
)

logger = logging.getLogger("rogator")


async def _close_async_generator(
    agen: Optional[AsyncGenerator[dict, None]],
) -> None:
    """显式关闭嵌套 async generator，避免 shutdown/断连时 aclose 未 await。"""
    if agen is None:
        return
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except StopAsyncIteration:
        return
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "already running" in msg or "cannot reuse" in msg:
            return
        raise
    except Exception:
        logger.debug("async generator aclose failed", exc_info=True)


async def _consume_stream_once(
    inner: AsyncGenerator[dict, None],
) -> AsyncGenerator[dict, None]:
    async for event in inner:
        yield event


async def _reset_stream(inner: Optional[AsyncGenerator[dict, None]]) -> None:
    await _close_async_generator(inner)


async def _handle_stream_retry_error(
    req_id: str,
    state: Any,
    exc: BaseException,
    retries: int,
    limit: int,
    client: Any | None,
    *,
    model: str | None = None,
) -> int:
    from server.retry.session_retry import _dispatch_session_retry

    return await _dispatch_session_retry(
        req_id, state, exc, retries=retries, limit=limit, client=client, model=model,
    )


async def stream_with_session_retry(
    req_id: str,
    state: Any,
    make_stream: Callable[[], AsyncGenerator[dict, None]],
    *,
    max_retry: Optional[int] = None,
    client: Any | None = None,
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """流式换号重试：失败则整段重开。"""
    from server.config import CONFIG

    limit = CONFIG.max_retry_on_error if max_retry is None else max_retry
    retries = 0
    inner: Optional[AsyncGenerator[dict, None]] = None

    try:
        while True:
            inner = make_stream()
            try:
                async for event in _consume_stream_once(inner):
                    yield event
                return
            except asyncio.CancelledError:
                await _reset_stream(inner)
                inner = None
                raise
            except (
                TokenExpiredError,
                BaxiaSmBlockedError,
                UpstreamWafBlockedError,
                PayloadTooLargeError,
                UpstreamTimeoutError,
                UpstreamConnectionError,
                UpstreamChatNotFoundError,
            ) as exc:
                await _reset_stream(inner)
                inner = None
                retries = await _handle_stream_retry_error(
                    req_id, state, exc, retries, limit, client, model=model,
                )
            except Exception:
                await _reset_stream(inner)
                inner = None
                raise
    finally:
        await _reset_stream(inner)
