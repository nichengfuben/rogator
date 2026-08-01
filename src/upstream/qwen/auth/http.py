from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

import aiohttp

from server.formats import UpstreamConnectionError, as_upstream_connection_error
from server.retry.http_client import client_session

logger = logging.getLogger("rogator")

T = TypeVar("T")


def create_http_session() -> aiohttp.ClientSession:
    """Create one Qwen HTTP session respecting proxy env when present."""
    return client_session()


@asynccontextmanager
async def borrow_http_session(
    shared: aiohttp.ClientSession | None = None,
) -> AsyncIterator[aiohttp.ClientSession]:
    if shared is not None and not shared.closed:
        yield shared
        return
    session = create_http_session()
    try:
        yield session
    finally:
        await session.close()


def map_connection_error(exc: BaseException) -> UpstreamConnectionError | None:
    return as_upstream_connection_error(exc, upstream="qwen")


async def run_with_connection_retry(
    label: str,
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    delay_seconds: float = 0.6,
    transport_owner: Optional[Any] = None,
) -> T:
    """Retry transient Qwen connection failures a small number of times."""
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            conn_err = map_connection_error(exc)
            if conn_err is None or attempt >= attempts:
                raise conn_err or exc
            reset = getattr(transport_owner, "reset_http_transport", None)
            if callable(reset):
                await reset()
            logger.warning(
                "Qwen %s connection failed (retry %d/%d): %s",
                label, attempt, attempts - 1, conn_err.message,
            )
            await asyncio.sleep(delay_seconds * attempt)
    raise RuntimeError("Qwen {0} retry exhausted".format(label))
