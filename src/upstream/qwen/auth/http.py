from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

import aiohttp

from core.transport.conn_retry import run_with_connection_retry as _run_with_connection_retry
from server.formats import UpstreamConnectionError, as_upstream_connection_error
from server.retry.http_client import client_session

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
    """Qwen 兼容入口：固定 upstream=qwen。"""
    return await _run_with_connection_retry(
        label,
        func,
        upstream="qwen",
        attempts=attempts,
        delay_seconds=delay_seconds,
        transport_owner=transport_owner,
    )
