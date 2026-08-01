from __future__ import annotations

"""上游 ClientSession 持有：ensure / reset / close，供各 upstream client 复用。"""

import asyncio
from typing import Optional

import aiohttp

from core.transport.http import reset_upstream_transport
from server.retry.http_client import client_session


class HttpTransportMixin:
    """进程级共享 connector 上的 per-client ClientSession 生命周期。"""

    _http: Optional[aiohttp.ClientSession]
    _transport_lock: asyncio.Lock

    def _init_http_transport(self) -> None:
        self._http = None
        self._transport_lock = asyncio.Lock()

    def _on_http_session_created(self, session: aiohttp.ClientSession) -> None:
        """新建 session 后钩子（如 DeepSeek rebind HIF）。"""

    def _should_recreate_http_on_reset(self) -> bool:
        """reset 后是否立即重建 session；默认由下次 ensure 惰性创建。"""
        return False

    def _ensure_http_unlocked(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = client_session()
            self._on_http_session_created(self._http)
        return self._http

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        async with self._transport_lock:
            return self._ensure_http_unlocked()

    async def ensure_http_session(self) -> aiohttp.ClientSession:
        return await self._ensure_http_session()

    async def reset_http_transport(self) -> None:
        async with self._transport_lock:
            old = self._http
            self._http = None
            await reset_upstream_transport(old)
            if self._should_recreate_http_on_reset():
                self._http = client_session()
                self._on_http_session_created(self._http)

    async def close_http_transport(self) -> None:
        """关闭 session 且不重建（shutdown 用）。"""
        async with self._transport_lock:
            old = self._http
            self._http = None
            await reset_upstream_transport(old)
