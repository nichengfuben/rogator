"""共享上游 HTTP 传输：连接池、TLS、超时、ClientSession 生命周期。"""

from __future__ import annotations

import asyncio
import logging
import ssl
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional, TypeVar

import aiohttp

from server.formats import UpstreamTimeoutError, as_upstream_connection_error

logger = logging.getLogger("rogator")

T = TypeVar("T")

_POOL_LIMIT = 200
_POOL_LIMIT_PER_HOST = 20
_POOL_KEEPALIVE_TIMEOUT = 30
_POOL_CONNECT_TIMEOUT = 10.0

_RESET_COOLDOWN_SECONDS = 2.0

_connector: Optional[aiohttp.TCPConnector] = None
_ssl_context: Optional[ssl.SSLContext] = None


def get_upstream_ssl_context() -> ssl.SSLContext:
    """不校验证书；关闭 TLS session ticket，避免连接池复用失效 ticket。"""
    global _ssl_context
    if _ssl_context is not None:
        return _ssl_context
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # 老 OpenSSL / Windows schannel 不暴露 OP_NO_TICKET：忽略 AttributeError
    try:
        ctx.options |= ssl.OP_NO_TICKET
    except AttributeError:
        logger.debug("ssl.OP_NO_TICKET not supported on this OpenSSL build; skipping")
    _ssl_context = ctx
    return _ssl_context


def make_connector() -> aiohttp.TCPConnector:
    """进程级共享 TCPConnector；避免多 ClientSession 各自建池。"""
    global _connector
    if _connector is not None and not _connector.closed:
        return _connector
    _connector = aiohttp.TCPConnector(
        ssl=get_upstream_ssl_context(),
        limit=_POOL_LIMIT,
        limit_per_host=_POOL_LIMIT_PER_HOST,
        keepalive_timeout=_POOL_KEEPALIVE_TIMEOUT,
        force_close=False,
    )
    return _connector


async def close_shared_connector() -> None:
    global _connector, _ssl_context
    conn = _connector
    _connector = None
    _ssl_context = None
    if conn is None or conn.closed:
        return
    await conn.close()


async def reset_upstream_transport(
    session: Optional[aiohttp.ClientSession] = None,
) -> None:
    """关闭指定 ClientSession，供 transport 重试前调用。

    注意：不再关闭共享 connector，避免其他并发 client 的 session 被连带失效。
    共享 connector 仅在进程 shutdown 时由 ``close_shared_connector()`` 关闭。
    """
    if session is not None and not session.closed:
        await session.close()


def build_connector(*, ssl: bool = False) -> aiohttp.TCPConnector:
    """兼容旧调用；上游 HTTPS 请用 ``make_connector()``。"""
    if ssl:
        return aiohttp.TCPConnector(ssl=get_upstream_ssl_context())
    return make_connector()


def client_timeout(
    total: Optional[float] = None,
    sock_read: Optional[float] = None,
    *,
    connect: Optional[float] = None,
) -> aiohttp.ClientTimeout:
    conn = _POOL_CONNECT_TIMEOUT if connect is None else connect
    return aiohttp.ClientTimeout(
        total=total,
        connect=conn,
        sock_connect=conn,
        sock_read=sock_read if sock_read is not None else total,
    )


def upstream_timeout(
    total: float,
    *,
    connect: float = _POOL_CONNECT_TIMEOUT,
    sock_read: Optional[float] = None,
) -> aiohttp.ClientTimeout:
    return client_timeout(total, sock_read=sock_read, connect=connect)


async def request_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
) -> tuple[int, Any]:
    async with session.request(
        method, url, headers=headers, json=json, data=data, timeout=timeout
    ) as resp:
        try:
            body = await resp.json(content_type=None)
        except Exception:
            body = await resp.text()
        return resp.status, body


# ---------------------------------------------------------------------------
# 跨 Python 3.8–3.14、Win/Linux/macOS 的小兼容层
# ---------------------------------------------------------------------------


def removeprefix(text: str, prefix: str) -> str:
    """``str.removeprefix``（3.9+）在 3.8 上的回退。"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def removesuffix(text: str, suffix: str) -> str:
    """``str.removesuffix``（3.9+）在 3.8 上的回退。"""
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


if sys.version_info >= (3, 10):
    from contextlib import aclosing
else:

    @asynccontextmanager
    async def aclosing(thing: T) -> AsyncIterator[T]:
        try:
            yield thing
        finally:
            aclose = getattr(thing, "aclose", None)
            if aclose is not None:
                await aclose()


# ---------------------------------------------------------------------------
# 上游连接级短重试
# ---------------------------------------------------------------------------


def reraise_transport_error(
    exc: BaseException,
    *,
    upstream: str,
    timeout_message: str = "",
) -> None:
    """将超时/连接类异常映射为 session_retry 可识别类型后抛出。"""
    if isinstance(exc, asyncio.TimeoutError):
        msg = timeout_message or "{0} upstream timeout".format(upstream)
        raise UpstreamTimeoutError(msg) from exc
    conn_err = as_upstream_connection_error(exc, upstream=upstream)
    if conn_err is not None:
        raise conn_err from exc
    raise exc


async def run_with_connection_retry(
    label: str,
    func: Callable[[], Awaitable[T]],
    *,
    upstream: str,
    attempts: int = 2,
    delay_seconds: float = 0.6,
    transport_owner: Optional[Any] = None,
) -> T:
    """对瞬时连接失败做少量重试，并在重试前 reset transport。"""
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            conn_err = as_upstream_connection_error(exc, upstream=upstream)
            if conn_err is None or attempt >= attempts:
                raise conn_err or exc
            reset = getattr(transport_owner, "reset_http_transport", None)
            if callable(reset):
                await reset()
            logger.warning(
                "%s %s connection failed (retry %d/%d): %s",
                upstream,
                label,
                attempt,
                attempts - 1,
                conn_err.message,
            )
            await asyncio.sleep(delay_seconds * attempt)
    raise RuntimeError("{0} {1} retry exhausted".format(upstream, label))


# ---------------------------------------------------------------------------
# 进程级共享 connector 上的 per-client ClientSession 生命周期
# ---------------------------------------------------------------------------


def session_is_usable(session: aiohttp.ClientSession | None) -> bool:
    """判断 ClientSession 是否仍可安全发起请求。"""
    if session is None or session.closed:
        return False
    connector = session.connector
    return connector is not None and not connector.closed


class HttpTransportMixin:
    """进程级共享 connector 上的 per-client ClientSession 生命周期。"""

    _http: Optional[aiohttp.ClientSession]
    _transport_lock_holder: Optional[asyncio.Lock]

    @property
    def _transport_lock(self) -> asyncio.Lock:
        if self._transport_lock_holder is None:
            self._transport_lock_holder = asyncio.Lock()
        return self._transport_lock_holder

    def _init_http_transport(self) -> None:
        self._http = None
        self._transport_lock_holder = None
        self._last_reset_mono: float = 0.0

    def _on_http_session_created(self, session: aiohttp.ClientSession) -> None:
        """新建 session 后钩子（如 DeepSeek rebind HIF）。"""

    def _should_recreate_http_on_reset(self) -> bool:
        """reset 后是否立即重建 session；默认由下次 ensure 惰性创建。"""
        return False

    def _client_session_kwargs(self) -> dict:
        """子类可覆盖以向 client_session 传递额外参数（如 use_env_proxy）。"""
        return {}

    def _ensure_http_unlocked(self) -> aiohttp.ClientSession:
        if not session_is_usable(self._http):
            from server.retry.http_client import client_session
            self._http = client_session(**self._client_session_kwargs())
            self._on_http_session_created(self._http)
        return self._http

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        async with self._transport_lock:
            return self._ensure_http_unlocked()

    async def ensure_http_session(self) -> aiohttp.ClientSession:
        return await self._ensure_http_session()

    async def reset_http_transport(self) -> None:
        """软重置 transport：关闭当前 ClientSession 并丢弃引用。

        使用 ``connector_owner=False`` 的共享 connector 不会被关闭；
        其它 client 持有的 session 实例不受影响。

        Cooldown 期内的重复 reset 仅丢弃引用而不关闭 session，
        避免并发请求因级联 reset 触发 ``_timeout_ceil_threshold`` 等 stale session 异常。
        """
        async with self._transport_lock:
            now = time.monotonic()
            in_cooldown = (now - self._last_reset_mono) < _RESET_COOLDOWN_SECONDS
            old = self._http
            self._http = None
            self._last_reset_mono = now
            # 始终关闭旧 session 防止泄漏；cooldown 仅跳过立即重建
            if old is not None and not old.closed:
                await reset_upstream_transport(old)
            if not in_cooldown and self._should_recreate_http_on_reset():
                from server.retry.http_client import client_session
                self._http = client_session(**self._client_session_kwargs())
                self._on_http_session_created(self._http)

    async def close_http_transport(self) -> None:
        """关闭 session 且不重建（shutdown 用）。"""
        async with self._transport_lock:
            old = self._http
            self._http = None
            await reset_upstream_transport(old)