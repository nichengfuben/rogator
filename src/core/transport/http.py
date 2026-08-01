from __future__ import annotations

"""共享上游 HTTP 传输：连接池、TLS、超时（对齐 Provider-Evo provider-core）。"""

import ssl
from typing import Any, Mapping, Optional

import aiohttp

# 与 provider-core template [http_pool] 缺省一致
_POOL_LIMIT = 200
_POOL_LIMIT_PER_HOST = 20
_POOL_KEEPALIVE_TIMEOUT = 30
_POOL_CONNECT_TIMEOUT = 10.0

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
    ctx.options |= ssl.OP_NO_TICKET
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
        enable_cleanup_closed=True,
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


async def reset_upstream_transport(session: Optional[aiohttp.ClientSession] = None) -> None:
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
