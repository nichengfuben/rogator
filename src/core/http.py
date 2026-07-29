from __future__ import annotations

"""Shared HTTP helpers (SSL off by default for upstream gateways)."""

from typing import Any, Mapping, Optional

import aiohttp


def build_connector(*, ssl: bool = False) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(ssl=ssl)


def client_timeout(total: Optional[float] = None, sock_read: Optional[float] = None) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=total, sock_read=sock_read)


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
