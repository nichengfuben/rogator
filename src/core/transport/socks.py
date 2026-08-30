"""SOCKS 代理支持：模块顶部强制 ``import aiohttp_socks``，缺失即 ``ImportError``。

启动期 fail-fast，避免运行时发现缺包导致请求静默走直连（之前在 ``client_session``
里 ``try: from aiohttp_socks import ProxyConnector`` 走 warning + return None，
配置 SOCKS 代理却没装包时只是请求退化为直连，运维难以察觉）。
"""

from __future__ import annotations

from typing import Optional

import aiohttp
from aiohttp_socks import ProxyConnector  # noqa: F401  顶层导入：启动期缺失即失败


def socks_connector(proxy_url: str) -> Optional[aiohttp.BaseConnector]:
    """把 SOCKS 代理 URL 解析成 ``aiohttp_socks.ProxyConnector``。"""
    try:
        return ProxyConnector.from_url(proxy_url)
    except Exception as exc:  # noqa: BLE001
        # 协议层面支持 socks4/socks5/socks5h；URL 解析失败时由调用方决定降级
        raise RuntimeError(f"SOCKS 代理连接器创建失败: {exc}") from exc


__all__ = ["ProxyConnector", "socks_connector"]