from __future__ import annotations

"""aiohttp 客户端：读取 HTTP(S)/SOCKS 代理环境变量。"""

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp

from core.transport.http import make_connector

logger = logging.getLogger("rogator")

_PROXY_PAIRS = (
    ("HTTPS_PROXY", "https_proxy"),
    ("HTTP_PROXY", "http_proxy"),
    ("ALL_PROXY", "all_proxy"),
    ("NO_PROXY", "no_proxy"),
)


def sync_proxy_env() -> None:
    """补齐大小写成对变量，便于 urllib / aiohttp 读取。"""
    for upper, lower in _PROXY_PAIRS:
        upper_val = os.environ.get(upper, "").strip()
        lower_val = os.environ.get(lower, "").strip()
        if upper_val and not lower_val:
            os.environ[lower] = upper_val
        elif lower_val and not upper_val:
            os.environ[upper] = lower_val


def active_proxy_url() -> Optional[str]:
    sync_proxy_env()
    for upper, lower in _PROXY_PAIRS[:3]:
        for key in (upper, lower):
            val = os.environ.get(key, "").strip()
            if val:
                return val
    return None


def _redact_proxy(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://***:***@{host}{port}"
    except Exception:
        pass
    return url


def _socks_connector(proxy_url: str) -> Optional[aiohttp.BaseConnector]:
    try:
        from aiohttp_socks import ProxyConnector
    except ImportError:
        logger.warning(
            "SOCKS 代理已配置但缺少 aiohttp-socks，请执行: pip install aiohttp-socks"
        )
        return None
    try:
        return ProxyConnector.from_url(proxy_url)
    except Exception as exc:
        logger.warning("SOCKS 代理连接器创建失败: %s", exc)
        return None


def client_session(**kwargs: Any) -> aiohttp.ClientSession:
    """创建尊重环境变量代理的 ClientSession。"""
    if "connector" not in kwargs:
        proxy = active_proxy_url()
        if proxy and proxy.lower().startswith("socks"):
            connector = _socks_connector(proxy)
            if connector is not None:
                kwargs["connector"] = connector
                return aiohttp.ClientSession(**kwargs)
        kwargs["connector"] = make_connector()
    if "trust_env" not in kwargs:
        kwargs["trust_env"] = True
    return aiohttp.ClientSession(**kwargs)


def init_http_proxy_from_env(log: Optional[logging.Logger] = None) -> Optional[str]:
    """启动时同步并记录代理环境变量。"""
    sync_proxy_env()
    proxy = active_proxy_url()
    sink = log or logger
    if proxy:
        sink.info("HTTP proxy enabled (env): %s", _redact_proxy(proxy))
    else:
        sink.debug("HTTP proxy disabled (env not set)")
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    if no_proxy.strip():
        sink.debug("NO_PROXY: %s", no_proxy.strip())
    return proxy
