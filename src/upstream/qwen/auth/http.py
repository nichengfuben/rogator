from __future__ import annotations

"""HTTP session helpers + web Cookie jar（对齐 main.js JO / 抓包）。"""

import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Final, Optional, TypeVar

import aiohttp

from core.transport.conn_retry import run_with_connection_retry as _run_with_connection_retry
from server.formats import UpstreamConnectionError, as_upstream_connection_error
from server.retry.http_client import client_session

T = TypeVar("T")

HASH_FIELDS: Final[list] = [
    "ssxmod_itna",
    "ssxmod_itna2",
    "bx-umidtoken",
    "bx-ua",
]
_COOKIE_META_KEYS: Final[frozenset] = frozenset({"fingerprint", "timestamp"})
_CNA_CHARS: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
_HARVEST_COOKIE_KEYS: Final[frozenset] = frozenset(
    {
        "acw_tc",
        "x-ap",
        "sca",
        "cna",
        "aui",
        "cnaui",
        "atpsida",
        "tfstk",
        "isg",
        "xlly_s",
        "ssxmod_itna",
        "ssxmod_itna2",
        "qwen-theme",
        "qwen-locale",
        "qwen-thinking_mode",
        "token",
    }
)


def _rand_cna() -> str:
    return "".join(secrets.choice(_CNA_CHARS) for _ in range(24))


def generate_cookies(
    fingerprint: str = "",
    *,
    user_id: str = "",
    thinking_mode: str = "Fast",
) -> Dict[str, Any]:
    # main.js JO(qwen-theme/locale)；抓包另有 thinking_mode / xlly_s / cna / aui
    jar: Dict[str, Any] = {
        "qwen-theme": "dark",
        "qwen-locale": "zh-CN",
        "qwen-thinking_mode": thinking_mode or "Fast",
        "xlly_s": "1",
        "x-ap": "cn-hongkong",
        "cna": _rand_cna(),
        "sca": secrets.token_hex(4),
    }
    if user_id:
        jar["cnaui"] = user_id
        jar["aui"] = user_id
    if fingerprint:
        jar["fingerprint"] = fingerprint
    return jar


def merge_session_cookies(
    token: str, extra: Optional[Dict[str, Any]] = None, *, user_id: str = ""
) -> Dict[str, str]:
    # 以 extra 为底稿补齐缺省项；避免每次调用都重新 roll cna/sca（建聊与 completion 须同会话）。
    merged: Dict[str, Any] = dict(extra) if extra else {}
    defaults = generate_cookies(user_id=user_id)
    for key, value in defaults.items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value
    if token:
        merged["token"] = token
    if user_id:
        merged.setdefault("cnaui", user_id)
        merged.setdefault("aui", user_id)
    return {
        str(k): str(v)
        for k, v in merged.items()
        if k not in _COOKIE_META_KEYS and v not in (None, "")
    }


def sync_cookie_store(store: Dict[str, str], merged: Dict[str, str]) -> None:
    """把 merge 结果（含首次生成的 cna/sca）回写账号级持久 jar。"""
    for key, value in merged.items():
        if key not in _COOKIE_META_KEYS and value not in (None, ""):
            store[key] = str(value)


def build_cookie_string(cookies: Optional[Dict[str, Any]]) -> str:
    if not cookies:
        return ""
    return "; ".join(
        f"{key}={value}"
        for key, value in cookies.items()
        if key not in _COOKIE_META_KEYS and value not in {None, ""}
    )


def absorb_response_cookies(jar: Dict[str, str], response: Any) -> None:
    """把上游 Set-Cookie 并入 jar（acw_tc / cna / tfstk / ssxmod…）。"""
    cookies = getattr(response, "cookies", None)
    if not cookies:
        return
    for key, morsel in cookies.items():
        name = str(key)
        if name in _HARVEST_COOKIE_KEYS or name.startswith("ssxmod"):
            value = getattr(morsel, "value", None)
            if value not in (None, ""):
                jar[name] = str(value)


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
