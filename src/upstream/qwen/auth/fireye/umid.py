from __future__ import annotations

"""会话级 umid：ynuf 下发优先，失败则 T2gA 形态本地合成。"""

import hashlib
import logging
import ssl
import urllib.error
import urllib.request
from typing import Final, Optional

from upstream.qwen.auth.http import get_qwen_proxy
from upstream.qwen.chat.routes import CHAT_ORIGIN, USER_AGENT

logger = logging.getLogger("rogator")

_UMID_PREFIX: Final[str] = "T2gA"
_UMID_BODY_LEN: Final[int] = 68
_UMID_ALPHABET: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-_"
)
_YNUF_HOST: Final[str] = "ph5.ynuf.aliapp.org"


def _synthetic_umid(fingerprint: str) -> str:
    seed = hashlib.sha256(fingerprint.encode()).digest()
    body_len = _UMID_BODY_LEN - len(_UMID_PREFIX) - 1
    chars = [
        _UMID_ALPHABET[seed[idx % len(seed)] % len(_UMID_ALPHABET)]
        for idx in range(body_len)
    ]
    return _UMID_PREFIX + "".join(chars) + "="


def _ynuf_post_body(*, bx_ua: str, fingerprint: str) -> bytes:
    # 对齐 fireyejs：text/plain，携带 bx-ua 与指纹。
    parts = []
    if bx_ua:
        parts.append(f"bx-ua={bx_ua}")
    if fingerprint:
        parts.append(f"fingerprint={fingerprint}")
    return "&".join(parts).encode("utf-8")


def fetch_umid_from_ynuf_sync(
    *,
    fingerprint: str,
    bx_ua: str,
) -> Optional[str]:
    url = f"https://{_YNUF_HOST}/"
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "text/plain",
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
    }
    body = _ynuf_post_body(bx_ua=bx_ua, fingerprint=fingerprint)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
            if text.startswith(_UMID_PREFIX) and len(text) >= 20:
                return text
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.debug("ynuf umid sync fetch failed: %s", exc)
    return None


async def fetch_umid_from_ynuf(
    *,
    fingerprint: str,
    bx_ua: str,
    user_agent: str = "",
) -> Optional[str]:
    """POST ph5.ynuf.aliapp.org；网络/证书失败时返回 None。"""
    try:
        import aiohttp
    except ImportError:
        return fetch_umid_from_ynuf_sync(fingerprint=fingerprint, bx_ua=bx_ua)

    ssl_ctx = ssl.create_default_context()
    url = f"https://{_YNUF_HOST}/"
    headers = {
        "User-Agent": user_agent or USER_AGENT,
        "Content-Type": "text/plain",
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
    }
    body = _ynuf_post_body(bx_ua=bx_ua, fingerprint=fingerprint)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=body,
                headers=headers,
                ssl=ssl_ctx,
                timeout=aiohttp.ClientTimeout(total=8),
                proxy=get_qwen_proxy(),
            ) as resp:
                text = (await resp.text()).strip()
                if resp.status == 200 and text.startswith(_UMID_PREFIX):
                    return text
    except Exception as exc:
        logger.debug("ynuf umid fetch failed: %s", exc)
    return None


def get_umid_token(
    fingerprint: str,
    *,
    cached: str = "",
    bx_ua: str = "",
) -> str:
    if cached:
        return cached
    if bx_ua:
        ynuf = fetch_umid_from_ynuf_sync(fingerprint=fingerprint, bx_ua=bx_ua)
        if ynuf:
            return ynuf
    return _synthetic_umid(fingerprint)
