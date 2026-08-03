from __future__ import annotations

"""会话级 umid：优先 ynuf，失败则 T2gA 形态本地合成。"""

import hashlib
import logging
from typing import Final, Optional

logger = logging.getLogger("rogator")

_UMID_PREFIX: Final[str] = "T2gA"
_UMID_BODY_LEN: Final[int] = 64
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


async def fetch_umid_from_ynuf(
    *,
    fingerprint: str,
    bx_ua: str,
    user_agent: str,
) -> Optional[str]:
    """POST ph5.ynuf.aliapp.org；网络/证书失败时返回 None。"""
    try:
        import aiohttp
        import ssl
    except ImportError:
        return None

    ssl_ctx = ssl.create_default_context()
    url = f"https://{_YNUF_HOST}/"
    headers = {
        "User-Agent": user_agent,
        "Content-Type": "text/plain",
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
    }
    body = f"bx-ua={bx_ua}&fingerprint={fingerprint}".encode()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=body,
                headers=headers,
                ssl=ssl_ctx,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                text = (await resp.text()).strip()
                if resp.status == 200 and text.startswith(_UMID_PREFIX):
                    return text
    except Exception as exc:
        logger.debug("ynuf umid fetch failed: %s", exc)
    return None


def get_umid_token(fingerprint: str, *, cached: str = "") -> str:
    if cached:
        return cached
    return _synthetic_umid(fingerprint)
