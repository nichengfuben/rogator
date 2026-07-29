from __future__ import annotations

"""Cryptographic, header, cookie, and error helpers.

Merged from: crypto.py, headers.py, cookies.py, password.py, bxumid.py, errors.py
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import struct
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Optional

# ---------------------------------------------------------------------------
# Constants (from endpoints.py, used locally)
# ---------------------------------------------------------------------------

BAXIA_VERSION: Final[str] = "0.0.3"
CUSTOM_BASE64_CHARS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
APP_VERSION: Final[str] = "0.2.64"
WEB_VERSION: Final[str] = "0.2.9"
BASE_URL: Final[str] = "https://chat.qwen.ai"
CHAT_ORIGIN: Final[str] = "https://chat.qwen.ai"
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
SEC_CH_UA: Final[str] = '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"'
SEC_CH_UA_PLATFORM: Final[str] = '"macOS"'


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WafBlockedError(RuntimeError):
    """Raised when the upstream returns an HTML block page instead of SSE."""


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return the SHA-256 digest used by the Qwen web login flow."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BXUMID
# ---------------------------------------------------------------------------


def validate_bxumidtoken(token: str) -> bool:
    """Return whether the token matches the expected compact format."""
    return bool(token and re.fullmatch(r"(?:T2gA)?[A-Za-z0-9+/=]{20,}", token))


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------

HASH_FIELDS: Final[list] = [
    "ssxmod_itna",
    "ssxmod_itna2",
    "bx-umidtoken",
    "bx-ua",
]


def generate_cookies(fingerprint: str) -> Dict[str, Any]:
    """Return a compatibility cookie mapping."""
    return {
        "ssxmod_itna": "",
        "ssxmod_itna2": "",
        "fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# Crypto / Baxia
# ---------------------------------------------------------------------------


def generate_device_id() -> str:
    """Return a browser-like device identifier."""
    return uuid.uuid4().hex


def collect_fingerprint_data() -> str:
    """Build the compact fingerprint string used for Baxia headers."""
    device_id = generate_device_id()
    fields = [
        device_id,
        "1.0.0",
        "web",
        "Chrome",
        "148.0.0.0",
        "zh-CN",
        "Asia/Shanghai",
        "1920x1080",
        "24",
        "Win32",
        "macOS",
        "Apple GPU",
        "Apple GPU",
        "desktop",
        "arena",
        "stable",
    ]
    return "^".join(fields)


def generate_fingerprint() -> str:
    """Return a stable-format fingerprint string."""
    return collect_fingerprint_data()


def _encode_payload(text: str) -> str:
    """Encode a Baxia payload with URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def generate_bxua(fingerprint: str) -> str:
    """Build the ``bx-ua`` header value."""
    payload = f"{fingerprint}|{int(time.time() * 1000)}|{BAXIA_VERSION}"
    return _encode_payload(payload)


def get_bxumidtoken(token: str = "") -> str:
    """Return the ``bx-umidtoken`` value, using env override when present."""
    if token:
        return token
    env_value = os.environ.get("QWEN_BX_UMIDTOKEN", "").strip()
    if env_value:
        return env_value
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "T2gA" + "".join(secrets.choice(alphabet) for _ in range(40))


def lzw_compress(data: str, bits: int = 6, alphabet: str = CUSTOM_BASE64_CHARS) -> str:
    """Compatibility placeholder for the historical LZW helper."""
    if not data:
        return ""
    encoded = base64.urlsafe_b64encode(data.encode("utf-8")).decode("ascii").rstrip("=")
    if alphabet == CUSTOM_BASE64_CHARS:
        return encoded.replace("_", "-")
    return encoded


def custom_encode(data: str, url_safe: bool = True) -> str:
    """Compatibility wrapper around the legacy custom encoder name."""
    encoded = lzw_compress(data)
    if url_safe:
        return encoded
    remainder = len(encoded) % 4
    if remainder:
        encoded += "=" * (4 - remainder)
    return encoded


def get_baxia_tokens() -> Dict[str, str]:
    """Return the Baxia header triplet required by current web requests."""
    fingerprint = generate_fingerprint()
    return {
        "bxV": BAXIA_VERSION,
        "bxUa": generate_bxua(fingerprint),
        "bxUmidToken": get_bxumidtoken(),
        "fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# HTTP Headers
# ---------------------------------------------------------------------------


def make_request_id() -> str:
    """Return a new request identifier."""
    return str(uuid.uuid4())


def make_timezone() -> str:
    """Return the browser-like timezone header value."""
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    if offset is None:
        suffix = "+0000"
    else:
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        suffix = f"{sign}{hours:02d}{minutes:02d}"
    return now.strftime(f"%a %b %d %Y %H:%M:%S GMT{suffix}")


def build_cookie_string(cookies: Optional[Dict[str, Any]]) -> str:
    """Convert a cookie mapping into a request header string."""
    if not cookies:
        return ""
    return "; ".join(f"{key}={value}" for key, value in cookies.items() if value not in {None, ""})


def _base_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
        "Source": "web",
        "X-Request-Id": make_request_id(),
        "Timezone": make_timezone(),
        "Sec-Ch-Ua": SEC_CH_UA,
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": SEC_CH_UA_PLATFORM,
    }


def build_login_headers() -> Dict[str, str]:
    """Build headers for the v2 sign-in endpoint."""
    headers = _base_headers()
    headers["Version"] = APP_VERSION
    headers["x-request-origin"] = BASE_URL
    return headers


def build_headers(
    token: str,
    *,
    chat_id: str = "",
    include_sse: bool = False,
    include_version: bool = True,
    fingerprint: str = "",
    cookies: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build authenticated headers for Qwen chat APIs."""
    headers = _base_headers()
    headers["Authorization"] = f"Bearer {token}"
    baxia = get_baxia_tokens()
    headers["bx-v"] = baxia["bxV"]
    headers["bx-ua"] = baxia["bxUa"]
    headers["bx-umidtoken"] = baxia["bxUmidToken"]
    if include_version:
        headers["version"] = WEB_VERSION
    if chat_id:
        headers["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
    if include_sse:
        headers["X-Accel-Buffering"] = "no"
    cookie_string = build_cookie_string(cookies)
    if cookie_string:
        headers["Cookie"] = cookie_string
    if extra_headers:
        headers.update(extra_headers)
    return headers


def build_stop_headers(token: str) -> Dict[str, str]:
    """Build headers for the stop-generation endpoint."""
    return build_headers(token, include_version=False)
