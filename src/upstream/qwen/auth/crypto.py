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
from typing import Any, Dict, Final, List, Literal, Optional
import logging

logger = logging.getLogger("rogator")

# ---------------------------------------------------------------------------
# Constants — 与 routes.py 对齐
# ---------------------------------------------------------------------------
from upstream.qwen.chat.routes import (
    APP_VERSION,
    BASE_URL,
    CHAT_ORIGIN,
    SEC_CH_UA,
    SEC_CH_UA_PLATFORM,
    USER_AGENT,
)

BAXIA_VERSION: Final[str] = "0.0.3"
CUSTOM_BASE64_CHARS: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
)


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BXUMID
# ---------------------------------------------------------------------------


def validate_bxumidtoken(token: str) -> bool:
    # HAR：T2gA… 含 -/=，长度约 68。
    return bool(token and re.fullmatch(r"(?:T2gA)?[A-Za-z0-9+/=_-]{20,}", token))


# ---------------------------------------------------------------------------
# Cookies — 实现见 auth.http；此处重导出保持旧 import 路径
# ---------------------------------------------------------------------------

from upstream.qwen.auth.http import (  # noqa: E402
    HASH_FIELDS,
    absorb_response_cookies,
    build_cookie_string,
    generate_cookies,
    merge_session_cookies,
    sync_cookie_store,
)


# ---------------------------------------------------------------------------
# Crypto / Baxia
# ---------------------------------------------------------------------------


def generate_device_id() -> str:
    return uuid.uuid4().hex


def build_fingerprint(*, device_id: str | None = None) -> str:
    """构建账号级稳定指纹串（device_id 省略则随机）。"""
    did = device_id or generate_device_id()
    fields = [
        did,
        "1.0.0",
        "web",
        "Chrome",
        "153.0.0.0",
        "zh-CN",
        "Asia/Shanghai",
        "1920x1080",
        "24",
        "Win32",
        "Windows",
        "Google Inc. (NVIDIA)",
        "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0)",
        "desktop",
        "arena",
        "stable",
    ]
    return "^".join(fields)


def collect_fingerprint_data() -> str:

    return build_fingerprint()


def generate_fingerprint() -> str:

    return build_fingerprint()


def _encode_payload(text: str) -> str:

    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def generate_bxua(fingerprint: str) -> str:
    """占位 bx-ua；正常路径走 fireye 纯 Python 模块。"""
    nonce = secrets.token_hex(4)
    payload = f"{fingerprint}|{int(time.time() * 1000)}|{nonce}|{BAXIA_VERSION}"
    return _encode_payload(payload)


_UMID_BODY_LEN: Final[int] = 64
_UMID_ALPHABET: Final[str] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-_"
)


def _new_random_bx_umidtoken() -> str:
    body = "".join(secrets.choice(_UMID_ALPHABET) for _ in range(_UMID_BODY_LEN - 1))
    return "T2gA" + body + "="


def get_bxumidtoken(token: str = "") -> str:
    if token:
        return token
    env_value = os.environ.get("QWEN_BX_UMIDTOKEN", "").strip()
    if env_value:
        return env_value
    return _new_random_bx_umidtoken()


from upstream.qwen.auth.baxia_runtime import (  # noqa: E402
    BAXIA_UA_PATH_MARKERS,
    BaxiaMode,
    ensure_baxia_runtime,
    get_baxia_tokens,
    path_needs_baxia_ua,
    reset_baxia_runtime,
    resolve_baxia_mode,
)


def lzw_compress(data: str, bits: int = 6, alphabet: str = CUSTOM_BASE64_CHARS) -> str:

    if not data:
        return ""
    encoded = base64.urlsafe_b64encode(data.encode("utf-8")).decode("ascii").rstrip("=")
    if alphabet == CUSTOM_BASE64_CHARS:
        return encoded.replace("_", "-")
    return encoded


def custom_encode(data: str, url_safe: bool = True) -> str:

    encoded = lzw_compress(data)
    if url_safe:
        return encoded
    remainder = len(encoded) % 4
    if remainder:
        encoded += "=" * (4 - remainder)
    return encoded


# ---------------------------------------------------------------------------
# HTTP Headers
# ---------------------------------------------------------------------------


def make_request_id() -> str:

    return str(uuid.uuid4())


def make_timezone() -> str:
    """Timezone header：Date.toString() 去掉括号时区名。"""
    return datetime.now().astimezone().strftime("%a %b %d %Y %H:%M:%S GMT%z")


def _base_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
        "source": "web",
        "X-Request-Id": make_request_id(),
        "Timezone": make_timezone(),
        "Sec-Ch-Ua": SEC_CH_UA,
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": SEC_CH_UA_PLATFORM,
    }


def build_login_headers() -> Dict[str, str]:
    headers = _base_headers()
    headers["Version"] = APP_VERSION
    headers["x-request-origin"] = BASE_URL
    return headers


def _apply_header_options(
    headers: Dict[str, str],
    *,
    token: str,
    chat_id: str,
    include_sse: bool,
    include_version: bool,
    cookies: Optional[Dict[str, Any]],
    extra_headers: Optional[Dict[str, str]],
    baxia_tokens: Optional[Dict[str, str]],
    baxia_mode: BaxiaMode,
) -> Dict[str, str]:
    if baxia_mode != "none" and baxia_tokens is not None:
        headers["bx-v"] = baxia_tokens["bxV"]
        if baxia_mode == "full":
            headers["bx-ua"] = baxia_tokens["bxUa"]
            headers["bx-umidtoken"] = baxia_tokens["bxUmidToken"]
    if include_version:
        headers["Version"] = APP_VERSION
    # Completions: Accept=application/json, Referer=/c/local; new-chat: /c/new-chat.
    if include_sse:
        headers["Accept"] = "application/json"
        headers["X-Accel-Buffering"] = "no"
        headers["Referer"] = f"{CHAT_ORIGIN}/c/local"
    elif chat_id:
        headers["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
    else:
        headers["Referer"] = f"{CHAT_ORIGIN}/c/new-chat"
    merged = merge_session_cookies(token, cookies) if token or cookies else {}
    cookie_string = build_cookie_string(merged)
    if cookie_string:
        headers["Cookie"] = cookie_string
    if extra_headers:
        headers.update(extra_headers)
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
    use_bearer: bool = False,
    baxia: Optional[BaxiaMode] = None,
    api_path: str = "",
) -> Dict[str, str]:
    # 上游 web 以 Cookie token= 鉴权，默认不发 Authorization。
    headers = _base_headers()
    if use_bearer and token:
        headers["Authorization"] = f"Bearer {token}"
    mode = resolve_baxia_mode(api_path, explicit=baxia)
    tokens: Optional[Dict[str, str]] = None
    if mode != "none":
        from upstream.qwen.auth.fireye import resolve_baxia_req_url

        req_url = resolve_baxia_req_url(api_path, chat_id=chat_id)
        tokens = get_baxia_tokens(
            fingerprint_override=fingerprint,
            req_url=req_url,
        )
    return _apply_header_options(
        headers,
        token=token,
        chat_id=chat_id,
        include_sse=include_sse,
        include_version=include_version,
        cookies=cookies,
        extra_headers=extra_headers,
        baxia_tokens=tokens,
        baxia_mode=mode,
    )


async def build_headers_async(
    token: str,
    *,
    chat_id: str = "",
    include_sse: bool = False,
    include_version: bool = True,
    fingerprint: str = "",
    cookies: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    use_bearer: bool = False,
    baxia: Optional[BaxiaMode] = None,
    api_path: str = "",
) -> Dict[str, str]:
    from core.transport.blocking import fireye_limiter, run_blocking
    from upstream.qwen.auth.fireye import resolve_baxia_req_url

    headers = _base_headers()
    if use_bearer and token:
        headers["Authorization"] = f"Bearer {token}"
    mode = resolve_baxia_mode(api_path, explicit=baxia)
    tokens: Optional[Dict[str, str]] = None
    if mode != "none":
        req_url = resolve_baxia_req_url(api_path, chat_id=chat_id)
        tokens = await run_blocking(
            get_baxia_tokens,
            fingerprint_override=fingerprint,
            req_url=req_url,
            limiter=fireye_limiter(),
        )
    return _apply_header_options(
        headers,
        token=token,
        chat_id=chat_id,
        include_sse=include_sse,
        include_version=include_version,
        cookies=cookies,
        extra_headers=extra_headers,
        baxia_tokens=tokens,
        baxia_mode=mode,
    )


async def build_stop_headers_async(token: str) -> Dict[str, str]:
    return await build_headers_async(token, include_version=True)


async def build_asr_ws_headers_async(token: str) -> Dict[str, str]:
    headers = await build_headers_async(token, include_version=True)
    headers.pop("Content-Type", None)
    return headers


def build_stop_headers(token: str) -> Dict[str, str]:

    return build_headers(token, include_version=True)


def build_asr_ws_headers(token: str) -> Dict[str, str]:
    # 需 Baxia + Cookie 以绕过 WAF。
    headers = build_headers(token, include_version=True)
    headers.pop("Content-Type", None)
    return headers
