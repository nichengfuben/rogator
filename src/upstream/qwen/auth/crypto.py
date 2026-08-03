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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Literal, Optional, Tuple
import logging

logger = logging.getLogger("rogator")

# ---------------------------------------------------------------------------
# Constants — 与 routes.py 对齐
# ---------------------------------------------------------------------------
from upstream.qwen.chat.routes import (
    APP_VERSION,
    BASE_URL,
    BAXIA_SDK_VERSION,
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
# Cookies
# ---------------------------------------------------------------------------

HASH_FIELDS: Final[list] = [
    "ssxmod_itna",
    "ssxmod_itna2",
    "bx-umidtoken",
    "bx-ua",
]


def generate_cookies(fingerprint: str) -> Dict[str, Any]:

    return {
        "ssxmod_itna": "",
        "ssxmod_itna2": "",
        "fingerprint": fingerprint,
    }


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
        "Apple GPU",
        "Apple GPU",
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
# FE main.js checkApiPath：仅这些路径注入 bx-ua / bx-umidtoken。
BAXIA_UA_PATH_MARKERS: Final[Tuple[str, ...]] = (
    "/api/chat/completions",
    "/api/chats/new",
    "/api/chat/completed",
    "/api/v1/chats",
    "/api/v1/chats/all/tags",
    "/api/task/suggestions/completions",
    "/api/v1/tasks/status",
    "/api/v1/files/getstsToken",
    "/api/task/title/completions",
    "/api/task/tags/completions",
    "/api/parse_url",
    "/api/v2/chats",
    "/api/v2/chat/completions",
    "/api/v2/task/suggestions/completions",
    "/api/v2/files/getstsToken",
    "/api/v2/community",
    "/api/v2/tts/completions",
    "/api/v2/files/getfilelink",
    "/api/v2/files/parse",
    "/api/v2/files/parse/status",
    "/api/v2/evaluations/feedback",
)
BaxiaMode = Literal["full", "version", "none"]
_runtime_lock = threading.Lock()
_runtime_fp: str = ""
_runtime_umid: str = ""


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


def ensure_baxia_runtime(*, fingerprint_override: str = "") -> Tuple[str, str]:
    """对齐 FE：页面会话内 umid/指纹稳定；在 getUidToken 就绪后才发受保护 API。"""
    global _runtime_fp, _runtime_umid
    with _runtime_lock:
        if fingerprint_override.strip():
            _runtime_fp = fingerprint_override.strip()
        elif not _runtime_fp:
            _runtime_fp = generate_fingerprint()
        if not _runtime_umid:
            _runtime_umid = get_bxumidtoken()
        return _runtime_fp, _runtime_umid


def reset_baxia_runtime() -> None:
    """SM/换号时轮换，对应浏览器新会话重新 init fireye。"""
    global _runtime_fp, _runtime_umid
    with _runtime_lock:
        _runtime_fp = ""
        _runtime_umid = ""
    try:
        from upstream.qwen.auth.fireye import reset_session

        reset_session()
    except Exception:
        pass


def path_needs_baxia_ua(path: str) -> bool:
    return any(marker in path for marker in BAXIA_UA_PATH_MARKERS)


def resolve_baxia_mode(path: str = "", *, explicit: Optional[BaxiaMode] = None) -> BaxiaMode:
    if explicit is not None:
        return explicit
    if not path:
        return "full"
    if path_needs_baxia_ua(path):
        return "full"
    return "version"


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


def get_baxia_tokens(
    *,
    fingerprint_override: str = "",
    req_url: str = "",
) -> Dict[str, str]:
    # umid/指纹会话级复用；bx-ua 每请求由 fireye 纯 Python 生成。
    fingerprint, umid = ensure_baxia_runtime(
        fingerprint_override=fingerprint_override,
    )
    bx_ua = ""
    try:
        from upstream.qwen.auth.fireye import bind_fingerprint, get_fy_token, get_uid_token

        bind_fingerprint(fingerprint, umid=umid)
        cand = get_fy_token(req_url, fingerprint=fingerprint)
        if cand.startswith("231!") and len(cand) > 100:
            bx_ua = cand
        fy_umid = get_uid_token(fingerprint=fingerprint).strip()
        if fy_umid and validate_bxumidtoken(fy_umid):
            umid = fy_umid
    except Exception as exc:
        logger.debug("fireye token fallback: %s", exc)
    if not bx_ua:
        bx_ua = generate_bxua(fingerprint)
    return {
        "bxV": BAXIA_SDK_VERSION,
        "bxUa": bx_ua,
        "bxUmidToken": umid,
        "fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# HTTP Headers
# ---------------------------------------------------------------------------


def make_request_id() -> str:

    return str(uuid.uuid4())


def make_timezone() -> str:
    """对齐 FE：Date.toString() 去掉括号时区名。"""
    return datetime.now().astimezone().strftime("%a %b %d %Y %H:%M:%S GMT%z")


def merge_session_cookies(
    token: str, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    cookies: Dict[str, str] = {"token": token}
    if extra:
        for key, value in extra.items():
            if value not in (None, ""):
                cookies[str(key)] = str(value)
    return cookies


def build_cookie_string(cookies: Optional[Dict[str, Any]]) -> str:

    if not cookies:
        return ""
    return "; ".join(
        f"{key}={value}" for key, value in cookies.items() if value not in {None, ""}
    )


def _base_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
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


def build_headers(
    token: str,
    *,
    chat_id: str = "",
    include_sse: bool = False,
    include_version: bool = True,
    fingerprint: str = "",
    cookies: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    use_bearer: bool = True,
    baxia: Optional[BaxiaMode] = None,
    api_path: str = "",
) -> Dict[str, str]:

    headers = _base_headers()
    if use_bearer and token:
        headers["Authorization"] = f"Bearer {token}"
    mode = resolve_baxia_mode(api_path, explicit=baxia)
    if mode != "none":
        tokens = get_baxia_tokens(fingerprint_override=fingerprint)
        headers["bx-v"] = tokens["bxV"]
        if mode == "full":
            headers["bx-ua"] = tokens["bxUa"]
            headers["bx-umidtoken"] = tokens["bxUmidToken"]
    if include_version:
        headers["Version"] = APP_VERSION
    if chat_id:
        headers["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
    if include_sse:
        headers["X-Accel-Buffering"] = "no"
    merged = merge_session_cookies(token) if token else {}
    if cookies:
        merged.update(
            {str(k): str(v) for k, v in cookies.items() if v not in (None, "")}
        )
    cookie_string = build_cookie_string(merged)
    if cookie_string:
        headers["Cookie"] = cookie_string
    if extra_headers:
        headers.update(extra_headers)
    return headers


def build_stop_headers(token: str) -> Dict[str, str]:

    return build_headers(token, include_version=True)


def build_asr_ws_headers(token: str) -> Dict[str, str]:
    # 需 Baxia + Cookie 以绕过 WAF。
    headers = build_headers(token, include_version=True)
    headers.pop("Content-Type", None)
    return headers
