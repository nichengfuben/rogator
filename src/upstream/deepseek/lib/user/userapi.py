from __future__ import annotations

# src/platforms/deepseek/core/userapi.py
"""DeepSeek 用户相关 API 封装（登录/注册/验证码/设置等）"""

import logging
import re
import uuid
from typing import Any, Dict, Tuple

import aiohttp

from upstream.deepseek.lib.adapter.helpers.biz_error import raise_if_waf_challenge
from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST
from upstream.deepseek.lib.protocol.headers import build_basic_headers

logger = logging.getLogger(__name__)

__all__ = [
    "build_login_payload",
    "login",
    "login_by_sms",
    "send_email_code",
    "send_sms_code",
    "get_current_user",
    "ensure_device_id",
    "logout",
]

_NON_DIGIT_RE = re.compile(r"\D+")


def _make_device_id() -> str:

    return str(uuid.uuid4())


def _normalize_area_code(area_code: str) -> str:
    code = (area_code or "+86").strip()
    if not code:
        return "+86"
    return code if code.startswith("+") else "+{}".format(code)


def _normalize_mobile(raw: str, area_code: str = "") -> Tuple[str, str]:

    text = raw.strip()
    code = _normalize_area_code(area_code)
    if text.startswith("+"):
        digits = _NON_DIGIT_RE.sub("", text)
        if digits.startswith("86") and len(digits) > 11:
            return "+86", digits[2:]
        return code, digits
    digits = _NON_DIGIT_RE.sub("", text)
    if digits.startswith("86") and len(digits) > 11:
        digits = digits[2:]
    return code, digits


def build_login_payload(
    username: str,
    password: str,
    device_id: str,
    *,
    area_code: str = "",
) -> Dict[str, Any]:

    identity = username.strip()
    payload: Dict[str, Any] = {
        "password": password,
        "device_id": device_id,
        "os": "web",
    }
    if "@" in identity:
        payload["email"] = identity
        payload["mobile"] = ""
        payload["area_code"] = ""
        return payload
    mobile_code, mobile = _normalize_mobile(identity, area_code)
    payload["email"] = ""
    payload["mobile"] = mobile
    payload["area_code"] = mobile_code
    return payload


def _build_login_payload(
    username: str, password: str, device_id: str, *, area_code: str = ""
) -> Dict[str, Any]:
    return build_login_payload(username, password, device_id, area_code=area_code)


def ensure_device_id(device_id: str = "") -> str:

    return device_id.strip() or _make_device_id()


async def get_current_user(
    session: aiohttp.ClientSession,
    token: str,
) -> Dict[str, Any]:

    headers = build_basic_headers(token)
    async with session.get(
        "https://{}/api/v0/users/current".format(DEFAULT_HOST),
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            raise Exception("users/current HTTP {}".format(resp.status))
        data = await resp.json()
        inner = data.get("data") or {}
        if inner.get("biz_code", -1) != 0:
            raise Exception("users/current biz error: {}".format(data))
        biz = inner.get("biz_data") or {}
        return biz if isinstance(biz, dict) else {}


async def logout(session: aiohttp.ClientSession, token: str) -> bool:

    headers = build_basic_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/users/logout".format(DEFAULT_HOST),
            headers=headers,
            json={},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("logout 失败: %s", exc)
        return False


async def login(
    session: aiohttp.ClientSession,
    username: str,
    password: str,
    *,
    device_id: str = "",
    area_code: str = "",
) -> Tuple[str, str, str]:

    did = ensure_device_id(device_id)
    payload = build_login_payload(username, password, did, area_code=area_code)
    headers = build_basic_headers()
    async with session.post(
        "https://{}/api/v0/users/login".format(DEFAULT_HOST),
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        raise_if_waf_challenge(resp.status, resp.headers)
        if resp.status != 200:
            raise Exception("登录 HTTP 错误 {}".format(resp.status))
        data = await resp.json()
        inner = data.get("data") or {}
        biz_code = inner.get("biz_code", -1)
        if biz_code != 0:
            biz_msg = inner.get("biz_msg", str(data))
            raise Exception("登录业务错误 {}: {}".format(biz_code, biz_msg))
        biz_data = inner.get("biz_data") or {}
        user = biz_data.get("user") or {}
        token = user.get("token", "")
        user_id = user.get("id", "")
        return str(token), str(user_id), did


async def login_by_sms(
    session: aiohttp.ClientSession,
    mobile_number: str,
    sms_code: str,
    area_code: str = "+86",
) -> Tuple[str, str]:

    headers = build_basic_headers()
    payload = {
        "mobile_number": mobile_number,
        "area_code": area_code,
        "sms_verification_code": sms_code,
        "device_id": _make_device_id(),
        "os": "web",
    }
    async with session.post(
        "https://{}/api/v0/users/login_by_mobile_sms".format(DEFAULT_HOST),
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        raise_if_waf_challenge(resp.status, resp.headers)
        if resp.status != 200:
            raise Exception("短信登录 HTTP 错误 {}".format(resp.status))
        data = await resp.json()
        inner = data.get("data") or {}
        biz_code = inner.get("biz_code", -1)
        if biz_code != 0:
            raise Exception("短信登录失败: {}".format(data))
        biz_data = inner.get("biz_data") or {}
        user = biz_data.get("user") or {}
        return str(user.get("token", "")), str(user.get("id", ""))


async def send_sms_code(
    session: aiohttp.ClientSession,
    mobile_number: str,
    scenario: str = "mobile_login",
    area_code: str = "+86",
) -> bool:

    headers = build_basic_headers()
    payload = {
        "locale": "zh_CN",
        "device_id": _make_device_id(),
        "scenario": scenario,
        "mobile_number": mobile_number,
        "area_code": area_code,
    }
    try:
        async with session.post(
            "https://{}/api/v0/users/create_sms_verification_code".format(DEFAULT_HOST),
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return (data.get("data") or {}).get("biz_code", -1) == 0
    except Exception as exc:
        logger.warning("send_sms_code 失败: %s", exc)
        return False


async def send_email_code(
    session: aiohttp.ClientSession,
    email: str,
    scenario: str = "register",
) -> bool:
    """发送邮箱验证码。"""
    headers = build_basic_headers()
    payload = {
        "email": email,
        "locale": "zh_CN",
        "device_id": _make_device_id(),
        "scenario": scenario,
    }
    try:
        async with session.post(
            "https://{}/api/v0/users/create_email_verification_code".format(
                DEFAULT_HOST
            ),
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return (data.get("data") or {}).get("biz_code", -1) == 0
    except Exception as exc:
        logger.warning("send_email_code 失败: %s", exc)
        return False
