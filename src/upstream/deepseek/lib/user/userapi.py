from __future__ import annotations

# src/platforms/deepseek/core/userapi.py
"""DeepSeek 用户相关 API 封装（登录/注册/验证码/设置等）"""

import logging
import uuid
from typing import Any, Dict, Tuple

import aiohttp

from upstream.deepseek.lib.biz_error import raise_if_waf_challenge
from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST
from upstream.deepseek.lib.protocol.headers import build_basic_headers
logger = logging.getLogger(__name__)

__all__ = [
    "login",
    "login_by_sms",
    "send_email_code",
    "send_sms_code",
]


def _make_device_id() -> str:
    """生成随机设备 ID（UUID v4 格式）。

    Returns:
        UUID 字符串。
    """
    return str(uuid.uuid4())


def _build_login_payload(username: str, password: str) -> Dict[str, Any]:
    """根据用户名（邮箱或手机号）构造登录请求体，从 login() 抽出。"""
    is_email = "@" in username
    payload: Dict[str, Any] = {
        "password": password,
        "device_id": _make_device_id(),
        "os": "web",
    }
    if is_email:
        payload["email"] = username
        payload["mobile"] = ""
        payload["area_code"] = ""
    else:
        payload["email"] = ""
        payload["mobile"] = username
        payload["area_code"] = "+86"
    return payload


async def login(
    session: aiohttp.ClientSession,
    username: str,
    password: str,
) -> Tuple[str, str]:
    """邮箱或手机号密码登录。

    Args:
        session: aiohttp ClientSession。
        username: 邮箱或手机号。
        password: 密码。

    Returns:
        (token, user_id) 二元组。

    Raises:
        Exception: 登录失败时抛出，含详细错误信息。
    """
    payload = _build_login_payload(username, password)
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
        return str(token), str(user_id)


async def login_by_sms(
    session: aiohttp.ClientSession,
    mobile_number: str,
    sms_code: str,
    area_code: str = "+86",
) -> Tuple[str, str]:
    """手机短信验证码登录。

    Args:
        session: aiohttp ClientSession。
        mobile_number: 手机号码。
        sms_code: 短信验证码。
        area_code: 区号。

    Returns:
        (token, user_id) 二元组。

    Raises:
        Exception: 登录失败时抛出。
    """
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
    """发送短信验证码。

    Args:
        session: aiohttp ClientSession。
        mobile_number: 手机号码。
        scenario: 场景（register/mobile_login/reset_password 等）。
        area_code: 区号。

    Returns:
        是否发送成功。
    """
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
            "https://{}/api/v0/users/create_sms_verification_code".format(
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
        logger.warning("send_sms_code 失败: %s", exc)
        return False


async def send_email_code(
    session: aiohttp.ClientSession,
    email: str,
    scenario: str = "register",
) -> bool:
    """发送邮箱验证码。

    Args:
        session: aiohttp ClientSession。
        email: 邮箱地址。
        scenario: 场景（register/reset_password）。

    Returns:
        是否发送成功。
    """
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
