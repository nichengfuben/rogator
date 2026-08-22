from __future__ import annotations

"""Qwen 上报公共常量与请求辅助。"""

import json
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urlencode

from upstream.qwen.auth.http import get_qwen_proxy
from upstream.qwen.chat.routes import (
    APLUS_BASE_URL,
    APP_VERSION,
    CHAT_ORIGIN,
    USER_AGENT,
)

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

SPM_HOME = "a2ty_o01.29997169"
SPM_NEW_CHAT = "a2ty_o01.29997170"
SPM_CHAT = "a2ty_o01.29997173"
PID = "chat_qwen_ai"
ORGID = "tongyi"
PAGE_HOME = f"{CHAT_ORIGIN}/?temporary-chat=true"
PAGE_NEW_CHAT = f"{CHAT_ORIGIN}/c/new-chat"
PAGE_LOCAL = f"{CHAT_ORIGIN}/c/local"
_CNA_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def ms_now() -> int:
    return int(time.time() * 1000)


def uid(session: "QwenSession") -> str:
    return str(session.user_id or session.username or "").strip()


def cna() -> str:
    return "".join(secrets.choice(_CNA_ALPHABET) for _ in range(24))


def cache_tag() -> str:
    return secrets.token_hex(4)[:7]


def spm_suffix() -> str:
    return f"{secrets.token_hex(4)}{secrets.token_hex(3)}"


def spm_cnt(base: str) -> str:
    return f"{base}.0.0.{spm_suffix()}"


def spm_pre_home() -> str:
    return f"{SPM_HOME}.0.i8.{spm_suffix()}"


def base_typarms(session: "QwenSession") -> Dict[str, str]:
    return {
        "typarm1": "web",
        "typarm2": uid(session),
        "typarm3": "prod",
        "typarm4": "qwen_chat",
        "typarm5": "product",
        "orgid": ORGID,
        "cdn_version": APP_VERSION,
    }


def device_gokey_fields(session: "QwenSession") -> Dict[str, str]:
    fields = base_typarms(session)
    fields.update(
        {
            "pid": PID,
            "cache": cache_tag(),
            "jsver": "aplus.js",
            "lver": "1.13.26",
            "customSdkId": "",
            "platformType": "pc",
            "device_model": "Windows",
            "os": "Windows",
            "os_version": "win10",
            "language": "zh-CN",
            "o": "win10",
            "w": "webkit",
            "s": "1920x1080",
            "scr": "1920x1080",
            "m": "360ee",
            "ism": "pc",
            "p": "1",
            "b": "chrome153",
            "tag": "1",
            "stag": "-1",
            "lstag": "-1",
            "_g_encode": "utf-8",
            "_f_t": "false",
        }
    )
    return fields


def aes_device_tail() -> Dict[str, str]:
    return {
        "cache": cache_tag(),
        "jsver": "aplus.js",
        "lver": "1.13.26",
        "customSdkId": "",
        "platformType": "pc",
        "device_model": "Windows",
        "os": "Windows",
        "os_version": "win10",
        "o": "win10",
        "w": "webkit",
        "s": "1920x1080",
        "scr": "1920x1080",
        "m": "360ee",
        "ism": "pc",
        "p": "1",
        "b": "chrome153",
        "tag": "1",
        "stag": "-1",
        "lstag": "-1",
        "_g_encode": "utf-8",
    }


def pv_id() -> str:
    return "".join(secrets.choice(_CNA_ALPHABET) for _ in range(20))


def spm_for_path(page_path: str) -> str:
    if "/c/new-chat" in page_path:
        return SPM_NEW_CHAT
    if "/c/" in page_path:
        return SPM_CHAT
    return SPM_HOME


def aem_page_id(page_path: str) -> str:
    path = page_path if page_path.startswith("/") else f"/{page_path}"
    if path.startswith("/c/") and path != "/c/new-chat":
        return f"//{CHAT_ORIGIN.replace('https://', '')}/c/"
    return f"//{CHAT_ORIGIN.replace('https://', '')}{path}"


async def silent_request(
    client: "QwenClient",
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Any = None,
    data: Any = None,
    params: Optional[Dict[str, str]] = None,
) -> None:
    """独立短连接上报，不占用上游主 session / 连接池。"""
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=8.0, connect=3.0)
        http = None
        try:
            http = aiohttp.ClientSession(timeout=timeout)
            kwargs: Dict[str, Any] = {"headers": headers or {}, "ssl": False}
            if params:
                kwargs["params"] = params
            if data is not None:
                kwargs["data"] = data
            elif json_body is not None:
                kwargs["json"] = json_body
            kwargs["proxy"] = get_qwen_proxy()
            async with http.request(method, url, **kwargs) as resp:
                await resp.read()
        finally:
            # 上报 session 未注册到任何 transport 管理器，异常路径也要确保回收，
            # 否则 event loop 强压取消时会打 asyncio "Unclosed client session"。
            if http is not None and not http.closed:
                await http.close()
    except Exception as exc:
        logger.debug("Qwen report failed %s %s: %s", method, url.split("?")[0], exc)


async def report_aplus_event(
    client: "QwenClient",
    session: "QwenSession",
    event: str,
    *,
    gmkey: str,
    extra_gokey: Optional[Dict[str, str]] = None,
    page_url: str = PAGE_HOME,
    spm_cnt_base: str = SPM_HOME,
    spm_url: str = "",
    spm_pre: str = "",
    spm_cnt_full: str = "",
) -> None:
    user = uid(session)
    gokey = device_gokey_fields(session)
    cnt = spm_cnt_full or spm_cnt(spm_cnt_base)
    gokey.update({"spm-url": spm_url, "spm-pre": spm_pre, "spm-cnt": cnt})
    if extra_gokey:
        gokey.update(extra_gokey)
    params = {
        "gmkey": gmkey,
        "gokey": urlencode(gokey),
        "cna": cna(),
        "spm-cnt": cnt,
        "_gr_uid_": user,
        "uidaplus": user,
        "logtype": "2",
        "scr": "1920x1080",
        "_p_url": page_url,
    }
    if spm_url:
        params["spm-url"] = spm_url
    if spm_pre:
        params["spm-pre"] = spm_pre
    await silent_request(
        client,
        "GET",
        f"{APLUS_BASE_URL}/tongyi-sg.qwen_chat.{event}",
        headers={"Referer": page_url, "User-Agent": USER_AGENT},
        params=params,
    )


def _aes_gokey_base(
    session: "QwenSession",
    *,
    page_url: str,
    spm_cnt_base: str,
    spm_url: str,
    spm_pre: str,
    msg: str,
    page_id: str,
) -> tuple[Dict[str, str], str]:
    user = uid(session)
    cnt = spm_cnt(spm_cnt_base)
    gokey: Dict[str, str] = {
        "spm-url": spm_url,
        "spm-pre": spm_pre,
        "spm-cnt": cnt,
        "_f_t": "false",
        "pid": "qwen-webui",
        "sdk_version": "3.1.0",
        "pv_id": pv_id(),
        "timezone_offset": "-480",
        "title": "Qwen Studio",
        "spm_a": "a2ty_o01",
        "spm_b": spm_cnt_base.rsplit(".", 1)[-1],
        "dpi": "1",
        "sr": "1920x1080",
        "platform": "web",
        "language": "zh-CN",
        "env": "prod",
        "downlink": "10",
        "net_type": "4g",
        "origin_url": page_url,
        "page_id": page_id,
        "dim1": user,
        "msg": msg,
    }
    gokey.update(aes_device_tail())
    return gokey, cnt


async def report_aes_events(
    client: "QwenClient",
    session: "QwenSession",
    events: list[Dict[str, str]],
    *,
    page_url: str = PAGE_HOME,
    spm_cnt_base: str = SPM_HOME,
    spm_url: str = "",
    spm_pre: str = "",
    page_id: str = "",
) -> None:
    """POST aplus aes.1.1（EXP），msg 内可批量多事件（| 分隔）。"""
    if not events:
        return
    user = uid(session)
    msg = "|".join(urlencode(ev) for ev in events)
    resolved_page_id = page_id or f"//{CHAT_ORIGIN.replace('https://', '')}/"
    gokey, cnt = _aes_gokey_base(
        session,
        page_url=page_url,
        spm_cnt_base=spm_cnt_base,
        spm_url=spm_url,
        spm_pre=spm_pre,
        msg=msg,
        page_id=resolved_page_id,
    )
    body = _aes_request_body(user, gokey, cnt, page_url, spm_url, spm_pre)
    await silent_request(
        client,
        "POST",
        f"{APLUS_BASE_URL}/aes.1.1",
        headers={
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": CHAT_ORIGIN,
            "Referer": page_url,
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
    )


def _aes_request_body(
    user: str,
    gokey: Dict[str, str],
    cnt: str,
    page_url: str,
    spm_url: str,
    spm_pre: str,
) -> Dict[str, str]:
    body: Dict[str, str] = {
        "gmkey": "EXP",
        "gokey": urlencode(gokey),
        "cna": cna(),
        "spm-cnt": cnt,
        "_gr_uid_": user,
        "uidaplus": user,
        "logtype": "2",
        "scr": "1920x1080",
        "_p_url": page_url,
    }
    if spm_url:
        body["spm-url"] = spm_url
    if spm_pre:
        body["spm-pre"] = spm_pre
    return body

