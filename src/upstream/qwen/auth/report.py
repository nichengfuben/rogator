from __future__ import annotations

"""对齐 FE 的 Qwen 上报：users/status + aplus tongyi-sg 埋点。"""

import json
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urlencode

from upstream.qwen.auth.crypto import build_headers
from upstream.qwen.chat.routes import (
    APLUS_BASE_URL,
    APP_VERSION,
    BASE_URL,
    CHAT_ORIGIN,
    USER_AGENT,
    USERS_STATUS_PATH,
)

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

_SPM_HOME = "a2ty_o01.29997169"
_SPM_NEW_CHAT = "a2ty_o01.29997170"
_SPM_CHAT = "a2ty_o01.29997173"
_PID = "chat_qwen_ai"
_ORGID = "tongyi"
_CNA_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

def _ms_now() -> int:
    return int(time.time() * 1000)

def _uid(session: "QwenSession") -> str:
    return str(session.user_id or session.username or "").strip()

def _cna() -> str:
    return "".join(secrets.choice(_CNA_ALPHABET) for _ in range(24))

def _cache_tag() -> str:
    return secrets.token_hex(4)[:7]

def _base_typarms(session: "QwenSession") -> Dict[str, str]:
    return {
        "typarm1": "web",
        "typarm2": _uid(session),
        "typarm3": "prod",
        "typarm4": "qwen_chat",
        "typarm5": "product",
        "orgid": _ORGID,
        "cdn_version": APP_VERSION,
    }

def _device_gokey_fields(session: "QwenSession") -> Dict[str, str]:
    fields = _base_typarms(session)
    fields.update(
        {
            "pid": _PID,
            "cache": _cache_tag(),
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

def _spm_for_path(page_path: str) -> str:
    if "/c/new-chat" in page_path:
        return _SPM_NEW_CHAT
    if "/c/" in page_path:
        return _SPM_CHAT
    return _SPM_HOME

def _aem_page_id(page_path: str) -> str:
    path = page_path if page_path.startswith("/") else f"/{page_path}"
    if path.startswith("/c/") and path != "/c/new-chat":
        return f"//{CHAT_ORIGIN.replace('https://', '')}/c/"
    return f"//{CHAT_ORIGIN.replace('https://', '')}{path}"

async def _silent_request(
    client: "QwenClient",
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Any = None,
    params: Optional[Dict[str, str]] = None,
) -> None:
    """独立短连接上报，不占用上游主 session / 连接池。"""
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=8.0, connect=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            kwargs: Dict[str, Any] = {
                "headers": headers or {},
                "ssl": False,
            }
            if params:
                kwargs["params"] = params
            if json_body is not None:
                kwargs["json"] = json_body
            async with http.request(method, url, **kwargs) as resp:
                await resp.read()
    except Exception as exc:
        logger.debug("Qwen report failed %s %s: %s", method, url.split("?")[0], exc)

async def report_user_status(
    client: "QwenClient",
    session: "QwenSession",
    *,
    page_path: str = "/",
) -> None:
    """POST /api/v2/users/status —— FE 页面态上报。"""
    spm = _spm_for_path(page_path)
    body = {
        "typarms": {
            **_base_typarms(session),
            "typarm6": "",
            "share_id": "",
            "project_id": "",
            "channel_type": "",
            "community_type": "",
            "from_id": "",
            "spmId": spm,
            "aemPageId": _aem_page_id(page_path),
            "domain": "chat.qwen.ai",
        }
    }
    headers = build_headers(session.token, include_version=True, baxia="version")
    headers["Content-Type"] = "application/json"
    headers["Referer"] = f"{CHAT_ORIGIN}{page_path if page_path.startswith('/') else '/' + page_path}"
    await _silent_request(
        client,
        "POST",
        f"{BASE_URL}{USERS_STATUS_PATH}",
        headers=headers,
        json_body=body,
    )

async def _report_aplus_event(
    client: "QwenClient",
    session: "QwenSession",
    event: str,
    *,
    gmkey: str,
    extra_gokey: Optional[Dict[str, str]] = None,
    page_url: str = f"{CHAT_ORIGIN}/",
    spm_cnt: str = _SPM_HOME,
    spm_url: str = "",
    spm_pre: str = "",
) -> None:
    uid = _uid(session)
    gokey = _device_gokey_fields(session)
    gokey.update(
        {
            "spm-url": spm_url,
            "spm-pre": spm_pre,
            "spm-cnt": f"{spm_cnt}.0.0.{secrets.token_hex(4)}",
        }
    )
    if extra_gokey:
        gokey.update(extra_gokey)
    params = {
        "gmkey": gmkey,
        "gokey": urlencode(gokey),
        "cna": _cna(),
        "spm-cnt": gokey["spm-cnt"],
        "_gr_uid_": uid,
        "uidaplus": uid,
        "logtype": "2",
        "scr": "1920x1080",
        "_p_url": page_url,
    }
    if spm_url:
        params["spm-url"] = spm_url
    if spm_pre:
        params["spm-pre"] = spm_pre
    headers = {
        "Referer": page_url,
        "User-Agent": USER_AGENT,
    }
    await _silent_request(
        client,
        "GET",
        f"{APLUS_BASE_URL}/tongyi-sg.qwen_chat.{event}",
        headers=headers,
        params=params,
    )

async def report_chat_generation(client: "QwenClient", session: "QwenSession") -> None:
    """用户发起生成前：chatGeneration。"""
    await _report_aplus_event(
        client,
        session,
        "chatGeneration",
        gmkey="CLK",
        extra_gokey={"send_type": "enter"},
        page_url=f"{CHAT_ORIGIN}/",
        spm_cnt=_SPM_HOME,
    )

async def report_generation_create_return(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    *,
    msg_type: str = "t2t",
) -> None:
    page_url = f"{CHAT_ORIGIN}/c/{chat_id}"
    await _report_aplus_event(
        client,
        session,
        "generationCreateReturn",
        gmkey="self_define",
        extra_gokey={"c6": "true", "chat_id": chat_id, "msg_type": msg_type},
        page_url=page_url,
        spm_cnt=_SPM_CHAT,
        spm_url=f"{_SPM_NEW_CHAT}.0.0",
        spm_pre=f"{_SPM_HOME}.0.i0",
    )

async def report_completions_request_id(
    client: "QwenClient",
    session: "QwenSession",
    *,
    request_id: str,
    chat_id: str,
) -> None:
    page_url = f"{CHAT_ORIGIN}/c/{chat_id}"
    await _report_aplus_event(
        client,
        session,
        "sendCompletionsRequestId",
        gmkey="self_define",
        extra_gokey={"request_id": request_id},
        page_url=page_url,
        spm_cnt=_SPM_CHAT,
        spm_url=f"{_SPM_NEW_CHAT}.0.0",
        spm_pre=f"{_SPM_HOME}.0.i0",
    )

def _build_streaming_c8(
    *,
    chat_id: str,
    model: str,
    request_id: str,
    response_id: str,
    api_start_ms: int,
    first_chunk_ms: int,
    end_chunk_ms: int,
    total_chars: int,
    is_stop: bool,
    is_error: bool,
    chat_type: str,
) -> tuple[str, str, str]:
    first = first_chunk_ms or end_chunk_ms or api_start_ms
    end = end_chunk_ms or first
    c6 = max(0, first - api_start_ms)
    c7 = max(0, end - first)
    interval = 0.0
    if total_chars > 0 and c7 > 0:
        interval = round(c7 / max(total_chars, 1), 2)
    c8 = {
        "avg_chunk_interval": interval,
        "avg_char_rate": 0,
        "model": model,
        "api_start_time": api_start_ms,
        "total_chars": total_chars,
        "end_chunk_time": end,
        "first_chunk_time": first,
        "x_request_id": request_id,
        "is_stop": is_stop,
        "is_error": is_error,
        "is_leave": False,
        "error_info": None,
        "response_id": response_id,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "isMultiResponse": False,
        "response_index": 0,
    }
    return str(c6), str(c7), json.dumps(c8, ensure_ascii=False, separators=(",", ":"))

async def report_streaming_statistics(
    client: "QwenClient",
    session: "QwenSession",
    *,
    chat_id: str,
    model: str,
    request_id: str,
    response_id: str,
    api_start_ms: int,
    first_chunk_ms: int,
    end_chunk_ms: int,
    total_chars: int = 0,
    is_stop: bool = False,
    is_error: bool = False,
    chat_type: str = "t2t",
) -> None:
    c6, c7, c8 = _build_streaming_c8(
        chat_id=chat_id,
        model=model,
        request_id=request_id,
        response_id=response_id,
        api_start_ms=api_start_ms,
        first_chunk_ms=first_chunk_ms,
        end_chunk_ms=end_chunk_ms,
        total_chars=total_chars,
        is_stop=is_stop,
        is_error=is_error,
        chat_type=chat_type,
    )
    page_url = f"{CHAT_ORIGIN}/c/{chat_id}"
    await _report_aplus_event(
        client,
        session,
        "modelStreamingInterfaceStatistics",
        gmkey="self_define",
        extra_gokey={"c6": c6, "c7": c7, "c8": c8},
        page_url=page_url,
        spm_cnt=_SPM_CHAT,
        spm_url=f"{_SPM_NEW_CHAT}.0.0",
        spm_pre=f"{_SPM_HOME}.0.i0",
    )

async def report_compare_log_arrival(client: "QwenClient") -> None:
    """启动期 compareLogService beacon（对齐 FE sendBeacon）。"""
    log_id = secrets.token_hex(20)
    ts = _ms_now()
    for host, service in (
        ("https://aplus.qwen.ai", "aplus.qwen.ai"),
        ("https://ss.qwen.ai", "ss.qwen.ai"),
    ):
        path = (
            "/service.stability.log_arrival_rate"
            if "aplus" in host
            else "/ss.compare.service"
        )
        body = {
            "gmkey": "OTHER",
            "gokey": urlencode(
                {
                    "logId": log_id,
                    "timestamp": str(ts),
                    "domain": "chat.qwen.ai",
                    "testTag": "compareLogService",
                    "testVersion": "5.0.0",
                    "serviceName": service,
                    "requestType": "sendBeacon",
                }
            ),
        }
        await _silent_request(
            client,
            "POST",
            f"{host}{path}?logId={log_id}",
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "Origin": CHAT_ORIGIN,
                "Referer": f"{CHAT_ORIGIN}/",
            },
            json_body=body,
        )
