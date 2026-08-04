from __future__ import annotations

"""建聊 / completions 相关 tongyi-sg、aes、users/status、v.gif 上报。"""

import json
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from upstream.qwen.auth.crypto import build_headers_async
from upstream.qwen.auth.report.core import (
    PAGE_HOME,
    PAGE_LOCAL,
    PAGE_NEW_CHAT,
    SPM_CHAT,
    SPM_HOME,
    SPM_NEW_CHAT,
    aem_page_id,
    base_typarms,
    cache_tag,
    cna,
    ms_now,
    report_aes_events,
    report_aplus_event,
    silent_request,
    spm_cnt,
    spm_for_path,
    spm_pre_home,
    uid,
)
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


async def report_user_status(
    client: "QwenClient",
    session: "QwenSession",
    *,
    page_path: str = "/",
) -> None:
    """POST /api/v2/users/status —— 页面态上报（Cookie 鉴权，无 Bearer）。"""
    body = {
        "typarms": {
            **base_typarms(session),
            "typarm6": "",
            "share_id": "",
            "project_id": "",
            "channel_type": "",
            "community_type": "",
            "from_id": "",
            "spmId": spm_for_path(page_path),
            "aemPageId": aem_page_id(page_path),
            "domain": "chat.qwen.ai",
        }
    }
    headers = await build_headers_async(
        session.token, include_version=True, baxia="version", use_bearer=False,
    )
    headers["Content-Type"] = "application/json"
    path = page_path if page_path.startswith("/") else f"/{page_path}"
    headers["Referer"] = f"{CHAT_ORIGIN}{path}"
    await silent_request(
        client,
        "POST",
        f"{BASE_URL}{USERS_STATUS_PATH}",
        headers=headers,
        json_body=body,
    )


def _page_view_body(
    session: "QwenSession",
    *,
    page_url: str,
    spm_cnt_base: str,
    spm_url: str,
    spm_pre: str,
    pre_url: str,
) -> str:
    user = uid(session)
    cnt = spm_cnt(spm_cnt_base)
    fields = {
        "logtype": "1",
        "title": "Qwen Studio",
        "pre": pre_url,
        "scr": "1920x1080",
        "_p_url": page_url,
        "cna": cna(),
        "spm-cnt": cnt,
        "spm-url": spm_url,
        "spm-pre": spm_pre,
        "uidaplus": user,
        "aplus": "",
        "sidx": "aplusSidex",
        "ckx": "aplusCkx",
        "pid": "chat_qwen_ai",
        **base_typarms(session),
        "cache": cache_tag(),
        "jsver": "aplus.js",
        "lver": "1.13.26",
        "platformType": "pc",
        "mansndlog": "1",
        "device_model": "Windows",
        "os": "Windows",
        "os_version": "win10",
        "language": "zh-CN",
        "o": "win10",
        "w": "webkit",
        "s": "1920x1080",
        "m": "360ee",
        "ism": "pc",
        "p": "1",
        "b": "chrome153",
        "tag": "1",
        "stag": "-1",
        "lstag": "-1",
        "_g_encode": "utf-8",
    }
    return urlencode(fields)


async def report_page_view(
    client: "QwenClient",
    session: "QwenSession",
    *,
    page_url: str,
    spm_cnt_base: str,
    spm_url: str = "",
    spm_pre: str = "",
    pre_url: str = PAGE_LOCAL,
) -> None:
    """POST aplus /v.gif 页面 PV。"""
    body = _page_view_body(
        session,
        page_url=page_url,
        spm_cnt_base=spm_cnt_base,
        spm_url=spm_url,
        spm_pre=spm_pre,
        pre_url=pre_url,
    )
    await silent_request(
        client,
        "POST",
        f"{APLUS_BASE_URL}/v.gif",
        headers={
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": CHAT_ORIGIN,
            "Referer": page_url,
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
        data=body,
    )


async def report_clk_generate_mode(
    client: "QwenClient",
    session: "QwenSession",
    *,
    msg_type: str = "fast",
) -> None:
    """生成模式点击：tongyi-sg.clkGenerateMode + aes。"""
    await report_aplus_event(
        client,
        session,
        "clkGenerateMode",
        gmkey="CLK",
        extra_gokey={"msg_type": msg_type},
        page_url=PAGE_HOME,
        spm_cnt_base=SPM_HOME,
    )
    await report_aes_events(
        client,
        session,
        [
            {
                "c1": uid(session),
                "c4": "thinking",
                "c10": APP_VERSION,
                "p1": "clkGenerateMode",
                "p4": "CLK",
                "ts": str(ms_now()),
                "type": "event",
            }
        ],
        page_url=PAGE_HOME,
        spm_cnt_base=SPM_HOME,
    )


async def report_chat_generation(
    client: "QwenClient",
    session: "QwenSession",
    *,
    msg_type: str = "t2t",
    send_type: str = "click",
) -> None:
    """用户发起生成：chatGeneration（tongyi-sg + aes）。"""
    home_pre = spm_pre_home()
    await report_aplus_event(
        client,
        session,
        "chatGeneration",
        gmkey="CLK",
        extra_gokey={"msg_type": msg_type, "send_type": send_type},
        page_url=PAGE_HOME,
        spm_cnt_base=SPM_HOME,
    )
    await report_aes_events(
        client,
        session,
        [
            {
                "c1": uid(session),
                "c5": msg_type,
                "c6": send_type,
                "c10": APP_VERSION,
                "p1": "chatGeneration",
                "p4": "CLK",
                "ts": str(ms_now()),
                "type": "event",
            }
        ],
        page_url=PAGE_NEW_CHAT,
        spm_cnt_base=SPM_NEW_CHAT,
        spm_url=home_pre,
        page_id=f"//{CHAT_ORIGIN.replace('https://', '')}/",
    )


async def report_generation_create_return(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    *,
    msg_type: str = "t2t",
) -> None:
    await report_aplus_event(
        client,
        session,
        "generationCreateReturn",
        gmkey="self_define",
        extra_gokey={"c6": "true", "chat_id": chat_id, "msg_type": msg_type},
        page_url=PAGE_LOCAL,
        spm_cnt_base=SPM_CHAT,
        spm_url=f"{SPM_NEW_CHAT}.0.0",
        spm_pre=spm_pre_home(),
    )


async def report_completions_request_id(
    client: "QwenClient",
    session: "QwenSession",
    *,
    request_id: str,
    chat_id: str,
) -> None:
    _ = chat_id
    await report_aplus_event(
        client,
        session,
        "sendCompletionsRequestId",
        gmkey="self_define",
        extra_gokey={"request_id": request_id},
        page_url=PAGE_LOCAL,
        spm_cnt_base=SPM_CHAT,
        spm_url=f"{SPM_NEW_CHAT}.0.0",
        spm_pre=spm_pre_home(),
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
    await report_aplus_event(
        client,
        session,
        "modelStreamingInterfaceStatistics",
        gmkey="self_define",
        extra_gokey={"c6": c6, "c7": c7, "c8": c8},
        page_url=PAGE_LOCAL,
        spm_cnt_base=SPM_CHAT,
        spm_url=f"{SPM_NEW_CHAT}.0.0",
        spm_pre=spm_pre_home(),
    )


async def report_create_chat_sequence(
    client: "QwenClient",
    session: "QwenSession",
) -> None:
    """建聊前上报序列：clk → chatGeneration → new-chat PV → users/status。"""
    await report_clk_generate_mode(client, session)
    await report_chat_generation(client, session)
    await report_page_view(
        client,
        session,
        page_url=PAGE_NEW_CHAT,
        spm_cnt_base=SPM_NEW_CHAT,
        spm_url=spm_pre_home(),
        pre_url=PAGE_LOCAL,
    )
    await report_user_status(client, session, page_path="/c/new-chat")


async def report_after_chat_created(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
) -> None:
    """建聊成功后：generationCreateReturn → local PV → users/status。"""
    await report_generation_create_return(client, session, chat_id)
    await report_page_view(
        client,
        session,
        page_url=PAGE_LOCAL,
        spm_cnt_base=SPM_CHAT,
        spm_url=f"{SPM_NEW_CHAT}.0.0",
        spm_pre=spm_pre_home(),
        pre_url=PAGE_LOCAL,
    )
    await report_user_status(client, session, page_path="/c/local")
