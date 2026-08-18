from __future__ import annotations

"""Qwen 上游辅助 API：config、parse_url、SSE 重连、登录 warm-up。"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from upstream.qwen.auth.crypto import build_headers_async, merge_session_cookies
from upstream.qwen.chat.routes import (
    BASE_URL,
    CHAT_PATH,
    CONFIGS_PATH,
    PARSE_URL_PATH,
    SETTINGS_PATH,
    SETTINGS_UPDATE_PATH,
    SSE_RECONNECT_MAX,
)
from upstream.qwen.chat.sse import iter_sse_events
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import request_json, upstream_timeout

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

# function role 触发时自动下发的账号设置，避免上游返回非 assistant role
DEFAULT_USER_SETTINGS_PAYLOAD: Dict[str, Any] = {
    "ui": {
        "notificationEnabled": False,
        "theme": "dark",
        "language": "",
        "chatBubble": False,
        "showUsername": False,
        "widescreenMode": False,
        "title": {"auto": False},
        "autoTags": False,
        "largeTextAsFile": False,
        "splitLargeChunks": False,
        "scrollOnBranchChange": False,
        "responseAutoCopy": False,
        "models": ["qwen3.8-max"],
    },
    "mcp_remind": False,
    "mcp": {
        "code-interpreter": False,
        "fire-crawl": False,
        "amap": False,
        "image-generation": False,
    },
    "memory": {
        "enable_memory": False,
        "enable_history_memory": False,
        "memory_version_reminder": False,
    },
    "reminder": {"project_version_reminder": False},
    "tts_speaker": {
        "speaker": "Cherry",
        "description": "一位阳光、积极、友好且自然的年轻女士",
        "url": "",
        "gender": "female",
    },
    "tts_speaker_v2": {
        "speaker": "Nini",
        "description": '像糯米糍一样软糯黏腻的嗓音，一声声拉长的"哥哥"，甜得让人骨头都酥了。',
        "url": "",
        "gender": "female",
        "is_personal": False,
        "speaker_id": "",
        "spk_name": "邻家妹妹",
    },
    "aipodcast": {"host": "", "guest": ""},
    "code_settings": {
        "custom_prompt": "",
        "diff_display": "split",
        "branch_format": "",
        "last_repo_choice": "5291cd43-9ead-4d5c-941e-10d6eb2c1b1b",
        "last_branch_choice": "dev",
    },
    "manage_cookies": None,
    "personalization": {
        "name": "system",
        "description": (
            "In this environment you have access to a set of tools you can use "
            "to answer the user's question.\n"
            "You can invoke functions by writing a \"<invoke>\" block like the "
            "following as part of your reply to the user:\n"
            "<invoke name=\"$FUNCTION_NAME\">\n"
            "<parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</parameter>\n"
            "...\n"
            "</invoke>\n"
            "<invoke name=\"$FUNCTION_NAME2\">\n"
            "...\n"
            "</invoke>\n"
            "String and scalar parameters should be specified as is, while lists "
            "and objects should use JSON format.\n"
            "Your turn ends immediately at the closing tag of the last <invoke> "
            "block you emit. You append nothing after it — no comment, no result, "
            "no id, no visible text. The execution environment then runs each tool. "
            "Once a turn is complete, the environment logs it into "
            "<conversation_history> and appends, after each invocation in that log, "
            "an HTML comment stating the environment-generated result id in the form "
            "<!-- Tool Result ID:{id} -->. This comment is written by the environment "
            "when logging a completed turn; you never write it yourself, in this turn "
            "or in imitation of any prior turn, because at the moment you emit an "
            "invocation the id does not yet exist. Separately, the environment appends "
            "the full content of every result, matched by id, to a single flat top-level "
            "block named <funtions_results>, positioned outside and independent of "
            "<conversation_history>. This block accumulates across the whole conversation; "
            "it is never nested inside conversation_history and never adjacent to an invocation."
        ),
        "style": "Default",
        "instruction": "",
        "enable_for_new_chat": True,
    },
    "extension": {
        "authorization": False,
        "check_grant": False,
        "show_guide": False,
        "theme": "",
        "language": "",
    },
    "tools_enabled": {
        "code_interpreter": False,
        "web_extractor": False,
        "web_search": False,
        "image_zoom_in_tool": False,
        "bio": False,
        "history_retriever": False,
        "image_edit_tool": False,
        "image_gen_tool": False,
        "web_search_image": False,
        "image_search": True,
    },
    "model_config": {
    },
}


async def update_user_settings(client: "QwenClient", session: "QwenSession") -> bool:
    """function role 触发时下发默认设置，失败仅 warning 不阻断。"""
    async def _run() -> bool:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{SETTINGS_UPDATE_PATH}",
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or ""),
                ),
            ),
            json=DEFAULT_USER_SETTINGS_PAYLOAD,
            timeout=upstream_timeout(30.0),
        )
        if status == 200 and isinstance(body, dict) and body.get("success"):
            return True
        logger.warning(
            "update_user_settings failed: HTTP %d body=%s",
            status, str(body)[:200],
        )
        return False

    try:
        return await run_with_connection_retry(
            "update_settings", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.warning("update_user_settings failed: %s", exc)
        return False


async def fetch_app_config(client: "QwenClient", session: "QwenSession") -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "GET",
            f"{BASE_URL}{CONFIGS_PATH}",
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or "")
                ),
            ),
            timeout=upstream_timeout(30.0),
        )
        if status == 200 and isinstance(body, dict) and body.get("success"):
            data = body.get("data")
            return data if isinstance(data, dict) else {}
        return {}

    try:
        return await run_with_connection_retry(
            "fetch_config", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("fetch_app_config failed: %s", exc)
        return {}


async def fetch_user_settings(client: "QwenClient", session: "QwenSession") -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "GET",
            f"{BASE_URL}{SETTINGS_PATH}",
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or "")
                ),
            ),
            timeout=upstream_timeout(30.0),
        )
        if status == 200 and isinstance(body, dict):
            data = body.get("data", body)
            return data if isinstance(data, dict) else {}
        return {}

    try:
        return await run_with_connection_retry(
            "fetch_settings", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("fetch_user_settings failed: %s", exc)
        return {}


async def warmup_session(client: "QwenClient", session: "QwenSession") -> None:
    """登录后拉取 configs/settings，并补齐 FE 启动上报。"""
    from upstream.qwen.auth.report import report_compare_log_arrival, report_user_status

    await fetch_app_config(client, session)
    await fetch_user_settings(client, session)
    await report_compare_log_arrival(client)
    await report_user_status(client, session, page_path="/")


async def parse_urls(
    client: "QwenClient",
    session: "QwenSession",
    url_list: List[str],
) -> List[Dict[str, Any]]:
    if not url_list:
        return []

    async def _run() -> List[Dict[str, Any]]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{PARSE_URL_PATH}",
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or "")
                ),
            ),
            json={"url_list": url_list},
            timeout=upstream_timeout(60.0),
        )
        if status != 200 or not isinstance(body, dict) or not body.get("success"):
            return []
        parse_data = (body.get("data") or {}).get("parse_data") or []
        files: List[Dict[str, Any]] = []
        for item in parse_data:
            if not isinstance(item, dict) or item.get("status") != "success":
                continue
            oss_url = str(item.get("oss_url") or "")
            if not oss_url:
                continue
            files.append({
                "type": "file",
                "url": oss_url,
                "file_class": "url",
            })
        return files

    try:
        return await run_with_connection_retry(
            "parse_url", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("parse_urls failed: %s", exc)
        return []


async def reconnect_sse_events_with_retry(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    response_id: str,
    *,
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    last_exc: Exception | None = None
    for attempt in range(SSE_RECONNECT_MAX):
        try:
            async for event in reconnect_sse_events(
                client, session, chat_id, response_id, cookies=cookies,
            ):
                yield event
            return
        except Exception as exc:
            last_exc = exc
            logger.debug(
                "SSE reconnect attempt %d/%d failed: %s",
                attempt + 1, SSE_RECONNECT_MAX, exc,
            )
            if attempt + 1 < SSE_RECONNECT_MAX:
                await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
    if last_exc is not None:
        raise last_exc


async def reconnect_sse_events(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    response_id: str,
    *,
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """GET /chat/completions?chat_id=&response_id= 断线续流。"""
    if cookies is None:
        cookies_fn = getattr(client, "cookies_for_session", None)
        if callable(cookies_fn):
            cookies = cookies_fn(session)
        else:
            cookies = merge_session_cookies(
                session.token, user_id=str(session.user_id or ""),
            )
    http = await client._ensure_http_session()
    async with http.get(
        f"{BASE_URL}{CHAT_PATH}",
        params={"chat_id": chat_id, "response_id": response_id},
        headers=await build_headers_async(
            session.token,
            chat_id=chat_id,
            include_sse=True,
            cookies=cookies,
        ),
        timeout=upstream_timeout(600.0),
    ) as resp:
        absorb_fn = getattr(client, "absorb_cookies_for_session", None)
        if callable(absorb_fn):
            absorb_fn(session, resp, binding=cookies)
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"SSE reconnect HTTP {resp.status}: {body[:200]}")
        async for event in iter_sse_events(client, resp, session):
            yield event
