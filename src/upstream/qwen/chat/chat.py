from __future__ import annotations

"""Qwen chat creation and SSE parsing."""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import aiohttp

from upstream.qwen.auth.crypto import build_headers
from upstream.qwen.chat.routes import BASE_URL, NEW_CHAT_PATH
from upstream.qwen.chat.sse import parse_sse_event
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import request_json, upstream_timeout
from server.config import CONFIG
from server.formats import (
    PayloadTooLargeError,
    TokenExpiredError,
    UpstreamTimeoutError,
    UpstreamWafBlockedError,
)
from upstream.qwen.chat.store import QwenSession, is_session_fatal_error

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient

logger = logging.getLogger("rogator")


def raise_qwen_session_error(
    client: "QwenClient",
    session: QwenSession,
    text: str,
    *,
    http_status: Optional[int] = None,
) -> None:
    """若 text/status 表明会话失效则 invalidate 并抛 TokenExpiredError。"""
    if http_status in (401, 403):
        client._invalidate_session(session)
        raise TokenExpiredError(f"Token expired: HTTP {http_status}")
    if not is_session_fatal_error(text):
        return
    client._invalidate_session(session)
    if "RateLimited" in text or "daily usage" in text:
        logger.warning("Session %s rate limited", session.username[:6])
        raise TokenExpiredError(f"Rate limited: {text[:200]}")
    raise TokenExpiredError(f"Token expired: {text[:200]}")


def _raise_for_non_json_create_chat(
    client: QwenClient,
    session: QwenSession,
    text: str,
) -> None:
    """非 JSON 响应：先尝试会话失效判定，否则视为 WAF 拦截。"""
    raise_qwen_session_error(client, session, text)
    snippet = text.strip()[:200]
    raise UpstreamWafBlockedError(
        "Qwen create_chat 返回非 JSON（疑似 Baxia/WAF 拦截）。"
        "请配置有效 QWEN_BX_UMIDTOKEN 或重新登录账号。"
        f" 响应片段: {snippet}",
        upstream="qwen",
    )


def check_create_chat_error(client: QwenClient, session: QwenSession, data: Dict[str, Any]) -> None:
    data_obj = data.get("data") or {}
    if not isinstance(data_obj, dict):
        raise RuntimeError(f"Create chat failed: {data}")
    details = str(data_obj.get("details", ""))
    raise_qwen_session_error(
        client, session, f"{data_obj.get('code', '')} {details}",
    )
    raise RuntimeError(f"Create chat failed: {data}")


async def _post_create_chat(client: QwenClient, session: QwenSession, model: str, timeout_s: float) -> Dict[str, Any]:
    payload = {
        "title": "新建对话",
        "models": [model],
        "chat_mode": "local",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
        "project_id": "",
    }
    headers = build_headers(session.token, include_version=False)

    async def _run() -> Dict[str, Any]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{NEW_CHAT_PATH}",
            headers=headers,
            json=payload,
            timeout=upstream_timeout(timeout_s),
        )
        if status != 200:
            return {"_http_status": status}
        if isinstance(body, dict):
            return body
        _raise_for_non_json_create_chat(client, session, str(body))
        return {}
    return await run_with_connection_retry("create_chat", _run, transport_owner=client)


async def create_chat_for_session(
    client: QwenClient,
    session: QwenSession,
    model: str,
) -> str:
    timeout_s = CONFIG.create_chat_timeout
    try:
        data = await _post_create_chat(client, session, model, timeout_s)
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError(f"Create chat timed out after {timeout_s}s") from exc

    http_status = data.pop("_http_status", None)
    if http_status is not None:
        raise_qwen_session_error(client, session, "", http_status=http_status)
        raise RuntimeError(f"Create chat HTTP {http_status}")

    if not data.get("success"):
        check_create_chat_error(client, session, data)
    chat_id = str((data.get("data") or {}).get("id", ""))
    if not chat_id:
        raise RuntimeError(f"Create chat failed: no chat_id in {data}")
    return chat_id


async def handle_chat_error(client: QwenClient, resp: aiohttp.ClientResponse, session: QwenSession) -> None:
    raise_qwen_session_error(client, session, "", http_status=resp.status)
    body = await resp.text()
    if resp.status == 413:
        raise PayloadTooLargeError(f"Payload too large: {body[:200]}")
    raise_qwen_session_error(client, session, body)
    logger.error("Chat HTTP %d: %s", resp.status, body[:500])
    raise RuntimeError(f"Chat HTTP {resp.status}: {body[:200]}")


def check_sse_error_line(client: QwenClient, line: str, session: QwenSession) -> None:
    if not (line.startswith("{") and "success" in line):
        return
    err = json.loads(line)
    if err.get("success", True):
        return
    msg = json.dumps(err, ensure_ascii=False)
    raise_qwen_session_error(client, session, msg)
    raise RuntimeError(f"Qwen API error: {msg}")


async def iter_sse_events(
    client: QwenClient,
    resp: aiohttp.ClientResponse,
    session: QwenSession,
) -> AsyncGenerator[Dict[str, Any], None]:
    try:
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                check_sse_error_line(client, line, session)
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            event = parse_sse_event(data_str)
            if event:
                yield event
    except asyncio.TimeoutError as e:
        raise UpstreamTimeoutError("Upstream SSE read timed out") from e
