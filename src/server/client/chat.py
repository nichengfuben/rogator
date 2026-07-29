from __future__ import annotations

"""Qwen 聊天创建与 SSE 流解析。"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict

import aiohttp

from core.crypto.crypto import build_headers
from core.transport.routes import BASE_URL, NEW_CHAT_PATH
from core.transport.sse import parse_sse_event
from server.config import CONFIG
from server.formats import PayloadTooLargeError, TokenExpiredError, UpstreamTimeoutError
from server.client.session_store import QwenSession, is_session_fatal_error

if TYPE_CHECKING:
    from server.client.qwen_client import QwenClient

logger = logging.getLogger("rogator")


def check_create_chat_error(client: QwenClient, session: QwenSession, data: Dict[str, Any]) -> None:
    data_obj = data.get("data") or {}
    if not isinstance(data_obj, dict):
        raise RuntimeError(f"Create chat failed: {data}")
    details = str(data_obj.get("details", ""))
    if is_session_fatal_error(f"{data_obj.get('code', '')} {details}"):
        client._invalidate_session(session)
        raise TokenExpiredError(f"Token expired: {details}")
    raise RuntimeError(f"Create chat failed: {data}")


async def _post_create_chat(session: QwenSession, model: str, timeout_s: float) -> Dict[str, Any]:
    payload = {
        "title": "新建对话",
        "models": [model],
        "chat_mode": "local",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
        "project_id": "",
    }
    headers = build_headers(session.token, include_version=False)
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"{BASE_URL}{NEW_CHAT_PATH}",
            json=payload,
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                return {"_http_status": resp.status}
            return await resp.json()


async def create_chat_for_session(
    client: QwenClient,
    session: QwenSession,
    model: str,
) -> str:
    timeout_s = CONFIG.create_chat_timeout
    try:
        data = await _post_create_chat(session, model, timeout_s)
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError(f"Create chat timed out after {timeout_s}s") from exc

    http_status = data.pop("_http_status", None)
    if http_status in (401, 403):
        client._invalidate_session(session)
        raise TokenExpiredError(f"Token expired: HTTP {http_status}")
    if http_status is not None:
        raise RuntimeError(f"Create chat HTTP {http_status}")

    if not data.get("success"):
        check_create_chat_error(client, session, data)
    chat_id = str((data.get("data") or {}).get("id", ""))
    if not chat_id:
        raise RuntimeError(f"Create chat failed: no chat_id in {data}")
    return chat_id


async def handle_chat_error(client: QwenClient, resp: aiohttp.ClientResponse, session: QwenSession) -> None:
    if resp.status in (401, 403):
        client._invalidate_session(session)
        raise TokenExpiredError(f"Token expired: HTTP {resp.status}")
    body = await resp.text()
    if resp.status == 413:
        raise PayloadTooLargeError(f"Payload too large: {body[:200]}")
    if is_session_fatal_error(body):
        client._invalidate_session(session)
        if "RateLimited" in body or "daily usage" in body:
            logger.warning("Session %s rate limited", session.username[:6])
            raise TokenExpiredError(f"Rate limited: {body[:200]}")
        raise TokenExpiredError(f"Token expired: {body[:200]}")
    logger.error("Chat HTTP %d: %s", resp.status, body[:500])
    raise RuntimeError(f"Chat HTTP {resp.status}: {body[:200]}")


def check_sse_error_line(client: QwenClient, line: str, session: QwenSession) -> None:
    if not (line.startswith("{") and "success" in line):
        return
    err = json.loads(line)
    if err.get("success", True):
        return
    msg = json.dumps(err, ensure_ascii=False)
    if is_session_fatal_error(msg):
        client._invalidate_session(session)
        if "RateLimited" in msg or "daily usage" in msg:
            raise TokenExpiredError(f"Rate limited: {msg}")
        raise TokenExpiredError(f"Token expired: {msg}")
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
