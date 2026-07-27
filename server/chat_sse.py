from __future__ import annotations

"""聊天 SSE 流解析与 session 级错误处理。"""

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict

import aiohttp

from core.transport.sse import parse_sse_event
from server.formats import TokenExpiredError, PayloadTooLargeError
from server.session_store import QwenSession, is_session_fatal_error

if TYPE_CHECKING:
    from server.qwen_client import QwenClient

logger = logging.getLogger("rogator")


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
