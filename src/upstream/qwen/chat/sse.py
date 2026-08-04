from __future__ import annotations

"""SSE 流错误检测与 live 事件迭代。"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import aiohttp

from upstream.qwen.chat.upload.parse import SseEventAssembler, parse_sse_event, parse_sse_line
from server.formats import (
    BaxiaSmBlockedError,
    TokenExpiredError,
    UpstreamChatNotFoundError,
    UpstreamTimeoutError,
    UpstreamWafBlockedError,
)
from server.records.sse_record import append_sse_bytes_async

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

_BAXIA_SM_MARKERS: frozenset[str] = frozenset(
    {"RGV587", "FAIL_SYS", "FAIL_SYS_USER_VALIDATE", "RGV587_ERROR::SM"}
)
_UPSTREAM_RATE_LIMIT_CODES: frozenset[str] = frozenset(
    {"RateLimited", "ParallelLimited", "quotaLimited", "Too_Many_Requests"}
)


def _is_baxia_sm_block(message: str, *, punish_url: str = "") -> bool:
    if punish_url and any(marker in message for marker in _BAXIA_SM_MARKERS):
        return True
    return "RGV587_ERROR::SM" in message or "FAIL_SYS_USER_VALIDATE" in message


def _raise_for_success_false(
    client: "QwenClient",
    session: "QwenSession",
    obj: Dict[str, Any],
) -> None:
    from upstream.qwen.chat.chat import raise_qwen_session_error

    msg = json.dumps(obj, ensure_ascii=False)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    code = str(data.get("code") or "")
    if code in _UPSTREAM_RATE_LIMIT_CODES:
        client._invalidate_session(session)
        logger.warning("Session %s upstream rate limited (%s)", session.username[:6], code)
        raise TokenExpiredError(f"Rate limited: {msg[:200]}")
    if code == "CHAT_NOT_FOUND":
        raise UpstreamChatNotFoundError(f"Qwen chat not found: {msg[:200]}", upstream="qwen")
    raise_qwen_session_error(client, session, msg)
    raise RuntimeError(f"Qwen API error: {msg}")


def raise_sse_inline_error(
    client: "QwenClient",
    session: "QwenSession",
    line: str,
) -> None:
    """HTTP 200 但 body 为 Baxia/业务错误 JSON 时抛出可重试或 WAF 异常。"""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return
    try:
        obj = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(obj, dict):
        return

    if "success" in obj:
        if obj.get("success", True):
            return
        _raise_for_success_false(client, session, obj)

    ret = obj.get("ret")
    if not isinstance(ret, list) or not ret:
        return
    msg = " ".join(str(part) for part in ret if part)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    punish_url = str(data.get("url") or "")
    if _is_baxia_sm_block(msg, punish_url=punish_url):
        logger.debug(
            "Baxia SM blocked [%s]: %s",
            session.username[:6],
            msg[:160],
        )
        raise BaxiaSmBlockedError(msg[:200])
    if punish_url or "FAIL_SYS" in msg or "RGV587" in msg:
        raise UpstreamWafBlockedError(
            f"Qwen Baxia blocked: {msg[:200]}",
            upstream="qwen",
        )
    raise RuntimeError(f"Qwen upstream error: {msg[:200]}")


def _check_sse_error_line(client: "QwenClient", line: str, session: "QwenSession") -> None:
    raise_sse_inline_error(client, session, line)


def _track_response_id(
    event: Dict[str, Any],
    response_id_out: Optional[list],
) -> None:
    if response_id_out is None:
        return
    rid = event.get("response_id")
    if rid and event.get("type") in (
        "response_created",
        "response_stopped",
        "response_info",
    ):
        response_id_out[:] = [str(rid)]


def _event_from_sse_data(
    client: "QwenClient",
    session: "QwenSession",
    data_str: str,
    response_id_out: Optional[list],
) -> Optional[Dict[str, Any]]:
    if not data_str or data_str == "[DONE]":
        return None
    event = parse_sse_event(data_str)
    if event:
        _track_response_id(event, response_id_out)
    return event


def _dispatch_assembled_sse_line(
    client: "QwenClient",
    session: "QwenSession",
    line: str,
    assembler: SseEventAssembler,
    response_id_out: Optional[list],
) -> Optional[Dict[str, Any]]:
    payload = assembler.feed_line(line)
    if payload is None:
        if line and not line.startswith("data:") and not line.startswith(":"):
            _check_sse_error_line(client, line, session)
        return None
    return _event_from_sse_data(client, session, payload, response_id_out)


async def iter_sse_events(
    client: "QwenClient",
    resp: aiohttp.ClientResponse,
    session: "QwenSession",
    *,
    response_id_out: Optional[list] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """逐行解析 SSE；对齐前端 kT/_T 组帧 + TCP chunk 行缓冲。"""
    pending = ""
    assembler = SseEventAssembler()
    try:
        async for raw in resp.content:
            await append_sse_bytes_async(raw)
            pending += raw.decode("utf-8", errors="replace")
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.rstrip("\r")
                event = _dispatch_assembled_sse_line(
                    client, session, line, assembler, response_id_out,
                )
                if event:
                    yield event
        tail = pending.rstrip("\r")
        if tail:
            event = _dispatch_assembled_sse_line(
                client, session, tail, assembler, response_id_out,
            )
            if event:
                yield event
        eof_payload = assembler.flush_eof()
        if eof_payload:
            event = _event_from_sse_data(
                client, session, eof_payload, response_id_out,
            )
            if event:
                yield event
    except asyncio.TimeoutError as e:
        raise UpstreamTimeoutError("Upstream SSE read timed out") from e
