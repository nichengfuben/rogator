from __future__ import annotations

"""Qwen chat creation, stop/delete, and SSE parsing."""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import aiohttp

from upstream.qwen.auth.crypto import build_headers, build_stop_headers
from upstream.qwen.chat.routes import BASE_URL, DELETE_CHAT_PATH, NEW_CHAT_PATH, STOP_CHAT_PATH
from upstream.qwen.chat.sse import SseEventAssembler, parse_sse_event
from upstream.qwen.chat.upload.payload import build_stop_payload
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import request_json, upstream_timeout
from server.config import CONFIG
from server.records.sse_record import append_sse_bytes
from server.formats import (
    PayloadTooLargeError,
    BaxiaSmBlockedError,
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
    raise_qwen_session_error(client, session, text)
    snippet = text.strip()[:200]
    raise UpstreamWafBlockedError(
        "Qwen create_chat returned non-JSON (possible Baxia/WAF block). "
        "Configure per-account Baxia profiles or re-login. "
        f"Snippet: {snippet}",
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
        "title": "New Chat",
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


async def stop_upstream_generation(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    response_id: str = "",
) -> bool:
    if not chat_id or not session.token:
        return False

    async def _run() -> bool:
        http = await client._ensure_http_session()
        status, _body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{STOP_CHAT_PATH}?chat_id={chat_id}",
            headers=build_stop_headers(session.token),
            json=build_stop_payload(chat_id, response_id),
            timeout=upstream_timeout(15.0),
        )
        return status in (200, 204)

    try:
        return await run_with_connection_retry(
            "stop_generation", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("Stop generation failed chat=%s: %s", chat_id[:8], exc)
        return False


async def delete_upstream_chat(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
) -> bool:
    if not chat_id or not session.token:
        return False

    async def _run() -> bool:
        http = await client._ensure_http_session()
        status, _body = await request_json(
            http,
            "DELETE",
            f"{BASE_URL}{DELETE_CHAT_PATH.format(chat_id=chat_id)}",
            headers=build_headers(session.token),
            timeout=upstream_timeout(15.0),
        )
        return status in (200, 204)

    try:
        return await run_with_connection_retry(
            "delete_chat", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("Delete chat failed chat=%s: %s", chat_id[:8], exc)
        return False


async def abort_upstream_on_cancel(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    response_id: str = "",
) -> None:
    if not chat_id:
        return
    stopped = await stop_upstream_generation(client, session, chat_id, response_id)
    if stopped:
        logger.info(
            "Stopped upstream generation [%s] chat=%s",
            session.username[:6],
            chat_id[:8],
        )
    await delete_upstream_chat(client, session, chat_id)


async def handle_chat_error(client: QwenClient, resp: aiohttp.ClientResponse, session: QwenSession) -> None:
    raise_qwen_session_error(client, session, "", http_status=resp.status)
    body = await resp.text()
    if resp.status == 413:
        raise PayloadTooLargeError(f"Payload too large: {body[:200]}")
    raise_qwen_session_error(client, session, body)
    logger.error("Chat HTTP %d: %s", resp.status, body[:500])
    raise RuntimeError(f"Chat HTTP {resp.status}: {body[:200]}")


def check_sse_error_line(client: QwenClient, line: str, session: QwenSession) -> None:
    """解析 SSE 流中的 inline JSON 错误（含 Baxia punish / success:false）。"""
    raise_sse_inline_error(client, session, line)


_BAXIA_SM_MARKERS: frozenset[str] = frozenset(
    {"RGV587", "FAIL_SYS", "FAIL_SYS_USER_VALIDATE", "RGV587_ERROR::SM"}
)
_UPSTREAM_RATE_LIMIT_CODES: frozenset[str] = frozenset(
    {"RateLimited", "ParallelLimited", "quotaLimited", "Too_Many_Requests"}
)


def _is_baxia_sm_block(message: str, *, punish_url: str = "") -> bool:
    """Baxia SM 人机验证拦截（非业务 RateLimited）。"""
    if punish_url and any(marker in message for marker in _BAXIA_SM_MARKERS):
        return True
    return "RGV587_ERROR::SM" in message or "FAIL_SYS_USER_VALIDATE" in message


def _raise_for_success_false(
    client: "QwenClient",
    session: QwenSession,
    obj: Dict[str, Any],
) -> None:
    msg = json.dumps(obj, ensure_ascii=False)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    code = str(data.get("code") or "")
    if code in _UPSTREAM_RATE_LIMIT_CODES:
        client._invalidate_session(session)
        logger.warning("Session %s upstream rate limited (%s)", session.username[:6], code)
        raise TokenExpiredError(f"Rate limited: {msg[:200]}")
    raise_qwen_session_error(client, session, msg)
    raise RuntimeError(f"Qwen API error: {msg}")


def raise_sse_inline_error(
    client: "QwenClient",
    session: QwenSession,
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
    session: QwenSession,
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
    session: QwenSession,
    line: str,
    assembler: SseEventAssembler,
    response_id_out: Optional[list],
) -> Optional[Dict[str, Any]]:
    payload = assembler.feed_line(line)
    if payload is None:
        if line and not line.startswith("data:") and not line.startswith(":"):
            check_sse_error_line(client, line, session)
        return None
    return _event_from_sse_data(client, session, payload, response_id_out)


async def iter_sse_events(
    client: "QwenClient",
    resp: aiohttp.ClientResponse,
    session: QwenSession,
    *,
    response_id_out: Optional[list] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """逐行解析 SSE；对齐前端 kT/_T 组帧 + TCP chunk 行缓冲。"""
    pending = ""
    assembler = SseEventAssembler()
    try:
        async for raw in resp.content:
            append_sse_bytes(raw)
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
