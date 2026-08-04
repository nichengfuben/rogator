from __future__ import annotations

"""Qwen chat creation, stop/delete, and SSE parsing."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import aiohttp

from upstream.qwen.auth.crypto import build_headers_async, build_stop_headers_async
from upstream.qwen.chat.sse import iter_sse_events, raise_sse_inline_error
from upstream.qwen.chat.routes import BASE_URL, DELETE_CHAT_PATH, NEW_CHAT_PATH, STOP_CHAT_PATH
from upstream.qwen.chat.upload.payload import build_stop_payload
from upstream.qwen.auth.http import run_with_connection_retry, absorb_response_cookies
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
        "Configure Baxia headers or re-login. "
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


async def _post_create_chat(
    client: QwenClient,
    session: QwenSession,
    model: str,
    timeout_s: float,
    cookies: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    from upstream.qwen.chat.upload.payload import build_new_chat_payload

    payload = build_new_chat_payload(model)
    if cookies is None:
        cookies_fn = getattr(client, "begin_chat_cookies", None) or getattr(
            client, "cookies_for_session", None,
        )
        if callable(cookies_fn):
            cookies = cookies_fn(session)
    headers = await build_headers_async(
        session.token,
        include_version=True,
        api_path=NEW_CHAT_PATH,
        cookies=cookies,
    )

    async def _run() -> Dict[str, Any]:
        http = await client._ensure_http_session()
        async with http.post(
            f"{BASE_URL}{NEW_CHAT_PATH}",
            headers=headers,
            json=payload,
            timeout=upstream_timeout(timeout_s),
        ) as resp:
            absorb_fn = getattr(client, "absorb_cookies_for_session", None)
            if callable(absorb_fn):
                absorb_fn(session, resp, binding=cookies)
            elif cookies is not None:
                absorb_response_cookies(cookies, resp)
            if resp.status != 200:
                return {"_http_status": resp.status}
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = await resp.text()
            if isinstance(body, dict):
                return body
            _raise_for_non_json_create_chat(client, session, str(body))
            return {}

    return await run_with_connection_retry("create_chat", _run, transport_owner=client)


async def create_chat_for_session(
    client: QwenClient,
    session: QwenSession,
    model: str,
    *,
    cookies: Optional[Dict[str, str]] = None,
) -> str:
    from upstream.qwen.auth.report import (
        report_after_chat_created,
        report_create_chat_sequence,
    )

    await report_create_chat_sequence(client, session)
    timeout_s = CONFIG.create_chat_timeout
    try:
        data = await _post_create_chat(client, session, model, timeout_s, cookies=cookies)
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
    await report_after_chat_created(client, session, chat_id)
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
            headers=await build_stop_headers_async(session.token),
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
            headers=await build_headers_async(session.token),
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
