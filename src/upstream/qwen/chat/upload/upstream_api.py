from __future__ import annotations

"""Qwen 上游辅助 API：config、parse_url、SSE 重连、登录 warm-up。"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List

from upstream.qwen.auth.crypto import build_headers, merge_session_cookies
from upstream.qwen.chat.routes import (
    BASE_URL,
    CHAT_PATH,
    CONFIGS_PATH,
    PARSE_URL_PATH,
    SETTINGS_PATH,
    SSE_RECONNECT_MAX,
)
from upstream.qwen.chat.chat import iter_sse_events
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import request_json, upstream_timeout

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")


async def fetch_app_config(client: "QwenClient", session: "QwenSession") -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "GET",
            f"{BASE_URL}{CONFIGS_PATH}",
            headers=build_headers(
                session.token,
                username=session.username,
                cookies=merge_session_cookies(session.token),
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
            headers=build_headers(
                session.token,
                username=session.username,
                cookies=merge_session_cookies(session.token),
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
    """登录后拉取 configs/settings（对齐 FE 启动序列）。"""
    await fetch_app_config(client, session)
    await fetch_user_settings(client, session)


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
            headers=build_headers(
                session.token,
                username=session.username,
                cookies=merge_session_cookies(session.token),
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
) -> AsyncGenerator[Dict[str, Any], None]:
    last_exc: Exception | None = None
    for attempt in range(SSE_RECONNECT_MAX):
        try:
            async for event in reconnect_sse_events(client, session, chat_id, response_id):
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
) -> AsyncGenerator[Dict[str, Any], None]:
    """GET /chat/completions?chat_id=&response_id= 断线续流。"""
    http = await client._ensure_http_session()
    async with http.get(
        f"{BASE_URL}{CHAT_PATH}",
        params={"chat_id": chat_id, "response_id": response_id},
        headers=build_headers(
            session.token,
            username=session.username,
            chat_id=chat_id,
            include_sse=True,
            cookies=merge_session_cookies(session.token),
        ),
        timeout=upstream_timeout(600.0),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"SSE reconnect HTTP {resp.status}: {body[:200]}")
        async for event in iter_sse_events(client, resp, session):
            yield event
