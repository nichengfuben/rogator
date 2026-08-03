from __future__ import annotations

"""Qwen 文档解析：POST /files/parse + 轮询 /files/parse/status。"""

import asyncio
import logging
from typing import TYPE_CHECKING, List

from upstream.qwen.auth.crypto import build_headers, merge_session_cookies
from upstream.qwen.chat.routes import (
    BASE_URL,
    FILE_PARSE_POLL_INTERVAL,
    FILE_PARSE_TIMEOUT,
    PARSE_FILE_PATH,
    PARSE_STATUS_PATH,
)
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import request_json, upstream_timeout

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")


async def trigger_file_parse(
    client: "QwenClient",
    session: "QwenSession",
    file_id: str,
) -> bool:
    if not file_id:
        return False

    async def _run() -> bool:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{PARSE_FILE_PATH}",
            headers=build_headers(
                session.token,
                cookies=merge_session_cookies(session.token),
            ),
            json={"file_id": file_id},
            timeout=upstream_timeout(30.0),
        )
        if status != 200 or not isinstance(body, dict):
            return False
        return bool(body.get("success"))

    try:
        return await run_with_connection_retry(
            "file_parse", _run, transport_owner=client,
        )
    except Exception as exc:
        logger.debug("trigger_file_parse failed file=%s: %s", file_id[:8], exc)
        return False


async def poll_parse_status(
    client: "QwenClient",
    session: "QwenSession",
    file_ids: List[str],
) -> dict[str, str]:
    async def _run() -> dict[str, str]:
        http = await client._ensure_http_session()
        status, body = await request_json(
            http,
            "POST",
            f"{BASE_URL}{PARSE_STATUS_PATH}",
            headers=build_headers(
                session.token,
                cookies=merge_session_cookies(session.token),
            ),
            json={"file_id_list": file_ids},
            timeout=upstream_timeout(30.0),
        )
        out: dict[str, str] = {}
        if status != 200 or not isinstance(body, dict) or not body.get("success"):
            return out
        for item in body.get("data") or []:
            if isinstance(item, dict) and item.get("file_id"):
                out[str(item["file_id"])] = str(item.get("status") or "")
        return out

    return await run_with_connection_retry(
        "file_parse_status", _run, transport_owner=client,
    )


async def wait_file_parsed(
    client: "QwenClient",
    session: "QwenSession",
    file_id: str,
    *,
    timeout: float = FILE_PARSE_TIMEOUT,
    interval: float = FILE_PARSE_POLL_INTERVAL,
) -> bool:
    if not await trigger_file_parse(client, session, file_id):
        return False
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        statuses = await poll_parse_status(client, session, [file_id])
        st = statuses.get(file_id, "")
        if st == "success":
            return True
        if st == "failed":
            return False
        await asyncio.sleep(interval)
    return False
