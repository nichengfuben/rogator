from __future__ import annotations

"""Qwen 文档解析与 SSE 事件解析。"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from upstream.qwen.auth.crypto import build_headers_async, merge_session_cookies
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
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or "")
                ),
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
            headers=await build_headers_async(
                session.token,
                cookies=merge_session_cookies(
                    session.token, user_id=str(session.user_id or "")
                ),
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


def _safe_loads(data_str: str) -> Optional[Any]:
    if not data_str or data_str == "[DONE]":
        return None
    try:
        return json.loads(data_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _parse_head_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "error" in data:
        return {"type": "error", "message": str(data["error"])}
    created = data.get("response.created")
    if isinstance(created, dict):
        return {
            "type": "response_created",
            "response_id": created.get("response_id", ""),
            "response_index": created.get("response_index"),
        }
    stopped = data.get("response.stopped")
    if isinstance(stopped, dict):
        return {
            "type": "response_stopped",
            "response_id": stopped.get("response_id", ""),
        }
    info = data.get("response.info")
    if isinstance(info, dict) and info.get("response_id"):
        return {
            "type": "response_info",
            "response_id": info.get("response_id", ""),
        }
    return None


def _build_answer(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if content and delta.get("status") != "finished":
        return {"type": "answer", "content": content}
    return None


def _build_thinking(delta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "thinking_summary",
        "status": delta.get("status") or "",
        "extra": delta.get("extra", {}),
    }


def _build_image_tool(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if delta.get("role") != "function" or delta.get("status") != "finished":
        return None
    extra = delta.get("extra", {})
    imgs = extra.get("image_list", extra.get("tool_result", []))
    urls = [
        img.get("image", "")
        for img in imgs
        if isinstance(img, dict) and img.get("image")
    ]
    if not urls:
        return None
    return {"type": "image_gen_tool", "urls": urls}


def _build_image_gen(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if not content:
        return None
    return {"type": "image_gen", "content": content, "extra": delta.get("extra", {})}


def _build_video_gen(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if not content:
        return None
    return {"type": "video_gen", "content": content}


def _build_other(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    phase = delta.get("phase")
    content = delta.get("content")
    status = delta.get("status")
    if phase is not None and phase != "" and content and status != "finished":
        return {"type": "other", "content": content}
    return None


def _build_web_search(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if content and delta.get("status") != "finished":
        return {"type": "thinking", "content": str(content)}
    extra = delta.get("extra") or {}
    info = extra.get("web_search_info") or []
    if isinstance(info, list) and info:
        parts = []
        for item in info:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("url") or "")
                snippet = str(item.get("snippet") or item.get("content") or "")
                parts.append(f"{title}: {snippet}".strip(": "))
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
        if text:
            return {"type": "thinking", "content": text}
    return None


def _dispatch_phase(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    phase = delta.get("phase")
    if phase == "KeepAlive":
        return None
    if phase == "answer":
        return _build_answer(delta)
    if phase == "think":
        content = delta.get("content")
        return {"type": "thinking", "content": content} if content else None
    if phase == "thinking_summary":
        return _build_thinking(delta)
    if phase == "web_search":
        return _build_web_search(delta)
    if phase == "image_gen_tool":
        return _build_image_tool(delta)
    if phase == "image_gen":
        return _build_image_gen(delta)
    if phase == "video_gen":
        return _build_video_gen(delta)
    return _build_other(delta)


def _parse_choice_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = data.get("usage")
    choices = data.get("choices", [])
    if not choices:
        return {"type": "usage", "data": usage} if usage else None
    delta = choices[0].get("delta", {})
    # role 校验：非 assistant role 视为异常信号
    role = delta.get("role")
    if role and role != "assistant":
        return {"type": "_qwen_function_role", "role": role}
    result = _dispatch_phase(delta)
    if usage:
        if result is None:
            return {"type": "usage", "data": usage}
        result["usage"] = usage
    return result


def parse_sse_event(data_str: str) -> Optional[Dict[str, Any]]:
    data = _safe_loads(data_str)
    if data is None:
        return None
    head = _parse_head_event(data)
    if head is not None:
        return head
    return _parse_choice_event(data)


class SseEventAssembler:
    __slots__ = ("_data_parts",)

    def __init__(self) -> None:
        self._data_parts: List[str] = []

    def feed_line(self, line: str) -> Optional[str]:
        if line == "":
            return self._flush()
        if line.startswith(":"):
            return None
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data_parts.append(value)
            return None
        return None

    def flush_eof(self) -> Optional[str]:
        return self._flush()

    def _flush(self) -> Optional[str]:
        if not self._data_parts:
            return None
        payload = "\n".join(self._data_parts)
        self._data_parts.clear()
        return payload


def parse_sse_line(data_str: str) -> Optional[Union[str, Dict[str, Any]]]:
    event = parse_sse_event(data_str)
    if event is None:
        return None
    if event["type"] == "answer":
        return event.get("content", "")
    if event["type"] == "thinking":
        return {"thinking": event.get("content", "")}
    if event["type"] == "thinking_summary":
        extra = event.get("extra", {})
        titles: List[str] = extra.get("summary_title", {}).get("content", [])
        thoughts: List[str] = extra.get("summary_thought", {}).get("content", [])
        if not titles and not thoughts:
            return None
        parts: List[str] = []
        for index in range(max(len(titles), len(thoughts))):
            title = titles[index] if index < len(titles) else ""
            thought = thoughts[index] if index < len(thoughts) else ""
            if title or thought:
                parts.append(f"{title}: {thought}" if title else thought)
        return {"thinking": "\n".join(parts)} if parts else None
    if event["type"] == "usage":
        return {"usage": event.get("data", {})}
    return None
