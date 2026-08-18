from __future__ import annotations

"""Ollama 逐行 JSON 流解析与 HTTP POST 请求。"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, TYPE_CHECKING

import aiohttp

from server.formats import (
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from upstream.ollama.routes import (
    CHAT_PATH,
    CONNECT_TIMEOUT,
    STREAM_READ_TIMEOUT,
    STREAM_TOTAL_TIMEOUT,
)

if TYPE_CHECKING:
    from upstream.ollama.client import OllamaClient

logger = logging.getLogger("rogator")


def _extract_usage(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 Ollama done 响应中提取 usage 信息。"""
    prompt_eval = obj.get("prompt_eval_count")
    eval_count = obj.get("eval_count")
    if prompt_eval is None and eval_count is None:
        return None
    return {
        "prompt_tokens": prompt_eval or 0,
        "completion_tokens": eval_count or 0,
        "total_tokens": (prompt_eval or 0) + (eval_count or 0),
    }


def parse_ollama_line(obj: Dict[str, Any]) -> list[Dict[str, Any]]:
    """解析单行 Ollama JSON 响应为内部事件列表。"""
    events: list[Dict[str, Any]] = []
    err = obj.get("error")
    if err:
        events.append({"type": "error", "message": str(err)})
        return events

    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content:
            events.append({"type": "answer", "content": content})

    if obj.get("done"):
        usage = _extract_usage(obj)
        if usage:
            events.append({"type": "usage", "data": usage})

    return events


def _parse_ndjson_line(raw_line: str) -> list[Dict[str, Any]]:
    """解析单行 NDJSON 文本为事件列表；非 JSON 或非 dict 返回空列表。"""
    line = raw_line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []
    return parse_ollama_line(obj)


def _raise_on_error_event(event: Dict[str, Any]) -> None:
    """若事件为 error 类型则抛出 UpstreamUnavailableError。"""
    if event.get("type") != "error":
        return
    raise UpstreamUnavailableError(
        str(event.get("message") or "ollama stream error"),
        upstream="ollama",
    )


async def iter_ollama_lines(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Dict[str, Any], None]:
    """逐行读取 Ollama NDJSON 流并产出事件。"""
    buffer = ""
    got_valid = False
    try:
        async for raw in resp.content.iter_any():
            if not raw:
                continue
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for event in _parse_ndjson_line(line):
                    _raise_on_error_event(event)
                    got_valid = True
                    yield event
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError("Ollama stream read timed out") from exc

    for event in _parse_ndjson_line(buffer):
        _raise_on_error_event(event)
        got_valid = True
        yield event

    if not got_valid:
        raise UpstreamUnavailableError(
            "Empty stream response: no valid events",
            upstream="ollama",
        )


async def post_chat_stream(
    client: "OllamaClient",
    payload: Dict[str, Any],
    server_url: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """向指定 Ollama 服务器发送 chat 请求并流式返回事件。"""
    url = f"{server_url}{CHAT_PATH}"
    timeout = aiohttp.ClientTimeout(
        total=STREAM_TOTAL_TIMEOUT,
        connect=CONNECT_TIMEOUT,
        sock_read=STREAM_READ_TIMEOUT,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http = await client._ensure_http_session()
    headers = {"Content-Type": "application/json"}
    try:
        async with http.post(url, data=body, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise UpstreamUnavailableError(
                    f"HTTP {resp.status} - {text[:300]}",
                    upstream="ollama",
                )
            async for event in iter_ollama_lines(resp):
                yield event
    except UpstreamUnavailableError:
        raise
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError("Ollama request timed out") from exc
    except aiohttp.ClientConnectorError as exc:
        raise UpstreamConnectionError(str(exc), upstream="ollama") from exc
    except Exception as exc:
        raise UpstreamConnectionError(str(exc), upstream="ollama") from exc
