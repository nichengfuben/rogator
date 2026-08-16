from __future__ import annotations

"""Zen chat completions 流式请求与 OpenAI SSE 解析。"""

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
from upstream.zen.openai_chat import build_headers
from upstream.zen.proxy import ZenProxyError, is_proxy_error
from upstream.zen.routes import (
    BASE_URL,
    CHAT_PATH,
    CONNECT_TIMEOUT,
    STREAM_READ_TIMEOUT,
    STREAM_TOTAL_TIMEOUT,
)

if TYPE_CHECKING:
    from upstream.zen.client import ZenClient

logger = logging.getLogger("rogator")


def _safe_loads(data_str: str) -> Optional[Any]:
    if not data_str or data_str == "[DONE]":
        return None
    try:
        return json.loads(data_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _extract_error(obj: Dict[str, Any]) -> Optional[str]:
    err = obj.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if err is not None:
        return str(err)
    if obj.get("type") == "error":
        nested = obj.get("error")
        if isinstance(nested, dict):
            return str(nested.get("message") or nested)
        return str(obj)
    return None


def _delta_events(delta: Dict[str, Any]) -> list:
    events = []
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        events.append({"type": "thinking", "content": reasoning})
    content = delta.get("content")
    if isinstance(content, str) and content:
        events.append({"type": "answer", "content": content})
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                events.append({"type": "tool_call", "tool_call": tc, "native": True})
    return events


def parse_openai_sse_data(data_str: str) -> list:
    obj = _safe_loads(data_str)
    if not isinstance(obj, dict):
        return []
    err_msg = _extract_error(obj)
    if err_msg is not None:
        return [{"type": "error", "message": err_msg}]
    events = []
    usage = obj.get("usage")
    choices = obj.get("choices") or []
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta") or {}
        if isinstance(delta, dict):
            events.extend(_delta_events(delta))
        message = choices[0].get("message") or {}
        if isinstance(message, dict) and not delta:
            events.extend(_delta_events(message))
    if usage and isinstance(usage, dict):
        events.append({"type": "usage", "data": usage})
    return events


class SseLineAssembler:
    __slots__ = ("_data_parts",)

    def __init__(self) -> None:
        self._data_parts: list = []

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

    def flush_eof(self) -> Optional[str]:
        return self._flush()

    def _flush(self) -> Optional[str]:
        if not self._data_parts:
            return None
        payload = "\n".join(self._data_parts)
        self._data_parts.clear()
        return payload


def has_valid_sse_event(data: bytes) -> bool:
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("data:"):
            after = stripped[5:].strip()
            if after and after != "[DONE]":
                return True
    return False


def extract_error_info(data: Any) -> Optional[Dict[str, str]]:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if data.get("type") == "error" and not isinstance(error_obj, dict):
        error_obj = data.get("error")
    if error_obj is None:
        return None
    if not isinstance(error_obj, dict):
        return {"type": "", "message": str(error_obj)}
    result = {
        "type": str(error_obj.get("type", "") or ""),
        "message": str(error_obj.get("message", "") or error_obj),
    }
    param = error_obj.get("param")
    if param:
        result["param"] = str(param)
    return result


def is_model_error(err_info: Dict[str, str]) -> bool:
    err_type = (err_info.get("type") or "").lower()
    err_msg = (err_info.get("message") or "").lower()
    if "modelerror" in err_type:
        return True
    return "not supported" in err_msg and "model" in err_msg


def is_validation_error(err_info: Dict[str, str]) -> bool:
    if err_info.get("param"):
        return True
    msg = (err_info.get("message") or "").lower()
    markers = (
        "param incorrect", "missing function.name", "invalid_request",
        "invalid request", "bad request", "is missing",
    )
    return any(kw in msg for kw in markers)


def classify_http_error(
    status: int,
    err_info: Optional[Dict[str, str]],
    raw: str,
) -> Exception:
    from upstream.zen.client import ZenModelNotSupportedError, ZenValidationError

    msg = err_info["message"] if err_info else (raw or "")[:300]
    if err_info and is_model_error(err_info):
        return ZenModelNotSupportedError(msg)
    if status == 401:
        return ZenModelNotSupportedError(f"Model requires authentication (401): {msg}")
    if status == 400 and err_info and is_validation_error(err_info):
        return ZenValidationError(msg)
    if status == 429:
        return UpstreamUnavailableError(f"HTTP 429 - {msg}", upstream="zen")
    return UpstreamUnavailableError(f"HTTP {status} - {msg}", upstream="zen")


async def raise_for_bad_status(resp: aiohttp.ClientResponse) -> None:
    text = await resp.text()
    err_info = None
    try:
        err_info = extract_error_info(json.loads(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    raise classify_http_error(resp.status, err_info, text)


async def iter_sse_events(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Dict[str, Any], None]:
    pending = ""
    assembler = SseLineAssembler()
    buffer = b""
    got_valid = False
    try:
        async for raw in resp.content.iter_any():
            if not raw:
                continue
            buffer += raw
            pending += raw.decode("utf-8", errors="replace")
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                event_payload = assembler.feed_line(line.rstrip("\r"))
                if event_payload is None:
                    continue
                async for event in _yield_parsed(event_payload):
                    got_valid = True
                    yield event
        async for event in _flush_tail(assembler, pending):
            got_valid = True
            yield event
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError("Zen SSE read timed out") from exc
    if buffer and not got_valid and not has_valid_sse_event(buffer):
        raise UpstreamUnavailableError(
            "Empty stream response: no valid events",
            upstream="zen",
        )


async def _yield_parsed(payload: str) -> AsyncGenerator[Dict[str, Any], None]:
    for event in parse_openai_sse_data(payload):
        if event.get("type") == "error":
            raise UpstreamUnavailableError(
                str(event.get("message") or "zen sse error"),
                upstream="zen",
            )
        yield event


async def _flush_tail(
    assembler: SseLineAssembler,
    pending: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    tail = pending.rstrip("\r")
    if tail:
        payload = assembler.feed_line(tail)
        if payload is None:
            payload = assembler.flush_eof()
        else:
            assembler.flush_eof()
        if payload:
            async for event in _yield_parsed(payload):
                yield event
        return
    eof = assembler.flush_eof()
    if eof:
        async for event in _yield_parsed(eof):
            yield event


def _map_request_error(exc: Exception, *, proxy: Optional[str]) -> Exception:
    if isinstance(exc, (ZenProxyError, UpstreamUnavailableError)):
        return exc
    if isinstance(exc, asyncio.TimeoutError):
        if proxy:
            return ZenProxyError("Proxy timeout: {}".format(exc))
        return UpstreamTimeoutError("Zen request timed out")
    if isinstance(exc, aiohttp.ClientProxyConnectionError):
        return ZenProxyError("Proxy connection failed: {}".format(exc))
    if isinstance(exc, aiohttp.ClientConnectorError):
        if proxy or "proxy" in str(exc).lower():
            return ZenProxyError("Proxy connection error: {}".format(exc))
        return UpstreamConnectionError(str(exc), upstream="zen")
    if is_proxy_error(exc):
        return ZenProxyError("Proxy error: {}".format(exc))
    return UpstreamConnectionError(str(exc), upstream="zen")


async def post_chat_stream(
    client: "ZenClient",
    payload: Dict[str, Any],
    *,
    proxy: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    from upstream.zen.client import ZenModelNotSupportedError, ZenValidationError

    url = f"{BASE_URL}{CHAT_PATH}"
    timeout = aiohttp.ClientTimeout(
        total=STREAM_TOTAL_TIMEOUT,
        connect=CONNECT_TIMEOUT,
        sock_read=STREAM_READ_TIMEOUT,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http = await client._ensure_http_session()
    req_kw: Dict[str, Any] = {
        "data": body,
        "headers": build_headers(stream=True),
        "timeout": timeout,
    }
    if proxy:
        req_kw["proxy"] = proxy
    try:
        async with http.post(url, **req_kw) as resp:
            if resp.status != 200:
                await raise_for_bad_status(resp)
            async for event in iter_sse_events(resp):
                yield event
    except (ZenModelNotSupportedError, ZenValidationError, UpstreamUnavailableError):
        raise
    except Exception as exc:
        raise _map_request_error(exc, proxy=proxy) from exc
