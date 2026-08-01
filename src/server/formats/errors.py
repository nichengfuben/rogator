from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from aiohttp import web
from aiohttp.client_exceptions import (
    ClientConnectionError,
    ClientConnectorError,
    ClientError,
    ServerConnectionError,
    ServerDisconnectedError,
)


class PayloadTooLargeError(RuntimeError):
    """?? Qwen ????????HTTP 413??"""


class UpstreamTimeoutError(RuntimeError):
    """?? HTTP / SSE ????"""


class UpstreamUnavailableError(RuntimeError):
    """??????????????"""

    status: int = 503
    error_type: str = "upstream_unavailable"

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.upstream = upstream


class UpstreamConnectionError(RuntimeError):
    """?????????/??/DNS??"""

    status: int = 502
    error_type: str = "upstream_connection_error"

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.upstream = upstream


class TokenExpiredError(Exception):
    """Token ??????? session"""


class ClientDisconnectedError(Exception):
    """????????????????"""


_CLIENT_DISCONNECT_ERRORS = (
    asyncio.CancelledError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    ConnectionError,
    ClientConnectionError,
    ServerDisconnectedError,
)


async def read_request_json(request: web.Request) -> Dict[str, Any]:
    """?? JSON ???????????? ``ClientDisconnectedError``?"""
    if not request.can_read_body:
        return {}
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise
    except _CLIENT_DISCONNECT_ERRORS as exc:
        raise ClientDisconnectedError() from exc
    if not isinstance(body, dict):
        return {}
    return body


def client_disconnected_response() -> web.Response:
    """???????????499 Client Closed Request??"""
    return web.Response(status=499, text="Client disconnected")


def json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(
        data, status=status,
        dumps=lambda x: json.dumps(x, ensure_ascii=False),
    )


def error_response(
    status: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> web.Response:
    return json_response(
        {"error": {"message": message, "type": error_type, "code": status}},
        status=status,
    )


def fix_tool_call_id(tc: Dict[str, Any]) -> Dict[str, Any]:
    """?? echotools ???? call_0000 ??? UUID?"""
    from server.formats.constants import gen_tool_id

    raw_id = tc.get("id", "")
    if (
        not raw_id
        or raw_id == "call_0000"
        or raw_id == "toolu_call_0001"
        or raw_id.startswith("toolu_call_")
        or raw_id.startswith("call_")
    ):
        call_id = gen_tool_id()
    else:
        call_id = raw_id
    func = tc.get("function", {})
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "arguments": func.get("arguments", "{}"),
        },
    }


def _connection_error_message(hint: str, *, upstream: str = "") -> str:
    if upstream:
        return "{0} 连接失败: {1}".format(upstream, hint)
    return "上游连接失败: {0}".format(hint)


def as_upstream_connection_error(
    exc: BaseException,
    *,
    upstream: str = "",
) -> Optional[UpstreamConnectionError]:
    # TimeoutError 走独立超时重试路径，不当作连接错误。
    if isinstance(exc, asyncio.TimeoutError):
        return None
    if isinstance(exc, RuntimeError):
        text = str(exc).strip().lower()
        if "session is closed" in text or "connector is closed" in text:
            return UpstreamConnectionError(
                _connection_error_message(str(exc).strip() or exc.__class__.__name__, upstream=upstream),
                upstream=upstream,
            )
    if isinstance(exc, (ClientConnectorError, ServerConnectionError, ConnectionResetError)):
        hint = str(exc).strip() or exc.__class__.__name__
        return UpstreamConnectionError(
            _connection_error_message(hint, upstream=upstream),
            upstream=upstream,
        )
    if isinstance(exc, OSError):
        hint = str(exc).strip() or exc.__class__.__name__
        return UpstreamConnectionError(
            _connection_error_message(hint, upstream=upstream),
            upstream=upstream,
        )
    if isinstance(exc, ClientError) and not isinstance(exc, ClientConnectorError):
        return UpstreamConnectionError(str(exc), upstream=upstream)
    return None
