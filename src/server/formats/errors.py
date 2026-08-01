from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from aiohttp import web
from aiohttp.client_exceptions import (
    ClientConnectionError,
    ClientConnectorError,
    ClientError,
    ServerConnectionError,
    ServerDisconnectedError,
)


class PayloadTooLargeError(RuntimeError):
    """上游 Qwen 拒绝请求体过大（HTTP 413）。"""


class UpstreamTimeoutError(RuntimeError):
    """上游 HTTP / SSE 读超时。"""


class UpstreamUnavailableError(RuntimeError):
    """上游无可用会话、账号或凭证。"""

    status: int = 503
    error_type: str = "upstream_unavailable"

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.upstream = upstream


class UpstreamConnectionError(RuntimeError):
    """无法连接上游（网络/代理/DNS）。"""

    status: int = 502
    error_type: str = "upstream_connection_error"

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.upstream = upstream


class TokenExpiredError(Exception):
    """Token 过期，需要切换 session"""


class ClientDisconnectedError(Exception):
    """客户端在请求体读完之前关闭连接。"""


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
    """读取 JSON 请求体；客户端断连时抛出 ``ClientDisconnectedError``。"""
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
    """客户端已断开时的响应（499 Client Closed Request）。"""
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
    """替换 echotools 硬编码的 call_0000 为唯一 UUID。"""
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


def as_upstream_connection_error(
    exc: BaseException,
    *,
    upstream: str = "",
) -> UpstreamConnectionError | None:
    # Py3.8+：TimeoutError 是 OSError 子类，须先于 OSError 排除，否则会误判为连接失败。
    if isinstance(exc, asyncio.TimeoutError):
        return None
    if isinstance(exc, (ClientConnectorError, ServerConnectionError, ConnectionResetError)):
        hint = str(exc).strip() or exc.__class__.__name__
        if upstream:
            msg = "{0} 连接失败: {1}".format(upstream, hint)
        else:
            msg = "上游连接失败: {0}".format(hint)
        return UpstreamConnectionError(msg, upstream=upstream)
    if isinstance(exc, OSError):
        hint = str(exc).strip() or exc.__class__.__name__
        if upstream:
            msg = "{0} 连接失败: {1}".format(upstream, hint)
        else:
            msg = "上游连接失败: {0}".format(hint)
        return UpstreamConnectionError(msg, upstream=upstream)
    if isinstance(exc, ClientError) and not isinstance(exc, ClientConnectorError):
        return UpstreamConnectionError(str(exc), upstream=upstream)
    return None
