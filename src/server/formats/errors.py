from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionError, ServerDisconnectedError


class PayloadTooLargeError(RuntimeError):
    """上游 Qwen 拒绝请求体过大（HTTP 413）。"""


class UpstreamTimeoutError(RuntimeError):
    """上游 HTTP / SSE 读超时。"""


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
