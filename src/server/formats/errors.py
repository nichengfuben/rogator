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


class UpstreamWafBlockedError(UpstreamUnavailableError):
    """上游 WAF/Baxia 拦截，返回 HTML 等非 JSON 响应。"""

    error_type: str = "upstream_waf_blocked"


class UpstreamChatNotFoundError(UpstreamUnavailableError):
    """上游 chat_id 不存在（常见为建聊与发消息 Cookie 会话不一致）。"""

    error_type: str = "upstream_chat_not_found"


class UpstreamConnectionError(RuntimeError):
    """?????????/??/DNS??"""

    status: int = 502
    error_type: str = "upstream_connection_error"

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.upstream = upstream


class UpstreamStsError(UpstreamConnectionError):
    """上游 STS 取 token 全部端点失败（代理/网络不可达）。"""

    error_type: str = "upstream_sts_error"


class TokenExpiredError(Exception):
    """Token ??????? session"""


class BaxiaSmBlockedError(Exception):
    """Baxia SM 人机验证拦截：账号仍有效，换号重试即可。"""


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
    # body 而非 text：避免 aiohttp 追加 "; charset=utf-8"（官方为纯 application/json）；
    # 统一带伪造的 Cloudflare 边缘头，掩盖 aiohttp 特征
    from server.formats.headers import cloudflare_headers

    return web.Response(
        status=status,
        body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={**cloudflare_headers(), "Content-Type": "application/json"},
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


from echotools.exec.fncall.tool_id import fix_tool_call_id


def _connection_error_message(hint: str, *, upstream: str = "") -> str:
    if upstream:
        return "{0} 连接失败: {1}".format(upstream, hint)
    return "上游连接失败: {0}".format(hint)


def _traceback_touches_aiohttp_client(exc: BaseException) -> bool:
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename.replace("\\", "/").endswith("aiohttp/client.py"):
            return True
        tb = tb.tb_next
    return False


def _is_stale_http_session_error(exc: BaseException) -> bool:
    """识别 aiohttp ClientSession 被并发 reset/close 后的典型异常。"""
    if isinstance(exc, RuntimeError):
        text = str(exc).strip().lower()
        return "session is closed" in text or "connector is closed" in text
    if isinstance(exc, AttributeError):
        text = str(exc).strip()
        return "_timeout_ceil_threshold" in text
    if isinstance(exc, AssertionError) and _traceback_touches_aiohttp_client(exc):
        # aiohttp 在 session._connector 已被 detach 后以 post/get 进入时会 assert
        return True
    return False


def as_upstream_connection_error(
    exc: BaseException,
    *,
    upstream: str = "",
) -> Optional[UpstreamConnectionError]:
    # TimeoutError 走独立超时重试路径，不当作连接错误。
    if isinstance(exc, asyncio.TimeoutError):
        return None
    if _is_stale_http_session_error(exc):
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
        hint = str(exc).strip() or exc.__class__.__name__
        return UpstreamConnectionError(
            _connection_error_message(hint, upstream=upstream),
            upstream=upstream,
        )
    return None
