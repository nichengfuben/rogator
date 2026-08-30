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
    """客户端请求体超过上限；上游 Qwen 触发 HTTP 413 时抛出。"""


class UpstreamTimeoutError(RuntimeError):
    """上游 HTTP / SSE 读超时统一抛出。"""


class UpstreamUnavailableError(RuntimeError):
    """上游不可用（业务层语义，非网络层）。"""

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
    """上游连接失败：TCP/SSL/DNS 错误统一映射。"""

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
    """Token 失效，需切换/重建 session。"""


class BaxiaSmBlockedError(Exception):
    """Baxia SM 人机验证拦截：账号仍有效，换号重试即可。"""

    def __init__(
        self,
        message: str = "",
        *,
        proxy_used_enabled: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.proxy_used_enabled = proxy_used_enabled


class UpstreamWafBlockedErrorWithProxy(UpstreamWafBlockedError):
    """携带请求期代理开关快照的 WAF 拦截异常。"""

    def __init__(
        self,
        message: str,
        *,
        upstream: str = "",
        proxy_used_enabled: Optional[bool] = None,
    ) -> None:
        super().__init__(message, upstream=upstream)
        self.proxy_used_enabled = proxy_used_enabled


def attach_proxy_toggle(exc: BaseException, used_enabled: Optional[bool]) -> BaseException:
    """把请求期的代理开关快照挂到现有异常上；不改原异常类型。

    兼容老代码 ``getattr(exc, "_proxy_used_enabled", None)`` 的同时，
    优先返回标准化字段 ``proxy_used_enabled``。``None`` 表示"未知"。
    """
    if used_enabled is None:
        return exc
    try:
        setattr(exc, "proxy_used_enabled", used_enabled)
        setattr(exc, "_proxy_used_enabled", used_enabled)  # noqa: legacy alias
    except Exception:
        pass
    return exc


def read_proxy_used_enabled(exc: BaseException, client: Any) -> Optional[bool]:
    """按"显式字段 → client 当前值 → None"顺序解析请求期代理开关。"""
    val = getattr(exc, "proxy_used_enabled", None)
    if val is None:
        val = getattr(exc, "_proxy_used_enabled", None)
    if val is not None:
        return val
    if client is not None:
        val = getattr(client, "_last_used_proxy_enabled", None)
        if val is not None:
            return val
    try:
        from upstream.qwen.media.proxy_toggle import get_proxy_toggle
        return get_proxy_toggle().enabled
    except Exception:
        return None


class DataInspectionFailedError(Exception):
    """Qwen 内容安全拦截（data_inspection_failed）：输入内容违规，直接透传给请求者。

    不属于账号/网络问题，不能换号重试，也不触发代理切换。
    """

    def __init__(self, message: str, *, code: str = "", stage: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


class ClientDisconnectedError(Exception):
    """客户端在响应写出前断开连接，handler 据此返 499。"""


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
    """读取请求 JSON；空 body 视为 ``{}``，客户端断连抛 ``ClientDisconnectedError``。

    Content-Length 与实际 body 不一致（恶意/损坏请求）直接抛 ``web.HTTPBadRequest``，
    让上游网关明确返 4xx 而非吞掉错误。
    """
    if not request.can_read_body:
        cl = request.headers.get("Content-Length")
        if cl is not None and cl.strip() != "0":
            raise web.HTTPBadRequest(
                reason="Content-Length declared but body is empty",
            )
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
    """返回 499 Client Closed Request，匹配 nginx 语义。"""
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
