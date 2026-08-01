from __future__ import annotations

"""Handler 层共享：HTTP/SSE 错误映射、模型解析、断连安全 write。"""

import asyncio
from dataclasses import dataclass

from aiohttp import web

from echotools.logger import get_logger
from server.formats import TokenExpiredError, UpstreamTimeoutError, _error_response
from server.model.model_registry import ModelResolveError, resolve_request_model
from state import AppState, QueueFullError

logger = get_logger("rogator")


@dataclass(frozen=True)
class StreamErrorInfo:
    """流式错误分类：协议层只负责按 kind 写 SSE。"""

    kind: str
    message: str
    code: int


def classify_stream_error(exc: BaseException) -> StreamErrorInfo:
    if isinstance(exc, TokenExpiredError):
        return StreamErrorInfo("rate_limited", str(exc), 429)
    if isinstance(exc, UpstreamTimeoutError):
        return StreamErrorInfo("timeout", str(exc), 504)
    return StreamErrorInfo("server_error", str(exc), 500)


def log_classified_stream_error(exc: BaseException, *, label: str) -> StreamErrorInfo:
    """分类流式异常并按 kind 打日志；协议层只负责写 SSE。"""
    info = classify_stream_error(exc)
    if info.kind == "rate_limited":
        logger.warning("%s token expired: %s", label, exc)
    elif info.kind == "timeout":
        logger.warning("%s upstream timeout: %s", label, exc)
    else:
        logger.error("%s error: %s", label, exc, exc_info=True)
    return info


async def safe_write(resp, data: bytes, disconnected: list) -> bool:
    if disconnected[0]:
        return False
    try:
        await resp.write(data)
        return True
    except (ConnectionError, OSError, asyncio.CancelledError):
        disconnected[0] = True
        return False


def resolve_handler_model(state: AppState, requested: str) -> str:
    """解析 API 模型外键，返回上游内键。"""
    return resolve_request_model(requested, state._models).internal_id


def model_resolve_error_response(exc: ModelResolveError) -> web.Response:
    return _error_response(exc.status, exc.message, exc.error_type)


def handler_error_response(exc: BaseException, *, label: str) -> web.Response:
    """将 handler 常见异常映射为 aiohttp 响应（非流式）。"""
    if isinstance(exc, ModelResolveError):
        return _error_response(exc.status, exc.message, exc.error_type)
    if isinstance(exc, QueueFullError):
        return web.Response(status=503, text=str(exc))
    if isinstance(exc, asyncio.CancelledError):
        return web.Response(status=503, text="Shutting down")
    if isinstance(exc, TokenExpiredError):
        logger.warning("%s token expired: %s", label, exc)
        return _error_response(429, str(exc), "rate_limited")
    if isinstance(exc, UpstreamTimeoutError):
        logger.warning("%s upstream timeout: %s", label, exc)
        return _error_response(504, str(exc), "timeout")
    logger.error("%s error: %s", label, exc, exc_info=True)
    return _error_response(500, str(exc))
