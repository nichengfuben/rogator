from __future__ import annotations

"""Handler 层共享的 HTTP / SSE 错误响应映射。"""

import asyncio

from aiohttp import web

from echotools.logger import get_logger
from server.formats import TokenExpiredError, UpstreamTimeoutError, _error_response
from server.model.model_registry import ModelResolveError
from state import QueueFullError

logger = get_logger("rogator")


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
