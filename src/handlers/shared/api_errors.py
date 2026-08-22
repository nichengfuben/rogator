from __future__ import annotations

"""Handler 层共享：HTTP/SSE 错误映射、模型解析、断连安全 write。"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from aiohttp import web

from echotools.base.logger import get_logger
from server.formats import (
    BaxiaSmBlockedError,
    TokenExpiredError,
    UpstreamConnectionError,
    UpstreamStsError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    UpstreamWafBlockedError,
    UpstreamChatNotFoundError,
    _error_response,
    _json_response,
)
from server.model.model_registry import ModelResolveError, resolve_request_model
from state import AppState, QueueFullError

logger = get_logger("rogator")

_ANTHROPIC_ERROR_TYPES = {
    "rate_limited": "rate_limit_error",
    "timeout": "timeout_error",
    "server_error": "api_error",
    "invalid_request_error": "invalid_request_error",
}


@dataclass(frozen=True)
class StreamErrorInfo:
    """流式错误分类：协议层只负责按 kind 写 SSE。"""

    kind: str
    message: str
    code: int


def classify_stream_error(exc: BaseException) -> StreamErrorInfo:
    if isinstance(exc, TokenExpiredError):
        return StreamErrorInfo("rate_limited", str(exc), 429)
    if isinstance(exc, BaxiaSmBlockedError):
        return StreamErrorInfo("server_error", f"Baxia SM blocked: {exc}", 503)
    if isinstance(exc, UpstreamTimeoutError):
        return StreamErrorInfo("timeout", str(exc), 504)
    if isinstance(exc, UpstreamStsError):
        # STS 取 token 失败属于连接级错误，重试后仍失败则不影响客户端使用 502
        return StreamErrorInfo("server_error", str(exc), exc.status)
    if isinstance(exc, UpstreamUnavailableError) and "429" in str(exc):
        return StreamErrorInfo("rate_limited", exc.message, 429)
    if isinstance(exc, (UpstreamWafBlockedError, UpstreamUnavailableError, UpstreamChatNotFoundError)):
        return StreamErrorInfo("server_error", exc.message, exc.status)
    if isinstance(exc, UpstreamConnectionError):
        return StreamErrorInfo("server_error", exc.message, exc.status)
    return StreamErrorInfo("server_error", str(exc), 500)


def log_classified_stream_error(exc: BaseException, *, label: str) -> StreamErrorInfo:
    """分类流式异常并按 kind 打日志；协议层只负责写 SSE。"""
    info = classify_stream_error(exc)
    # 已知可重试类错误(STS=连接错误)，穷尽后单行 WARN 即可，不输出完整栈
    if isinstance(exc, UpstreamStsError):
        logger.warning("%s: %s", label, exc)
        return info
    if isinstance(exc, UpstreamChatNotFoundError):
        logger.warning("%s: %s", label, exc)
        return info
    if isinstance(exc, UpstreamConnectionError):
        logger.warning("%s upstream connection: %s", label, exc.message)
        return info
    if isinstance(exc, BaxiaSmBlockedError):
        logger.debug("%s Baxia SM blocked: %s", label, exc)
    elif info.kind == "rate_limited":
        logger.debug("%s rate limited: %s", label, exc)
    elif info.kind == "timeout":
        logger.warning("%s upstream timeout: %s", label, exc)
    elif isinstance(exc, UpstreamUnavailableError) and getattr(exc, "upstream", None) == "zen":
        logger.debug("%s zen upstream error: %s", label, exc.message)
    else:
        logger.error("%s error: %s", label, exc, exc_info=True)
    return info


def anthropic_error_type(kind: str) -> str:
    if kind in _ANTHROPIC_ERROR_TYPES.values():
        return kind
    return _ANTHROPIC_ERROR_TYPES.get(kind, "api_error")


def anthropic_error_event(info: StreamErrorInfo) -> Dict[str, Any]:
    """Anthropic 流式 error 事件体。"""
    return {
        "type": "error",
        "error": {
            "type": anthropic_error_type(info.kind),
            "message": info.message,
        },
    }


def anthropic_error_response(
    status: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> web.Response:
    """Anthropic HTTP 错误 envelope：``{type:error, error:{type,message}}``。"""
    return _json_response(
        {
            "type": "error",
            "error": {
                "type": anthropic_error_type(error_type),
                "message": message,
            },
        },
        status=status,
    )


def apply_tool_choice(
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Any,
) -> List[Dict[str, Any]]:
    """``none`` 清空工具；其余（auto/any/tool/required）保留（上游无强制能力）。"""
    tools_list = list(tools or [])
    if tool_choice is None or tool_choice == "auto":
        return tools_list
    if tool_choice == "none":
        return []
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        return []
    return tools_list


def require_anthropic_max_tokens(body: dict) -> Union[int, web.Response]:
    """Anthropic 官方必填 max_tokens；返回正整数或 400 响应。"""
    raw = body.get("max_tokens")
    if raw is None:
        return anthropic_error_response(400, "max_tokens is required")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return anthropic_error_response(400, "max_tokens must be an integer")
    if value < 1:
        return anthropic_error_response(400, "max_tokens must be >= 1")
    return value


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


def model_resolve_error_response(
    exc: ModelResolveError, *, protocol: str = "openai",
) -> web.Response:
    if protocol == "anthropic":
        return anthropic_error_response(exc.status, exc.message, exc.error_type)
    return _error_response(exc.status, exc.message, exc.error_type)


def handler_error_response(
    exc: BaseException, *, label: str, protocol: str = "openai",
) -> web.Response:
    """将 handler 常见异常映射为 aiohttp 响应（非流式）。"""
    err = anthropic_error_response if protocol == "anthropic" else _error_response
    if isinstance(exc, ModelResolveError):
        return model_resolve_error_response(exc, protocol=protocol)
    if isinstance(exc, QueueFullError):
        return err(503, str(exc) or "Busy", "api_error" if protocol == "anthropic" else "server_error")
    if isinstance(exc, asyncio.CancelledError):
        return err(503, "Shutting down", "api_error" if protocol == "anthropic" else "server_error")
    if isinstance(exc, TokenExpiredError):
        logger.warning("%s token expired: %s", label, exc)
        kind = "rate_limit_error" if protocol == "anthropic" else "rate_limited"
        return err(429, str(exc), kind)
    if isinstance(exc, BaxiaSmBlockedError):
        logger.debug("%s Baxia SM blocked: %s", label, exc)
        kind = "api_error" if protocol == "anthropic" else "server_error"
        return err(503, f"Baxia SM blocked: {exc}", kind)
    if isinstance(exc, UpstreamTimeoutError):
        logger.warning("%s upstream timeout: %s", label, exc)
        kind = "timeout_error" if protocol == "anthropic" else "timeout"
        return err(504, str(exc), kind)
    if isinstance(exc, (UpstreamWafBlockedError, UpstreamUnavailableError)):
        logger.warning("%s upstream unavailable: %s", label, exc.message)
        kind = "api_error" if protocol == "anthropic" else "server_error"
        return err(exc.status, exc.message, kind)
    if isinstance(exc, UpstreamConnectionError):
        logger.warning("%s upstream connection: %s", label, exc.message)
        kind = "api_error" if protocol == "anthropic" else "server_error"
        return err(exc.status, exc.message, kind)
    if isinstance(exc, UpstreamChatNotFoundError):
        # CHAT_NOT_FOUND 重试耗尽后属于业务错误，单行 WARN，不输出完整栈
        logger.warning("%s stream error: %s", label, exc)
        return err(exc.status, exc.message, "api_error" if protocol == "anthropic" else "server_error")
    logger.error("%s error: %s", label, exc, exc_info=True)
    return err(500, str(exc), "api_error" if protocol == "anthropic" else "server_error")
