from __future__ import annotations

"""双协议 chat 请求解析、模型解析与上游 inject/截断。"""

import json
import logging
from contextlib import aclosing
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from aiohttp import web

from core.dispatch import resolve_upstream
from handlers import extract_system_for_inject
from handlers.openai.protocol import _inject_protocol_options
from handlers.openai.protocol import protocol_thinking_level
from handlers.openai.tools import convert_tools_to_openai
from handlers.shared.api_errors import (
    anthropic_error_response,
    model_resolve_error_response,
    resolve_handler_model,
)
from handlers.shared.fncall_inject import inject_fncall_for_request
from server.formats import (
    ClientDisconnectedError,
    _error_response,
    _gen_request_id,
    client_disconnected_response,
    read_request_json,
)
from server.model.model_registry import ModelResolveError
from server.model.model_thinking import ThinkingRoute, resolve_thinking_route
from server.retry import stream_with_session_retry

logger = logging.getLogger("rogator")

ChatJsonResult = Union[dict, web.Response]
ChatModelResult = Union[str, web.Response]
PrepareResult = Tuple[List[Dict[str, Any]], str, ThinkingRoute]
StreamEventHandler = Callable[[Dict[str, Any]], Awaitable[bool]]


async def read_chat_json(request: web.Request, *, protocol: str) -> ChatJsonResult:
    try:
        return await read_request_json(request)
    except ClientDisconnectedError:
        logger.info(
            "%s client disconnected while reading body from %s",
            protocol.capitalize(),
            request.remote,
        )
        return client_disconnected_response()
    except json.JSONDecodeError:
        if protocol == "anthropic":
            return anthropic_error_response(400, "Invalid JSON body")
        return _error_response(400, "Invalid JSON body")


def resolve_chat_model(
    state: Any,
    requested: Any,
    *,
    protocol: str = "openai",
) -> ChatModelResult:
    try:
        return resolve_handler_model(state, str(requested))
    except ModelResolveError as exc:
        return model_resolve_error_response(exc, protocol=protocol)


def log_chat_request(
    *,
    protocol: str,
    messages: list,
    model: str,
    stream: bool,
    tools: list,
    protocol_options: Optional[dict],
) -> None:
    req_level = protocol_thinking_level(protocol_options)
    route = resolve_thinking_route(model, req_level)
    logger.info(
        "%s: %d messages, model=%s, stream=%s, tools=%d, thinking_level=%s, "
        "entml=%s qwen_native=%s",
        protocol.capitalize(),
        len(messages),
        model,
        stream,
        len(tools),
        req_level,
        route.use_entml,
        route.qwen_native_enabled,
    )


def new_request_id() -> str:
    return _gen_request_id()


def resolve_retry_client(state: Any, model: str, messages: list, tools: list) -> Any:
    """按当前请求解析用于 session_retry 的上游 client。"""
    _, client = resolve_upstream(state, model, messages, tools)
    return client


async def iter_retried_chat_events(
    req_id: str,
    state: Any,
    make_stream: Callable[[], AsyncGenerator[Dict[str, Any], None]],
    *,
    model: str,
    messages: list,
    tools: list,
    disconnected: list,
    on_event: StreamEventHandler,
) -> None:
    """带换号重试的上游事件循环；on_event 返回 False 时中止。"""
    retry_client = resolve_retry_client(state, model, messages, tools)
    async with aclosing(
        stream_with_session_retry(req_id, state, make_stream, client=retry_client),
    ) as event_stream:
        async for event in event_stream:
            if disconnected[0]:
                break
            if not await on_event(event):
                break


def prepare_injected_messages(
    state: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    model: str,
    protocol_options: Optional[Dict[str, Any]],
    prompt_api: str,
) -> PrepareResult:
    """返回 (injected_messages, full_content, thinking_route)。"""
    route = resolve_thinking_route(
        model,
        protocol_thinking_level(protocol_options),
    )
    inject_options = _inject_protocol_options(protocol_options, route.use_entml)
    user_system_prompt, messages = extract_system_for_inject(messages)
    injected = inject_fncall_for_request(
        messages,
        convert_tools_to_openai(tools),
        state.protocol,
        req_id=req_id,
        api=prompt_api,
        model=model,
        lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=inject_options,
    )
    full_content = injected[0].get("content") or ""
    return injected, full_content, route


def apply_prompt_budget(
    state: Any,
    injected: List[Dict[str, Any]],
    full_content: str,
    *,
    use_file_split: bool = False,
) -> Tuple[List[Dict[str, Any]], str, Optional[str], Optional[bytes]]:
    """截断 prompt；``use_file_split=True`` 走 splitter.split（Qwen）。"""
    splitter = getattr(state, "splitter", None)
    if use_file_split and splitter is not None and hasattr(splitter, "split"):
        send_text, filename, file_bytes = splitter.split(full_content)
        messages = list(injected)
        messages[0] = {**messages[0], "content": send_text}
        return messages, send_text, filename, file_bytes
    send_text = full_content
    max_chars = int(getattr(splitter, "max_chars", 0) or 0) if splitter else 0
    send_full = bool(getattr(splitter, "send_full_prompt", True)) if splitter else True
    if not send_full and max_chars > 0 and len(send_text) > max_chars:
        send_text = send_text[-max_chars:]
        return [{**injected[0], "content": send_text}], send_text, None, None
    return injected, send_text, None, None
