from __future__ import annotations

import json
from typing import Any, Optional

from aiohttp import web
from echotools.logger import get_logger

from handlers import get_state
from handlers.api_errors import handler_error_response
from handlers.openai.chat import _process_openai_non_stream
from handlers.openai.protocol import _build_protocol_options
from handlers.openai.stream.handler import handle_openai_stream
from handlers.openai.thinking import protocol_thinking_level, thinking_level_is_active
from server.formats import (
    MAX_QUEUE_SIZE,
    ClientDisconnectedError,
    _error_response,
    _gen_request_id,
    _json_response,
    client_disconnected_response,
    openai_stream_include_usage,
    read_request_json,
)
from server.model.model_registry import ModelRegistryEntry
from server.model.model_thinking import always_qwen_thinking, resolve_qwen_thinking

logger = get_logger("rogator")


async def _handle_non_stream(state, messages, model, req_id, tools, protocol_options=None, *, registry_entry=None):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(
                state, messages, model, req_id, tools, protocol_options, registry_entry=registry_entry,
            ))
        return _json_response(result)
    except Exception as e:
        return handler_error_response(e, label="OpenAI non-stream")


def _resolve_openai_model(state, requested_model: str) -> tuple[ModelRegistryEntry, str]:
    from handlers.model_resolve import resolve_handler_model_entry
    from server.model.model_registry import ModelResolveError

    registry_entry = resolve_handler_model_entry(state, str(requested_model))
    return registry_entry, registry_entry.internal_id


async def openai_chat_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return web.Response(status=503, text="Shutting down")
    if state.scheduler.pending >= MAX_QUEUE_SIZE:
        return web.Response(status=503, text="Busy")
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("OpenAI client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except json.JSONDecodeError:
        return _error_response(400, "Invalid JSON body")
    messages = body.get("messages", [])
    requested_model = body.get("model", state.model)
    try:
        registry_entry, model = _resolve_openai_model(state, requested_model)
    except Exception as exc:
        from handlers.model_resolve import model_resolve_error_response
        from server.model.model_registry import ModelResolveError

        if isinstance(exc, ModelResolveError):
            return model_resolve_error_response(exc)
        raise
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    if not messages:
        return _error_response(400, "messages is required")
    protocol_options = _build_protocol_options(body)
    req_level = protocol_thinking_level(protocol_options)
    _, _, use_entml = resolve_qwen_thinking(model, req_level)
    qwen_thinking = not use_entml and (always_qwen_thinking(model) or thinking_level_is_active(req_level))
    logger.info(
        "OpenAI: %d messages, model=%s, stream=%s, tools=%d, thinking_level=%s, qwen_thinking=%s",
        len(messages), model, stream, len(tools), req_level, qwen_thinking,
    )
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(
            state, messages, model, req_id, tools, protocol_options, registry_entry=registry_entry,
        )
    return await handle_openai_stream(
        request, state, messages, model, req_id, tools, protocol_options,
        include_usage=openai_stream_include_usage(body),
        registry_entry=registry_entry,
    )
