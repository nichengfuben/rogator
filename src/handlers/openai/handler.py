from __future__ import annotations

from aiohttp import web

from handlers import get_state
from handlers.openai.chat import _process_openai_non_stream
from handlers.openai.protocol import _build_protocol_options
from handlers.openai.stream_run import _handle_stream
from handlers.shared.api_errors import apply_tool_choice, handler_error_response
from server.config import CONFIG
from server.formats import (
    _error_response,
    _json_response,
    openai_stream_include_usage,
)
from state import QueueFullError


async def openai_chat_handler(request: web.Request) -> web.StreamResponse:
    from handlers.chat_request import (
        log_chat_request,
        new_request_id,
        read_chat_json,
        resolve_chat_model,
    )

    state = get_state()
    if state.is_shutting_down:
        return _error_response(503, "Shutting down", "server_error")
    if state.scheduler.pending >= CONFIG.max_queue_size:
        return _error_response(503, "Busy", "server_error")
    body = await read_chat_json(request, protocol="openai")
    if isinstance(body, web.Response):
        return body
    messages = body.get("messages", [])
    model = resolve_chat_model(state, body.get("model", state.model))
    if isinstance(model, web.Response):
        return model
    stream = body.get("stream", False)
    tools = apply_tool_choice(body.get("tools", []), body.get("tool_choice"))
    if not messages:
        return _error_response(400, "messages is required")
    protocol_options = _build_protocol_options(body)
    log_chat_request(
        protocol="openai",
        messages=messages,
        model=model,
        stream=stream,
        tools=tools,
        protocol_options=protocol_options,
    )
    req_id = new_request_id()
    if not stream:
        return await _handle_non_stream(
            state, messages, model, req_id, tools, protocol_options
        )
    return await _handle_stream(
        request,
        state,
        messages,
        model,
        req_id,
        tools,
        protocol_options,
        include_usage=openai_stream_include_usage(body),
    )


async def _handle_non_stream(
    state, messages, model, req_id, tools, protocol_options=None
):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(
                state, messages, model, req_id, tools, protocol_options
            )
        )
        return _json_response(result)
    except Exception as e:
        return handler_error_response(e, label="OpenAI non-stream")
