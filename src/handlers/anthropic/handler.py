from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from aiohttp import web

from echotools.base.logger import get_logger

from server.formats import (
    _gen_msg_id,
    _json_response,
    convert_to_anthropic,
)
from handlers import get_state, prepend_anthropic_system
from handlers.shared.api_errors import (
    anthropic_error_event,
    anthropic_error_response,
    apply_tool_choice,
    classify_stream_error,
    handler_error_response,
    require_anthropic_max_tokens,
)
from handlers.chat_request import (
    log_chat_request,
    new_request_id,
    read_chat_json,
    resolve_chat_model,
)
from handlers.openai import _process_openai_non_stream
from handlers.anthropic.events import (
    _close_block,
    _send_anthropic_finish,
    _send_tool_use_blocks,
    _write_stream_error,
)
from handlers.anthropic.normalize import (
    _build_anthropic_protocol_options,
    _normalize_anthropic_messages,
    _normalize_anthropic_tools,
)
from handlers.anthropic.stream_core import _stream_anthropic
from state import QueueFullError, tracked_request

logger = get_logger("rogator")


async def _send_post_stream(
    resp, block_type, block_idx, all_tool_calls, disconnected,
    already_sent_tc_count: int = 0,
    usage: Optional[Dict[str, int]] = None,
):
    if block_type is not None:
        block_idx = await _close_block(resp, block_idx, disconnected)
    remaining = all_tool_calls[already_sent_tc_count:]
    if remaining:
        await _send_tool_use_blocks(resp, remaining, block_idx, disconnected)
    await _send_anthropic_finish(
        resp, all_tool_calls, disconnected, streamed_tool_count=already_sent_tc_count,
        usage=usage,
    )


async def _handle_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(
                state, messages, model, req_id, tools, protocol_options,
                prompt_api="anthropic",
            )
        )
        return _json_response(convert_to_anthropic(result))
    except Exception as e:
        return handler_error_response(e, label="Anthropic non-stream", protocol="anthropic")


async def _emit_anthropic_handler_stream_error(
    resp, req_id: str, exc: BaseException, disconnected: list,
) -> None:
    info = classify_stream_error(exc)
    if info.kind == "timeout":
        logger.warning("Anthropic stream upstream timeout (uncaught path) %s: %s", req_id, exc)
    else:
        logger.error("Anthropic stream error (uncaught path) %s: %s", req_id, exc, exc_info=True)
    await _write_stream_error(resp, anthropic_error_event(info), disconnected)


async def _handle_stream(request, state, messages, model, req_id, tools, protocol_options=None):
    try:
        async with tracked_request(state, req_id):
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                         "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
            await resp.prepare(request)
            disconnected = [False]
            msg_id = _gen_msg_id()
            try:
                block_idx, block_type, _full_answer, early_return, pending_tc_count, all_tool_calls, usage_tracker = await _stream_anthropic(
                    resp, state, messages, model, tools, req_id, disconnected, protocol_options,
                    msg_id=msg_id,
                )
            except asyncio.CancelledError:
                logger.info("Anthropic stream cancelled during shutdown %s", req_id)
                raise
            except Exception as e:
                await _emit_anthropic_handler_stream_error(resp, req_id, e, disconnected)
                return resp
            if disconnected[0] or early_return:
                logger.info("Anthropic client disconnected or early return %s", req_id)
                return resp
            await _send_post_stream(
                resp, block_type, block_idx, all_tool_calls, disconnected,
                already_sent_tc_count=pending_tc_count,
                usage=usage_tracker.anthropic_message_delta_usage,
            )
            return resp
    except QueueFullError as exc:
        return handler_error_response(exc, label="Anthropic stream", protocol="anthropic")


async def anthropic_messages_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return anthropic_error_response(503, "Shutting down", "api_error")
    body = await read_chat_json(request, protocol="anthropic")
    if isinstance(body, web.Response):
        return body
    max_tokens = require_anthropic_max_tokens(body)
    if isinstance(max_tokens, web.Response):
        return max_tokens
    raw_messages = body.get("messages", [])
    system = body.get("system")
    model = resolve_chat_model(
        state, body.get("model", state.model), protocol="anthropic",
    )
    if isinstance(model, web.Response):
        return model
    stream = body.get("stream", False)
    tools = apply_tool_choice(
        _normalize_anthropic_tools(body.get("tools", []) or []),
        body.get("tool_choice"),
    )
    messages = _normalize_anthropic_messages(raw_messages)
    messages = prepend_anthropic_system(messages, system)
    if not messages:
        return anthropic_error_response(400, "messages is required")
    try:
        protocol_options = _build_anthropic_protocol_options(body)
    except ValueError as e:
        return anthropic_error_response(400, str(e))
    protocol_options = dict(protocol_options or {})
    protocol_options["max_tokens"] = max_tokens
    log_chat_request(
        protocol="anthropic",
        messages=messages,
        model=model,
        stream=stream,
        tools=tools,
        protocol_options=protocol_options,
    )
    req_id = new_request_id()
    if not stream:
        return await _handle_non_stream(
            state, messages, model, req_id, tools, protocol_options,
        )
    return await _handle_stream(
        request, state, messages, model, req_id, tools, protocol_options,
    )
