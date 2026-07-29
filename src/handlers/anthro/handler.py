from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from aiohttp import web

from echotools.logger import get_logger

from server.formats import (
    ClientDisconnectedError,
    UpstreamTimeoutError,
    _error_response,
    _gen_msg_id,
    _gen_request_id,
    _json_response,
    client_disconnected_response,
    convert_to_anthropic,
    read_request_json,
)
from server.model.model_thinking import always_qwen_thinking, resolve_qwen_thinking

from handlers import get_state, prepend_anthropic_system
from handlers.api_errors import handler_error_response
from handlers.openai import (
    _process_openai_non_stream,
    protocol_thinking_level,
    thinking_level_is_active,
)
from handlers.anthro.events import (
    _close_block,
    _send_anthropic_finish,
    _send_tool_use_blocks,
    _write_stream_error,
)
from handlers.anthro.normalize import (
    _build_anthropic_protocol_options,
    _normalize_anthropic_messages,
    _normalize_anthropic_tools,
)
from handlers.anthro.stream_core import _stream_anthropic

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
            )
        )
        return _json_response(convert_to_anthropic(result))
    except Exception as e:
        return handler_error_response(e, label="Anthropic non-stream")


async def _emit_anthropic_handler_stream_error(
    resp, req_id: str, exc: Exception, disconnected: list, *, error_type: Optional[str] = None,
) -> None:
    err_body: Dict[str, Any] = {"message": str(exc)}
    if error_type:
        err_body["type"] = error_type
    await _write_stream_error(resp, {"type": "error", "error": err_body}, disconnected)


async def _handle_stream(request, state, messages, model, req_id, tools, protocol_options=None):
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
    except UpstreamTimeoutError as e:
        logger.warning("Anthropic stream upstream timeout (uncaught path) %s: %s", req_id, e)
        await _emit_anthropic_handler_stream_error(resp, req_id, e, disconnected, error_type="timeout")
        return resp
    except Exception as e:
        logger.error("Anthropic stream error (uncaught path) %s: %s", req_id, e, exc_info=True)
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


async def anthropic_messages_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return web.Response(status=503, text="Shutting down")
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("Anthropic client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except json.JSONDecodeError:
        return _error_response(400, "Invalid JSON body")
    raw_messages = body.get("messages", [])
    system = body.get("system")
    model = body.get("model", state.model)
    stream = body.get("stream", False)
    tools = _normalize_anthropic_tools(body.get("tools", []) or [])
    messages = _normalize_anthropic_messages(raw_messages)
    messages = prepend_anthropic_system(messages, system)
    if not messages:
        return _error_response(400, "messages is required")
    try:
        protocol_options = _build_anthropic_protocol_options(body)
    except ValueError as e:
        return _error_response(400, str(e))
    req_level = protocol_thinking_level(protocol_options)
    _, _, use_entml = resolve_qwen_thinking(model, req_level)
    qwen_thinking = not use_entml and (always_qwen_thinking(model) or thinking_level_is_active(req_level))
    logger.info(
        "Anthropic: %d messages, model=%s, stream=%s, tools=%d, thinking_level=%s, qwen_thinking=%s",
        len(messages), model, stream, len(tools), req_level, qwen_thinking,
    )
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(
            state, messages, model, req_id, tools, protocol_options,
        )
    return await _handle_stream(
        request, state, messages, model, req_id, tools, protocol_options,
    )
