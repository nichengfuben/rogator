from __future__ import annotations

"""Anthropic-compatible message handlers."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.logger import get_logger

from server.formats import (
    TokenExpiredError,
    _error_response,
    _gen_msg_id,
    _gen_request_id,
    _json_response,
    convert_to_anthropic,
)
from state import AppState, QueueFullError
from handlers import get_state
from handlers.openai import (
    _process_openai_non_stream,
    _parse_tool_calls,
    _chat_once,
)

logger = get_logger("rogator")


# ============================================================
# 流式辅助函数
# ============================================================

async def _safe_write(resp, data: bytes, disconnected: list) -> bool:
    if disconnected[0]:
        return False
    try:
        await resp.write(data)
        return True
    except (ConnectionError, OSError, asyncio.CancelledError):
        disconnected[0] = True
        return False


async def _close_block(resp, idx: int, disconnected: list) -> int:
    if idx >= 0:
        await _safe_write(
            resp,
            f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n".encode(),
            disconnected,
        )
    return idx + 1


async def _send_anthropic_finish(resp, tool_calls, disconnected):
    delta_msg = {
        "type": "message_delta",
        "delta": {
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "stop_sequence": None,
        },
        "usage": {"output_tokens": 0},
    }
    await _safe_write(resp, f"event: message_delta\ndata: {json.dumps(delta_msg)}\n\n".encode(), disconnected)
    await _safe_write(resp, f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode(), disconnected)


async def _send_text_block(resp, clean_text: str, block_idx: int, disconnected: list) -> int:
    block_idx += 1
    block_start = {
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {"type": "text", "text": ""},
    }
    await _safe_write(resp, f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode(), disconnected)
    chunk_size = 20
    for i in range(0, len(clean_text), chunk_size):
        block_delta = {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": clean_text[i:i+chunk_size]},
        }
        if not await _safe_write(resp, f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n".encode(), disconnected):
            break
    return await _close_block(resp, block_idx, disconnected)


async def _send_tool_use_blocks(resp, tool_calls, block_idx: int, disconnected: list):
    for tc in tool_calls:
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
            if not isinstance(args_dict, dict):
                args_dict = {"value": args_dict}
        except json.JSONDecodeError:
            args_dict = {}
        block_idx += 1
        block_start = {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "tool_use", "id": tc.get("id"),
                              "name": tc.get("function", {}).get("name"), "input": {}},
        }
        if not await _safe_write(resp, f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode(), disconnected):
            break
        params_json = json.dumps(args_dict, ensure_ascii=False)
        chunk_size = 20
        for i in range(0, len(params_json), chunk_size):
            partial = params_json[i:i+chunk_size]
            delta_event = {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            }
            if not await _safe_write(resp, f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n".encode(), disconnected):
                break
        block_idx = await _close_block(resp, block_idx, disconnected)
    return block_idx


async def _send_thinking_delta(resp, content, block_idx, block_type, disconnected):
    if block_type == "text":
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    if block_type != "thinking":
        if block_type is not None:
            block_idx = await _close_block(resp, block_idx, disconnected)
        else:
            block_idx += 1
        block_start = {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "thinking", "thinking": ""},
        }
        if not await _safe_write(resp, f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode(), disconnected):
            return block_idx, block_type, False
        block_type = "thinking"
    if content:
        block_delta = {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "thinking_delta", "thinking": content},
        }
        if not await _safe_write(resp, f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n".encode(), disconnected):
            return block_idx, block_type, False
    return block_idx, block_type, True


async def _stream_anthropic(resp, state, messages, model, tools, req_id, disconnected):
    block_idx = -1
    block_type: Optional[str] = None
    full_answer = ""
    try:
        async for event in _chat_once(state, messages, model, tools, req_id):
            if disconnected[0]:
                break
            etype = event.get("type")
            content = event.get("content", "")
            if etype == "thinking":
                block_idx, block_type, ok = await _send_thinking_delta(
                    resp, content, block_idx, block_type, disconnected
                )
                if not ok:
                    break
            elif etype == "answer":
                full_answer += content
    except asyncio.CancelledError:
        logger.info("Stream cancelled %s", req_id)
        await _safe_write(resp, b"data: [DONE]\n\n", disconnected)
        return block_idx, block_type, full_answer, True
    except TokenExpiredError as e:
        logger.warning("Anthropic stream token expired: %s", e)
        await _safe_write(resp, b"data: [DONE]\n\n", disconnected)
        return block_idx, block_type, full_answer, True
    except Exception as e:
        logger.error("Anthropic stream error: %s", e, exc_info=True)
        error_msg = json.dumps({"type": "error", "error": {"message": str(e)}})
        await _safe_write(resp, f"event: error\ndata: {error_msg}\n\n".encode(), disconnected)
        return block_idx, block_type, full_answer, True
    return block_idx, block_type, full_answer, False


async def _send_post_stream(resp, state, full_answer, block_type, block_idx, tools, disconnected):
    if block_type == "thinking":
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    clean_text, tool_calls = _parse_tool_calls(state, full_answer, tools)
    if clean_text and clean_text.strip():
        if block_type is not None:
            block_idx = await _close_block(resp, block_idx, disconnected)
        block_idx = await _send_text_block(resp, clean_text, block_idx, disconnected)
        block_type = None
    block_idx = await _send_tool_use_blocks(resp, tool_calls, block_idx, disconnected)
    await _send_anthropic_finish(resp, tool_calls, disconnected)


# ============================================================
# Anthropic message handlers
# ============================================================

async def anthropic_messages_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return web.Response(status=503, text="Shutting down")
    body = await request.json() if request.can_read_body else {}
    messages = body.get("messages", [])
    model = body.get("model", state.model)
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    if not messages:
        return _error_response(400, "messages is required")
    logger.info("Anthropic: %d messages, model=%s, stream=%s, tools=%d",
                len(messages), model, stream, len(tools))
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(state, messages, model, req_id, tools)
    return await _handle_stream(request, state, messages, model, req_id, tools)


async def _handle_non_stream(state, messages, model, req_id, tools):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(state, messages, model, req_id, tools))
        return _json_response(convert_to_anthropic(result))
    except QueueFullError as e:
        return web.Response(status=503, text=str(e))
    except asyncio.CancelledError:
        return web.Response(status=503, text="Shutting down")
    except TokenExpiredError as e:
        logger.warning("Anthropic non-stream token expired: %s", e)
        return _error_response(429, str(e), "rate_limited")
    except Exception as e:
        logger.error("Anthropic non-stream error: %s", e, exc_info=True)
        return _error_response(500, str(e))


async def _handle_stream(request, state, messages, model, req_id, tools):
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                 "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    await resp.prepare(request)
    disconnected = [False]
    start_msg = {"type": "message_start", "message": {
        "id": _gen_msg_id(), "type": "message", "role": "assistant",
        "content": [], "model": model, "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
    }}
    await _safe_write(resp, f"event: message_start\ndata: {json.dumps(start_msg)}\n\n".encode(), disconnected)
    block_idx, block_type, full_answer, early_return = await _stream_anthropic(
        resp, state, messages, model, tools, req_id, disconnected
    )
    if disconnected[0] or early_return:
        logger.info("Anthropic client disconnected or early return %s", req_id)
        return resp
    await _send_post_stream(resp, state, full_answer, block_type, block_idx, tools, disconnected)
    return resp
