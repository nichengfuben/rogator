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
    _fix_tool_call_id,
    _gen_msg_id,
    _gen_request_id,
    _json_response,
    convert_to_anthropic,
)
from state import AppState, QueueFullError
from echotools.fncall import FncallStreamParser

from handlers import get_state
from handlers.openai import (
    _build_protocol_options,
    _inject_protocol_options,
    _process_openai_non_stream,
    _parse_tool_calls,
    _chat_once,
    convert_tools_to_openai,
    protocol_thinking_level,
    thinking_level_is_active,
)
from server.model_thinking import always_qwen_thinking, resolve_qwen_thinking
from server.session_retry import stream_with_session_retry

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
    """关闭 content block。返回已关闭的 index（不自增，避免与下一块 start 的 +=1 双跳）。"""
    if idx >= 0:
        await _safe_write(
            resp,
            f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n".encode(),
            disconnected,
        )
    return idx


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


async def _emit_ready_tool_calls(
    resp, parser, block_idx, block_type, disconnected, pending_tc_count: int,
) -> Tuple[int, Optional[str], int, bool]:
    """增量发送 parser 中已完整闭合的 tool_use 块（对齐 mock.py input_json_delta）。"""
    ready = parser.get_ready_tool_calls()
    if not ready:
        return block_idx, block_type, pending_tc_count, True
    # tool_use 前关闭 thinking/text 块
    if block_type is not None:
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    fixed = [_fix_tool_call_id(tc) for tc in ready]
    block_idx = await _send_tool_use_blocks(resp, fixed, block_idx, disconnected)
    if disconnected[0]:
        return block_idx, block_type, pending_tc_count, False
    return block_idx, block_type, pending_tc_count + len(fixed), True


async def _stream_anthropic(
    resp, state, messages, model, tools, req_id, disconnected, protocol_options=None,
):
    block_idx = -1
    block_type: Optional[str] = None
    full_answer = ""
    parser = FncallStreamParser(protocol=state.protocol, tools=tools)
    last_safe_len = 0
    last_thinking_len = 0
    pending_tc_count = 0  # 已流式发送的 tool_use 数量
    try:
        async def _make_chat_stream():
            async for event in _chat_once(
                state, messages, model, tools, req_id, protocol_options=protocol_options,
            ):
                yield event

        async for event in stream_with_session_retry(req_id, state, _make_chat_stream):
            if disconnected[0]:
                break
            etype = event.get("type")
            content = event.get("content", "")
            if etype == "thinking":
                if content:
                    block_idx, block_type, ok = await _send_thinking_delta(
                        resp, content, block_idx, block_type, disconnected
                    )
                    if not ok:
                        break
                continue

            if etype != "answer":
                continue

            full_answer += content
            parser.feed(content)

            pt = parser.partial_thinking
            if len(pt) > last_thinking_len:
                new_thinking = pt[last_thinking_len:]
                last_thinking_len = len(pt)
                if new_thinking:
                    block_idx, block_type, ok = await _send_thinking_delta(
                        resp, new_thinking, block_idx, block_type, disconnected
                    )
                    if not ok:
                        break

            safe_text = parser.partial_text
            if len(safe_text) > last_safe_len:
                new_text = safe_text[last_safe_len:]
                last_safe_len = len(safe_text)
                if new_text:
                    if block_type != "text":
                        if block_type == "thinking":
                            block_idx = await _close_block(resp, block_idx, disconnected)
                            block_type = None
                        block_idx += 1
                        block_start = {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "text", "text": ""},
                        }
                        if not await _safe_write(
                            resp,
                            f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode(),
                            disconnected,
                        ):
                            break
                        block_type = "text"
                    block_delta = {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": new_text},
                    }
                    if not await _safe_write(
                        resp,
                        f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n".encode(),
                        disconnected,
                    ):
                        break

            # 增量发送已完整闭合的 invoke → tool_use（对齐 mock 流形态）
            block_idx, block_type, pending_tc_count, ok = await _emit_ready_tool_calls(
                resp, parser, block_idx, block_type, disconnected, pending_tc_count,
            )
            if not ok:
                break
    except asyncio.CancelledError:
        logger.info("Stream cancelled %s", req_id)
        await _safe_write(resp, b"data: [DONE]\n\n", disconnected)
        return block_idx, block_type, full_answer, True, pending_tc_count
    except TokenExpiredError as e:
        logger.warning("Anthropic stream token expired: %s", e)
        error_msg = json.dumps({"type": "error", "error": {"message": str(e), "type": "rate_limited"}})
        await _safe_write(resp, f"event: error\ndata: {error_msg}\n\n".encode(), disconnected)
        return block_idx, block_type, full_answer, True, pending_tc_count
    except Exception as e:
        logger.error("Anthropic stream error: %s", e, exc_info=True)
        error_msg = json.dumps({"type": "error", "error": {"message": str(e)}})
        await _safe_write(resp, f"event: error\ndata: {error_msg}\n\n".encode(), disconnected)
        return block_idx, block_type, full_answer, True, pending_tc_count

    # 刷 parser 尾部：holdback / thinking / 未闭合 invoke
    try:
        parser.finalize()
    except Exception as e:
        logger.warning("anthropic stream parser.finalize failed: %s", e)

    if not disconnected[0]:
        pt = parser.partial_thinking
        if len(pt) > last_thinking_len:
            new_thinking = pt[last_thinking_len:]
            last_thinking_len = len(pt)
            if new_thinking:
                block_idx, block_type, ok = await _send_thinking_delta(
                    resp, new_thinking, block_idx, block_type, disconnected
                )
                if not ok:
                    return block_idx, block_type, full_answer, True, pending_tc_count

    if not disconnected[0]:
        safe_text = parser.partial_text
        if len(safe_text) > last_safe_len:
            new_text = safe_text[last_safe_len:]
            last_safe_len = len(safe_text)
            if new_text:
                if block_type != "text":
                    if block_type == "thinking":
                        block_idx = await _close_block(resp, block_idx, disconnected)
                        block_type = None
                    block_idx += 1
                    block_start = {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                    if await _safe_write(
                        resp,
                        f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode(),
                        disconnected,
                    ):
                        block_type = "text"
                if block_type == "text":
                    block_delta = {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": new_text},
                    }
                    await _safe_write(
                        resp,
                        f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n".encode(),
                        disconnected,
                    )

    if not disconnected[0]:
        block_idx, block_type, pending_tc_count, ok = await _emit_ready_tool_calls(
            resp, parser, block_idx, block_type, disconnected, pending_tc_count,
        )
        if not ok:
            return block_idx, block_type, full_answer, True, pending_tc_count

    return block_idx, block_type, full_answer, False, pending_tc_count


async def _send_post_stream(
    resp, state, full_answer, block_type, block_idx, tools, disconnected,
    already_sent_tc_count: int = 0,
):
    # 关闭正在开着的块（文本/thinking 已实时推出，可能仍未关）
    if block_type is not None:
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    _, all_tool_calls = _parse_tool_calls(state, full_answer, tools)
    remaining = all_tool_calls[already_sent_tc_count:]
    block_idx = await _send_tool_use_blocks(resp, remaining, block_idx, disconnected)
    await _send_anthropic_finish(resp, all_tool_calls, disconnected)


# ============================================================
# Anthropic message handlers
# ============================================================

def _normalize_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 Anthropic messages（含 content 数组 / tool_use / tool_result）转成 OpenAI 风格。"""
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role") or "user"
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(dict(msg))
            continue

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text" or "text" in block and btype is None:
                text_parts.append(str(block.get("text") or ""))
            elif btype in ("thinking", "redacted_thinking"):
                t = str(block.get("thinking") or block.get("data") or "")
                if t:
                    thinking_parts.append(t)
            elif btype == "reasoning":
                t = str(block.get("text") or block.get("reasoning") or "")
                if t:
                    thinking_parts.append(t)
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif btype == "tool_result":
                out.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or block.get("tool_call_id") or "",
                    "content": block.get("content") if isinstance(block.get("content"), str)
                    else json.dumps(block.get("content"), ensure_ascii=False)
                    if block.get("content") is not None else "",
                    **({"is_error": True} if block.get("is_error") else {}),
                })
            else:
                # 忽略 image 等非文本块，文本化兜底
                if "text" in block:
                    text_parts.append(str(block.get("text") or ""))

        if role == "assistant":
            joined = "\n".join(p for p in text_parts if p) or None
            msg: Dict[str, Any] = {"role": "assistant", "content": joined}
            if thinking_parts:
                msg["reasoning"] = "\n".join(thinking_parts)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "tool":
            # 已在上面 tool_result 分支写出
            continue
        else:
            joined = "\n".join(p for p in text_parts if p)
            only_tool_results = all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ) if content else False
            if only_tool_results:
                continue
            out.append({"role": role, "content": joined})
    return out


def _normalize_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic tools（name/input_schema）→ OpenAI function tools。"""
    return convert_tools_to_openai(tools or [])


async def anthropic_messages_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return web.Response(status=503, text="Shutting down")
    body = await request.json() if request.can_read_body else {}
    raw_messages = body.get("messages", [])
    system = body.get("system")
    model = body.get("model", state.model)
    stream = body.get("stream", False)
    tools = _normalize_anthropic_tools(body.get("tools", []) or [])
    messages = _normalize_anthropic_messages(raw_messages)
    if system:
        sys_text = system if isinstance(system, str) else json.dumps(system, ensure_ascii=False)
        messages = [{"role": "system", "content": sys_text}, *messages]
    if not messages:
        return _error_response(400, "messages is required")
    protocol_options = _build_protocol_options(body)
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


async def _handle_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(
                state, messages, model, req_id, tools, protocol_options,
            )
        )
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


async def _handle_stream(request, state, messages, model, req_id, tools, protocol_options=None):
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
    block_idx, block_type, full_answer, early_return, pending_tc_count = await _stream_anthropic(
        resp, state, messages, model, tools, req_id, disconnected, protocol_options,
    )
    if disconnected[0] or early_return:
        logger.info("Anthropic client disconnected or early return %s", req_id)
        return resp
    await _send_post_stream(
        resp, state, full_answer, block_type, block_idx, tools, disconnected,
        already_sent_tc_count=pending_tc_count,
    )
    return resp
