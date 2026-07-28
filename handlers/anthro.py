from __future__ import annotations

"""Anthropic-compatible message handlers."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.logger import get_logger
from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    parse_max_thinking_length,
)

from server.formats import (
    TokenExpiredError,
    _error_response,
    _fix_tool_call_id,
    _gen_msg_id,
    _gen_request_id,
    _gen_tool_id,
    _json_response,
    convert_to_anthropic,
)
from state import AppState, QueueFullError
from echotools.fncall import FncallStreamParser

from handlers import get_state, prepend_anthropic_system
from handlers.openai import (
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

_ANTHROPIC_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _parse_anthropic_effort(body: Dict[str, Any]) -> str:
    """读取 output_config.effort；省略时官方默认为 high。"""
    output_config = body.get("output_config")
    if not isinstance(output_config, dict) or "effort" not in output_config:
        return "high"
    level = normalize_thinking_level(output_config["effort"])
    if level not in _ANTHROPIC_EFFORT_LEVELS:
        raise ValueError(f"invalid output_config.effort: {output_config['effort']!r}")
    return level


def _build_anthropic_protocol_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """按 Anthropic Messages API 解析 thinking 与 output_config.effort。

    - effort：仅来自 ``output_config.effort``（默认 high）
    - thinking.type：``disabled`` | ``enabled`` | ``adaptive``
    - ``enabled`` 时 ``budget_tokens`` → ``max_thinking_length``
    """
    effort = _parse_anthropic_effort(body)
    opts: Dict[str, Any] = {"include_thinking_in_history": True}

    thinking = body.get("thinking")
    if thinking is None:
        opts["thinking_level"] = effort
        return opts

    if not isinstance(thinking, dict):
        raise ValueError("thinking must be an object")

    thinking_type = thinking.get("type")
    if thinking_type is None:
        raise ValueError("thinking.type is required when thinking is set")

    mode = str(thinking_type).strip().lower()
    if mode == "disabled":
        opts["thinking_level"] = "none"
        return opts

    if mode == "adaptive":
        opts["thinking_level"] = effort
        return opts

    if mode == "enabled":
        opts["thinking_level"] = effort
        if "budget_tokens" in thinking:
            max_len = parse_max_thinking_length(thinking["budget_tokens"])
            if max_len is None:
                raise ValueError("thinking.budget_tokens must be a positive integer")
            opts["max_thinking_length"] = max_len
        return opts

    raise ValueError(f"unsupported thinking.type: {thinking_type!r}")


# ============================================================
# Anthropic SSE 流式事件（对齐 mock.py AnthropicBuilder）
# ============================================================

_STREAM_CHUNK_SIZE = 20


def _tool_call_input_dict(tc: Dict[str, Any]) -> Dict[str, Any]:
    args_str = tc.get("function", {}).get("arguments", "{}")
    try:
        args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
        if not isinstance(args_dict, dict):
            return {"value": args_dict}
        return args_dict
    except json.JSONDecodeError:
        return {}


def _message_start_event(model: str, msg_id: str) -> Dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def _content_block_stop_event(index: int) -> Dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def _message_delta_event(stop_reason: str) -> Dict[str, Any]:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }


def _message_stop_event() -> Dict[str, Any]:
    return {"type": "message_stop"}


def _tool_use_block_events(
    block_idx: int, tool_id: str, name: str, input_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """单个 tool_use 块的 SSE 事件序列（与 mock.py _build_tool_events 一致）。"""
    events: List[Dict[str, Any]] = [{
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    }]
    params_json = json.dumps(input_dict, ensure_ascii=False)
    for i in range(0, len(params_json), _STREAM_CHUNK_SIZE):
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": params_json[i : i + _STREAM_CHUNK_SIZE],
            },
        })
    events.append(_content_block_stop_event(block_idx))
    return events


def _anthropic_event_bytes(event: Dict[str, Any]) -> bytes:
    return (
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


async def _emit_anthropic_event(resp, event: Dict[str, Any], disconnected: list) -> bool:
    return await _safe_write(resp, _anthropic_event_bytes(event), disconnected)


async def _emit_anthropic_events(resp, events: List[Dict[str, Any]], disconnected: list) -> bool:
    for event in events:
        if not await _emit_anthropic_event(resp, event, disconnected):
            return False
    return True


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
        await _emit_anthropic_event(resp, _content_block_stop_event(idx), disconnected)
    return idx


async def _send_anthropic_finish(
    resp, tool_calls, disconnected, *, streamed_tool_count: int = 0,
):
    """message_delta + message_stop（对齐 mock.py _build_message_delta）。"""
    stop_reason = "tool_use" if (tool_calls or streamed_tool_count > 0) else "end_turn"
    await _emit_anthropic_event(resp, _message_delta_event(stop_reason), disconnected)
    await _emit_anthropic_event(resp, _message_stop_event(), disconnected)


async def _send_text_block(resp, clean_text: str, block_idx: int, disconnected: list) -> int:
    block_idx += 1
    events: List[Dict[str, Any]] = [{
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {"type": "text", "text": ""},
    }]
    for i in range(0, len(clean_text), _STREAM_CHUNK_SIZE):
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": clean_text[i : i + _STREAM_CHUNK_SIZE]},
        })
    events.append(_content_block_stop_event(block_idx))
    await _emit_anthropic_events(resp, events, disconnected)
    return block_idx


async def _send_tool_use_blocks(resp, tool_calls, block_idx: int, disconnected: list) -> int:
    for tc in tool_calls:
        fixed = _fix_tool_call_id(tc)
        block_idx += 1
        events = _tool_use_block_events(
            block_idx,
            fixed["id"],
            fixed.get("function", {}).get("name", ""),
            _tool_call_input_dict(fixed),
        )
        if not await _emit_anthropic_events(resp, events, disconnected):
            break
    return block_idx


async def _emit_tool_json_pieces(
    resp,
    stream_tool: Dict[str, Any],
    partial_json: str,
    disconnected: list,
) -> bool:
    stream_tool["json_buf"] = stream_tool.get("json_buf", "") + partial_json
    for i in range(0, len(partial_json), _STREAM_CHUNK_SIZE):
        piece = partial_json[i : i + _STREAM_CHUNK_SIZE]
        if not await _emit_anthropic_event(resp, {
            "type": "content_block_delta",
            "index": stream_tool["block_idx"],
            "delta": {"type": "input_json_delta", "partial_json": piece},
        }, disconnected):
            return False
    return True


async def _flush_open_stream_tool(
    resp,
    parser,
    stream_tool: Optional[Dict[str, Any]],
    disconnected: list,
    *,
    expected_arguments: Optional[str] = None,
) -> bool:
    """补全并关闭流式 tool_use；Anthropic 要求先 stop 再开下一块。"""
    if stream_tool is None:
        return True
    while True:
        delta_info = parser.consume_stream_delta()
        if not delta_info:
            break
        _name, partial_json = delta_info
        if partial_json:
            if not await _emit_tool_json_pieces(resp, stream_tool, partial_json, disconnected):
                return False
    if not parser.streaming_invoke_closed:
        final_delta = parser.complete_stream_delta_if_needed()
        if final_delta:
            _name, piece = final_delta
            if piece and not await _emit_tool_json_pieces(resp, stream_tool, piece, disconnected):
                return False
    if expected_arguments:
        buf = stream_tool.get("json_buf", "")
        if expected_arguments.startswith(buf) and len(expected_arguments) > len(buf):
            tail = expected_arguments[len(buf) :]
            if not await _emit_tool_json_pieces(resp, stream_tool, tail, disconnected):
                return False
    await _emit_anthropic_event(
        resp, _content_block_stop_event(stream_tool["block_idx"]), disconnected,
    )
    return not disconnected[0]


async def _send_thinking_delta(
    resp, content, block_idx, block_type, disconnected,
    *, parser=None, stream_tool=None,
):
    if stream_tool is not None and parser is not None:
        if not await _flush_open_stream_tool(resp, parser, stream_tool, disconnected):
            return block_idx, block_type, stream_tool, False
        stream_tool = None
        block_type = None
    elif block_type == "text":
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    if block_type != "thinking":
        if block_type is not None:
            block_idx = await _close_block(resp, block_idx, disconnected)
        block_idx += 1
        if not await _emit_anthropic_event(resp, {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "thinking", "thinking": ""},
        }, disconnected):
            return block_idx, block_type, stream_tool, False
        block_type = "thinking"
    if content:
        if not await _emit_anthropic_event(resp, {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "thinking_delta", "thinking": content},
        }, disconnected):
            return block_idx, block_type, stream_tool, False
    return block_idx, block_type, stream_tool, True


async def _emit_ready_tool_calls(
    resp,
    parser,
    block_idx,
    block_type,
    disconnected,
    pending_tc_count: int,
    ready: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, Optional[str], int, bool]:
    """增量发送 parser 中已完整闭合的 tool_use 块（对齐 mock.py input_json_delta）。"""
    if ready is None:
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


async def _emit_streaming_tool_delta(
    resp,
    parser,
    block_idx: int,
    block_type: Optional[str],
    stream_tool: Optional[Dict[str, Any]],
    disconnected: list,
) -> Tuple[int, Optional[str], Optional[Dict[str, Any]], bool]:
    """invoke 开标签就绪后，增量发送 input_json_delta（无需等 </entml:invoke>）。"""
    while True:
        delta_info = parser.consume_stream_delta()
        if not delta_info:
            break
        name, partial_json = delta_info
        if not partial_json:
            continue
        if block_type is not None and block_type != "tool_use":
            block_idx = await _close_block(resp, block_idx, disconnected)
            block_type = None
        if stream_tool is not None and stream_tool.get("name") != name:
            if not await _flush_open_stream_tool(resp, parser, stream_tool, disconnected):
                return block_idx, block_type, stream_tool, False
            stream_tool = None
        if stream_tool is None:
            block_idx += 1
            stream_tool = {
                "block_idx": block_idx,
                "tool_id": _gen_tool_id(),
                "name": name,
                "json_buf": "",
            }
            if not await _emit_anthropic_event(resp, {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": stream_tool["tool_id"],
                    "name": name,
                    "input": {},
                },
            }, disconnected):
                return block_idx, block_type, stream_tool, False
            block_type = "tool_use"
        if not await _emit_tool_json_pieces(resp, stream_tool, partial_json, disconnected):
            return block_idx, block_type, stream_tool, False
    return block_idx, block_type, stream_tool, True


async def _close_streaming_tool_block(
    resp,
    stream_tool: Optional[Dict[str, Any]],
    block_idx: int,
    disconnected: list,
    *,
    parser=None,
    expected_arguments: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """关闭流式 tool_use 块（兼容旧调用；优先走 _flush_open_stream_tool）。"""
    if stream_tool is None:
        return block_idx, None
    if parser is not None:
        await _flush_open_stream_tool(
            resp, parser, stream_tool, disconnected,
            expected_arguments=expected_arguments,
        )
        return block_idx, None
    await _emit_anthropic_event(
        resp, _content_block_stop_event(stream_tool["block_idx"]), disconnected,
    )
    return block_idx, None


async def _stream_anthropic(
    resp, state, messages, model, tools, req_id, disconnected, protocol_options=None,
) -> Tuple[int, Optional[str], str, bool, int, List[Dict[str, Any]]]:
    block_idx = -1
    block_type: Optional[str] = None
    full_answer = ""
    all_tool_calls: List[Dict[str, Any]] = []
    parser = FncallStreamParser(protocol=state.protocol, tools=tools)
    last_safe_len = 0
    last_thinking_len = 0
    pending_tc_count = 0  # 已流式发送的 tool_use 数量
    streamed_tool_calls: List[Dict[str, Any]] = []
    stream_tool: Optional[Dict[str, Any]] = None
    stream_tool_blocks_sent = 0
    try:
        async def _make_chat_stream():
            async for event in _chat_once(
                state, messages, model, tools, req_id, protocol_options=protocol_options,
                prompt_api="anthropic",
            ):
                yield event

        async for event in stream_with_session_retry(req_id, state, _make_chat_stream):
            if disconnected[0]:
                break
            etype = event.get("type")
            content = event.get("content", "")
            if etype == "thinking":
                if content:
                    block_idx, block_type, stream_tool, ok = await _send_thinking_delta(
                        resp, content, block_idx, block_type, disconnected,
                        parser=parser, stream_tool=stream_tool,
                    )
                    if not ok:
                        break
                continue

            if etype != "answer":
                continue

            full_answer += content
            ready_calls = parser.feed(content)

            had_stream_tool = stream_tool is not None
            block_idx, block_type, stream_tool, ok = await _emit_streaming_tool_delta(
                resp, parser, block_idx, block_type, stream_tool, disconnected,
            )
            if not had_stream_tool and stream_tool is not None:
                stream_tool_blocks_sent += 1
            if not ok:
                break

            pt = parser.partial_thinking
            if len(pt) > last_thinking_len:
                new_thinking = pt[last_thinking_len:]
                last_thinking_len = len(pt)
                if new_thinking:
                    block_idx, block_type, stream_tool, ok = await _send_thinking_delta(
                        resp, new_thinking, block_idx, block_type, disconnected,
                        parser=parser, stream_tool=stream_tool,
                    )
                    if not ok:
                        break

            safe_text = parser.partial_text
            if len(safe_text) > last_safe_len:
                new_text = safe_text[last_safe_len:]
                last_safe_len = len(safe_text)
                if new_text:
                    if stream_tool is not None:
                        if not await _flush_open_stream_tool(
                            resp, parser, stream_tool, disconnected,
                        ):
                            break
                        stream_tool = None
                        block_type = None
                    if block_type != "text":
                        if block_type == "thinking":
                            block_idx = await _close_block(resp, block_idx, disconnected)
                            block_type = None
                        block_idx += 1
                        if not await _emit_anthropic_event(resp, {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "text", "text": ""},
                        }, disconnected):
                            break
                        block_type = "text"
                    if not await _emit_anthropic_event(resp, {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": new_text},
                    }, disconnected):
                        break

            if ready_calls:
                fixed = [_fix_tool_call_id(tc) for tc in ready_calls]
                streamed_tool_calls.extend(fixed)
                if stream_tool is not None:
                    expected = next(
                        (
                            tc["function"]["arguments"]
                            for tc in fixed
                            if tc["function"]["name"] == stream_tool.get("name")
                        ),
                        fixed[0]["function"]["arguments"] if len(fixed) == 1 else None,
                    )
                    if not await _flush_open_stream_tool(
                        resp, parser, stream_tool, disconnected,
                        expected_arguments=expected,
                    ):
                        break
                    stream_tool = None
                    block_type = None
                    pending_tc_count += len(fixed)
                else:
                    block_idx, block_type, pending_tc_count, ok = await _emit_ready_tool_calls(
                        resp, parser, block_idx, block_type, disconnected, pending_tc_count,
                        ready=ready_calls,
                    )
                    if not ok:
                        break
    except asyncio.CancelledError:
        logger.info("Stream cancelled %s", req_id)
        return block_idx, block_type, full_answer, True, pending_tc_count, streamed_tool_calls or all_tool_calls
    except TokenExpiredError as e:
        logger.warning("Anthropic stream token expired: %s", e)
        error_msg = json.dumps({"type": "error", "error": {"message": str(e), "type": "rate_limited"}})
        await _safe_write(resp, f"event: error\ndata: {error_msg}\n\n".encode("utf-8"), disconnected)
        merged = all_tool_calls or streamed_tool_calls
        return block_idx, block_type, full_answer, True, pending_tc_count, merged
    except Exception as e:
        logger.error("Anthropic stream error: %s", e, exc_info=True)
        error_msg = json.dumps({"type": "error", "error": {"message": str(e)}})
        await _safe_write(resp, f"event: error\ndata: {error_msg}\n\n".encode("utf-8"), disconnected)
        merged = all_tool_calls or streamed_tool_calls
        return block_idx, block_type, full_answer, True, pending_tc_count, merged

    # 刷 parser 尾部：holdback / thinking / 未闭合 invoke
    final_text = parser.partial_text
    try:
        final_text, parsed_calls = parser.finalize()
        all_tool_calls = [_fix_tool_call_id(tc) for tc in parsed_calls]
    except Exception as e:
        logger.warning("anthropic stream parser.finalize failed: %s", e)
        final_text = parser.partial_text

    if not disconnected[0]:
        pt = parser.partial_thinking
        if len(pt) > last_thinking_len:
            new_thinking = pt[last_thinking_len:]
            last_thinking_len = len(pt)
            if new_thinking:
                block_idx, block_type, stream_tool, ok = await _send_thinking_delta(
                    resp, new_thinking, block_idx, block_type, disconnected,
                    parser=parser, stream_tool=stream_tool,
                )
                if not ok:
                    merged = all_tool_calls or streamed_tool_calls
                    return block_idx, block_type, full_answer, True, pending_tc_count, merged

    if not disconnected[0]:
        safe_text = parser.partial_text if parser.has_calls else (final_text or parser.partial_text)
        if len(safe_text) > last_safe_len:
            new_text = safe_text[last_safe_len:]
            last_safe_len = len(safe_text)
            if new_text:
                if stream_tool is not None:
                    if not await _flush_open_stream_tool(
                        resp, parser, stream_tool, disconnected,
                    ):
                        merged = all_tool_calls or streamed_tool_calls
                        return block_idx, block_type, full_answer, True, pending_tc_count, merged
                    stream_tool = None
                    block_type = None
                if block_type != "text":
                    if block_type == "thinking":
                        block_idx = await _close_block(resp, block_idx, disconnected)
                        block_type = None
                    block_idx += 1
                    if await _emit_anthropic_event(resp, {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }, disconnected):
                        block_type = "text"
                if block_type == "text":
                    await _emit_anthropic_event(resp, {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": new_text},
                    }, disconnected)

    if not disconnected[0]:
        late_ready = parser.get_ready_tool_calls()
        if late_ready:
            fixed = [_fix_tool_call_id(tc) for tc in late_ready]
            streamed_tool_calls.extend(fixed)
            if stream_tool is not None:
                expected = next(
                    (
                        tc["function"]["arguments"]
                        for tc in fixed
                        if tc["function"]["name"] == stream_tool.get("name")
                    ),
                    fixed[0]["function"]["arguments"] if len(fixed) == 1 else None,
                )
                if not await _flush_open_stream_tool(
                    resp, parser, stream_tool, disconnected,
                    expected_arguments=expected,
                ):
                    merged = all_tool_calls or streamed_tool_calls
                    return block_idx, block_type, full_answer, True, pending_tc_count, merged
                stream_tool = None
                block_type = None
                pending_tc_count += len(fixed)
            else:
                block_idx, block_type, pending_tc_count, ok = await _emit_ready_tool_calls(
                    resp, parser, block_idx, block_type, disconnected, pending_tc_count,
                    ready=late_ready,
                )
                if not ok:
                    merged = all_tool_calls or streamed_tool_calls
                    return block_idx, block_type, full_answer, True, pending_tc_count, merged
        elif stream_tool is not None:
            expected = None
            if all_tool_calls:
                expected = all_tool_calls[0]["function"]["arguments"]
            if not await _flush_open_stream_tool(
                resp, parser, stream_tool, disconnected,
                expected_arguments=expected,
            ):
                merged = all_tool_calls or streamed_tool_calls
                return block_idx, block_type, full_answer, True, pending_tc_count, merged
            stream_tool = None
            block_type = None
            if parser.streaming_invoke_closed or all_tool_calls:
                pending_tc_count += 1
            else:
                logger.warning(
                    "Anthropic stream ended with incomplete invoke %s", req_id,
                )
        else:
            block_idx, block_type, pending_tc_count, ok = await _emit_ready_tool_calls(
                resp, parser, block_idx, block_type, disconnected, pending_tc_count,
            )
            if not ok:
                merged = all_tool_calls or streamed_tool_calls
                return block_idx, block_type, full_answer, True, pending_tc_count, merged

    if not all_tool_calls:
        all_tool_calls = streamed_tool_calls

    if all_tool_calls and stream_tool_blocks_sent:
        pending_tc_count = max(
            pending_tc_count,
            min(len(all_tool_calls), stream_tool_blocks_sent),
        )

    return block_idx, block_type, full_answer, False, pending_tc_count, all_tool_calls


async def _send_post_stream(
    resp, block_type, block_idx, all_tool_calls, disconnected,
    already_sent_tc_count: int = 0,
):
    if block_type is not None:
        block_idx = await _close_block(resp, block_idx, disconnected)
    remaining = all_tool_calls[already_sent_tc_count:]
    if remaining:
        await _send_tool_use_blocks(resp, remaining, block_idx, disconnected)
    await _send_anthropic_finish(
        resp, all_tool_calls, disconnected, streamed_tool_count=already_sent_tc_count,
    )


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
    msg_id = _gen_msg_id()
    await _emit_anthropic_event(resp, _message_start_event(model, msg_id), disconnected)
    block_idx, block_type, _full_answer, early_return, pending_tc_count, all_tool_calls = await _stream_anthropic(
        resp, state, messages, model, tools, req_id, disconnected, protocol_options,
    )
    if disconnected[0] or early_return:
        logger.info("Anthropic client disconnected or early return %s", req_id)
        return resp
    await _send_post_stream(
        resp, block_type, block_idx, all_tool_calls, disconnected,
        already_sent_tc_count=pending_tc_count,
    )
    return resp
