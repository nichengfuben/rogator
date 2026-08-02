from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from handlers.shared.api_errors import safe_write as _safe_write
from handlers.shared.fncall_inject import STREAM_CHUNK_SIZE, iter_text_chunks
from server.formats import UpstreamUsageTracker, _fix_tool_call_id


def _tool_call_input_dict(tc: Dict[str, Any]) -> Dict[str, Any]:
    args_str = tc.get("function", {}).get("arguments", "{}")
    try:
        args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
        if not isinstance(args_dict, dict):
            return {"value": args_dict}
        return args_dict
    except json.JSONDecodeError:
        return {}


def _message_start_event(
    model: str,
    msg_id: str,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
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
            "usage": usage if usage is not None else {"input_tokens": 0, "output_tokens": 0},
        },
    }


def _content_block_stop_event(index: int) -> Dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def _message_delta_event(
    stop_reason: str,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage if usage is not None else {"output_tokens": 0},
    }


def _message_stop_event() -> Dict[str, Any]:
    return {"type": "message_stop"}


def _tool_use_block_events(
    block_idx: int, tool_id: str, name: str, input_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """鍗曚釜 tool_use 鍧楃殑 SSE 浜嬩欢搴忓垪锛堜笌 mock.py _build_tool_events 涓�鑷达級銆�"""
    events: List[Dict[str, Any]] = [{
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    }]
    params_json = json.dumps(input_dict, ensure_ascii=False)
    for piece in iter_text_chunks(params_json, STREAM_CHUNK_SIZE):
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": piece,
            },
        })
    events.append(_content_block_stop_event(block_idx))
    return events


def _anthropic_event_bytes(event: Dict[str, Any]) -> bytes:
    return (
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


async def _write_stream_error(resp, error_msg: dict, disconnected: list) -> None:
    """鍚� Anthropic SSE 娴佸啓鍏� error 浜嬩欢銆�"""
    payload = json.dumps(error_msg)
    await _safe_write(resp, f"event: error\ndata: {payload}\n\n".encode("utf-8"), disconnected)


async def _emit_anthropic_event(resp, event: Dict[str, Any], disconnected: list) -> bool:
    return await _safe_write(resp, _anthropic_event_bytes(event), disconnected)


async def _emit_anthropic_events(resp, events: List[Dict[str, Any]], disconnected: list) -> bool:
    for event in events:
        if not await _emit_anthropic_event(resp, event, disconnected):
            return False
    return True


async def _close_block(resp, idx: int, disconnected: list) -> int:
    """鍏抽棴 content block銆傝繑鍥炲凡鍏抽棴鐨� index锛堜笉鑷�澧烇紝閬垮厤涓庝笅涓�鍧� start 鐨� +=1 鍙岃烦锛夈��"""
    if idx >= 0:
        await _emit_anthropic_event(resp, _content_block_stop_event(idx), disconnected)
    return idx


async def _send_anthropic_finish(
    resp, tool_calls, disconnected, *, streamed_tool_count: int = 0,
    usage: Optional[Dict[str, int]] = None,
):
    """message_delta + message_stop锛堝�归綈 mock.py _build_message_delta锛夈��"""
    stop_reason = "tool_use" if (tool_calls or streamed_tool_count > 0) else "end_turn"
    await _emit_anthropic_event(resp, _message_delta_event(stop_reason, usage=usage), disconnected)
    await _emit_anthropic_event(resp, _message_stop_event(), disconnected)


async def _send_text_block(resp, clean_text: str, block_idx: int, disconnected: list) -> int:
    block_idx += 1
    events: List[Dict[str, Any]] = [{
        "type": "content_block_start",
        "index": block_idx,
        "content_block": {"type": "text", "text": ""},
    }]
    for piece in iter_text_chunks(clean_text, STREAM_CHUNK_SIZE):
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": piece},
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


@dataclass
class AnthropicStreamState:
    block_idx: int = -1
    block_type: Optional[str] = None
    full_answer: str = ""
    last_safe_len: int = 0
    last_thinking_len: int = 0
    pending_tc_count: int = 0
    streamed_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stream_tool: Optional[Dict[str, Any]] = None
    stream_tool_blocks_sent: int = 0
    message_started: bool = False
    deferred_content: List[Dict[str, Any]] = field(default_factory=list)
    all_tool_calls: List[Dict[str, Any]] = field(default_factory=list)


def merged_tool_calls(state: AnthropicStreamState) -> List[Dict[str, Any]]:
    return state.all_tool_calls or state.streamed_tool_calls


def stream_result_tuple(
    state: AnthropicStreamState,
    usage_tracker: UpstreamUsageTracker,
    *,
    early_return: bool = False,
) -> Tuple[int, Optional[str], str, bool, int, List[Dict[str, Any]], UpstreamUsageTracker]:
    return (
        state.block_idx,
        state.block_type,
        state.full_answer,
        early_return,
        state.pending_tc_count,
        merged_tool_calls(state),
        usage_tracker,
    )


def expected_arguments_for_stream_tool(
    fixed: List[Dict[str, Any]],
    stream_tool: Optional[Dict[str, Any]],
) -> Optional[str]:
    if stream_tool is None:
        return None
    for tc in fixed:
        if tc["function"]["name"] == stream_tool.get("name"):
            return tc["function"]["arguments"]
    if len(fixed) == 1:
        return fixed[0]["function"]["arguments"]
    return None
