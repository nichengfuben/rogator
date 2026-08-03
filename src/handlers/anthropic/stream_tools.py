from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from echotools.base.logger import get_logger

from handlers.anthropic.events import (
    _close_block,
    _content_block_stop_event,
    _emit_anthropic_event,
    _send_tool_use_blocks,
)
from handlers.shared.fncall_inject import (
    STREAM_CHUNK_SIZE,
    emit_parser_stream_deltas,
    iter_text_chunks,
    take_parser_final_delta,
)
from server.formats import _fix_tool_call_id, _gen_tool_id

logger = get_logger("rogator")


def _arguments_json_equal(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return False


async def _emit_tool_json_pieces(
    resp,
    stream_tool: Dict[str, Any],
    partial_json: str,
    disconnected: list,
) -> bool:
    stream_tool["json_buf"] = stream_tool.get("json_buf", "") + partial_json
    for piece in iter_text_chunks(partial_json, STREAM_CHUNK_SIZE):
        if not await _emit_anthropic_event(
            resp,
            {
                "type": "content_block_delta",
                "index": stream_tool["block_idx"],
                "delta": {"type": "input_json_delta", "partial_json": piece},
            },
            disconnected,
        ):
            return False
    return True


async def _drain_parser_stream_deltas(
    resp,
    parser,
    stream_tool: Dict[str, Any],
    disconnected: list,
) -> bool:
    async def _on_delta(_name: str, partial_json: str) -> bool:
        return await _emit_tool_json_pieces(
            resp, stream_tool, partial_json, disconnected
        )

    return await emit_parser_stream_deltas(parser, _on_delta)


async def _emit_parser_final_delta(
    resp,
    parser,
    stream_tool: Dict[str, Any],
    disconnected: list,
) -> bool:
    final_delta = take_parser_final_delta(parser)
    if not final_delta:
        return True
    _name, piece = final_delta
    return await _emit_tool_json_pieces(resp, stream_tool, piece, disconnected)


async def _sync_json_buf_prefix(
    resp,
    stream_tool: Dict[str, Any],
    buf: str,
    expected_arguments: str,
    disconnected: list,
) -> bool:
    if expected_arguments.startswith(buf) and len(expected_arguments) > len(buf):
        tail = expected_arguments[len(buf) :]
        return await _emit_tool_json_pieces(resp, stream_tool, tail, disconnected)
    return True


async def _sync_json_buf_fallback(
    resp,
    stream_tool: Dict[str, Any],
    buf: str,
    expected_arguments: str,
    disconnected: list,
) -> bool:
    if _arguments_json_equal(buf, expected_arguments):
        return True
    if expected_arguments.startswith(buf):
        tail = expected_arguments[len(buf) :]
        if tail:
            return await _emit_tool_json_pieces(resp, stream_tool, tail, disconnected)
    return True


async def _sync_json_buf_with_expected(
    resp,
    stream_tool: Dict[str, Any],
    expected_arguments: str,
    disconnected: list,
) -> bool:
    buf = stream_tool.get("json_buf", "")
    if _arguments_json_equal(buf, expected_arguments):
        return True
    if expected_arguments.startswith(buf) and len(expected_arguments) > len(buf):
        return await _sync_json_buf_prefix(
            resp,
            stream_tool,
            buf,
            expected_arguments,
            disconnected,
        )
    return await _sync_json_buf_fallback(
        resp,
        stream_tool,
        buf,
        expected_arguments,
        disconnected,
    )


def _warn_invalid_json_buf(buf: str) -> None:
    if not buf:
        return
    try:
        json.loads(buf)
    except json.JSONDecodeError:
        logger.debug(
            "tool json_buf incomplete after force_close (%d bytes)",
            len(buf),
        )


async def _stop_stream_tool_block(
    resp,
    stream_tool: Dict[str, Any],
    disconnected: list,
) -> bool:
    await _emit_anthropic_event(
        resp,
        _content_block_stop_event(stream_tool["block_idx"]),
        disconnected,
    )
    return not disconnected[0]


async def _flush_open_stream_tool(
    resp,
    parser,
    stream_tool: Optional[Dict[str, Any]],
    disconnected: list,
    *,
    expected_arguments: Optional[str] = None,
) -> bool:
    """琛ュ叏骞跺叧闂�娴佸紡 tool_use锛汚nthropic 瑕佹眰鍏� stop 鍐嶅紑涓嬩竴鍧椼��"""
    if stream_tool is None:
        return True
    if not await _drain_parser_stream_deltas(resp, parser, stream_tool, disconnected):
        return False
    if not await _emit_parser_final_delta(resp, parser, stream_tool, disconnected):
        return False
    if expected_arguments:
        if not await _sync_json_buf_with_expected(
            resp,
            stream_tool,
            expected_arguments,
            disconnected,
        ):
            return False
    else:
        _warn_invalid_json_buf(stream_tool.get("json_buf", ""))
    return await _stop_stream_tool_block(resp, stream_tool, disconnected)


async def _send_thinking_delta(
    resp,
    content,
    block_idx,
    block_type,
    disconnected,
    *,
    parser=None,
    stream_tool=None,
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
        if not await _emit_anthropic_event(
            resp,
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            disconnected,
        ):
            return block_idx, block_type, stream_tool, False
        block_type = "thinking"
    if content:
        if not await _emit_anthropic_event(
            resp,
            {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {"type": "thinking_delta", "thinking": content},
            },
            disconnected,
        ):
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
    """澧為噺鍙戦�� parser 涓�宸插畬鏁撮棴鍚堢殑 tool_use 鍧楋紙瀵归綈 mock.py input_json_delta锛夈��"""
    if ready is None:
        ready = parser.get_ready_tool_calls()
    if not ready:
        return block_idx, block_type, pending_tc_count, True
    if block_type is not None:
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    fixed = [_fix_tool_call_id(tc) for tc in ready]
    block_idx = await _send_tool_use_blocks(resp, fixed, block_idx, disconnected)
    if disconnected[0]:
        return block_idx, block_type, pending_tc_count, False
    return block_idx, block_type, pending_tc_count + len(fixed), True


async def _apply_streaming_tool_json_delta(
    resp,
    parser,
    state: Dict[str, Any],
    disconnected: list,
    name: str,
    partial_json: str,
) -> bool:
    block_idx = state["block_idx"]
    block_type = state["block_type"]
    stream_tool = state["stream_tool"]
    if block_type is not None and block_type != "tool_use":
        block_idx = await _close_block(resp, block_idx, disconnected)
        block_type = None
    if stream_tool is not None and stream_tool.get("name") != name:
        if not await _flush_open_stream_tool(resp, parser, stream_tool, disconnected):
            return False
        stream_tool = None
    if stream_tool is None:
        block_idx += 1
        stream_tool = {
            "block_idx": block_idx,
            "tool_id": _gen_tool_id(),
            "name": name,
            "json_buf": "",
        }
        if not await _emit_anthropic_event(
            resp,
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": stream_tool["tool_id"],
                    "name": name,
                    "input": {},
                },
            },
            disconnected,
        ):
            return False
        block_type = "tool_use"
    ok = await _emit_tool_json_pieces(resp, stream_tool, partial_json, disconnected)
    state["block_idx"] = block_idx
    state["block_type"] = block_type
    state["stream_tool"] = stream_tool
    return ok


async def _emit_streaming_tool_delta(
    resp,
    parser,
    block_idx: int,
    block_type: Optional[str],
    stream_tool: Optional[Dict[str, Any]],
    disconnected: list,
) -> Tuple[int, Optional[str], Optional[Dict[str, Any]], bool]:
    """invoke 开标签就绪后增量发送 input_json_delta。"""

    state = {
        "block_idx": block_idx,
        "block_type": block_type,
        "stream_tool": stream_tool,
    }

    async def _on_delta(name: str, partial_json: str) -> bool:
        return await _apply_streaming_tool_json_delta(
            resp,
            parser,
            state,
            disconnected,
            name,
            partial_json,
        )

    ok = await emit_parser_stream_deltas(parser, _on_delta)
    return state["block_idx"], state["block_type"], state["stream_tool"], ok


async def _close_streaming_tool_block(
    resp,
    stream_tool: Optional[Dict[str, Any]],
    block_idx: int,
    disconnected: list,
    *,
    parser=None,
    expected_arguments: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """鍏抽棴娴佸紡 tool_use 鍧楋紙鍏煎�规棫璋冪敤锛涗紭鍏堣蛋 _flush_open_stream_tool锛夈��"""
    if stream_tool is None:
        return block_idx, None
    if parser is not None:
        await _flush_open_stream_tool(
            resp,
            parser,
            stream_tool,
            disconnected,
            expected_arguments=expected_arguments,
        )
        return block_idx, None
    await _emit_anthropic_event(
        resp,
        _content_block_stop_event(stream_tool["block_idx"]),
        disconnected,
    )
    return block_idx, None
