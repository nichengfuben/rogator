from __future__ import annotations

from typing import Any, Dict, List, Optional

from handlers.anthropic.events import (
    AnthropicStreamState,
    _close_block,
    _emit_anthropic_event,
    expected_arguments_for_stream_tool,
)
from handlers.anthropic.stream_tools import (
    _emit_ready_tool_calls,
    _emit_streaming_tool_delta,
    _flush_open_stream_tool,
    _send_thinking_delta,
)
from server.formats import _fix_tool_call_id


async def _process_thinking_content_event(
    resp,
    content: str,
    state: AnthropicStreamState,
    disconnected: list,
    *,
    parser,
) -> bool:
    if not content:
        return True
    (
        state.block_idx,
        state.block_type,
        state.stream_tool,
        ok,
    ) = await _send_thinking_delta(
        resp,
        content,
        state.block_idx,
        state.block_type,
        disconnected,
        parser=parser,
        stream_tool=state.stream_tool,
    )
    return ok


async def _emit_partial_thinking(
    resp,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    new_thinking, state.last_thinking_len = advance_partial_buffer(
        state.last_thinking_len,
        parser.partial_thinking,
    )
    if not new_thinking:
        return True
    (
        state.block_idx,
        state.block_type,
        state.stream_tool,
        ok,
    ) = await _send_thinking_delta(
        resp,
        new_thinking,
        state.block_idx,
        state.block_type,
        disconnected,
        parser=parser,
        stream_tool=state.stream_tool,
    )
    return ok


async def _start_text_block_if_needed(
    resp,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    if state.block_type == "text":
        return True
    if state.block_type == "thinking":
        state.block_idx = await _close_block(resp, state.block_idx, disconnected)
        state.block_type = None
    state.block_idx += 1
    if not await _emit_anthropic_event(
        resp,
        {
            "type": "content_block_start",
            "index": state.block_idx,
            "content_block": {"type": "text", "text": ""},
        },
        disconnected,
    ):
        return False
    state.block_type = "text"
    return True


async def _emit_text_delta(
    resp,
    new_text: str,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    if state.stream_tool is not None:
        return False
    if not await _start_text_block_if_needed(resp, state, disconnected):
        return False
    return await _emit_anthropic_event(
        resp,
        {
            "type": "content_block_delta",
            "index": state.block_idx,
            "delta": {"type": "text_delta", "text": new_text},
        },
        disconnected,
    )


async def _emit_safe_text_delta(
    resp,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    new_text, state.last_safe_len = advance_partial_buffer(
        state.last_safe_len,
        parser.partial_text,
    )
    if not new_text:
        return True
    if state.stream_tool is not None:
        if not await _flush_open_stream_tool(
            resp, parser, state.stream_tool, disconnected
        ):
            return False
        state.stream_tool = None
        state.block_type = None
    return await _emit_text_delta(resp, new_text, state, disconnected)


async def _handle_ready_tool_calls_in_event(
    resp,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
    ready_calls: List[Dict[str, Any]],
) -> bool:
    if not ready_calls:
        return True
    fixed = [_fix_tool_call_id(tc) for tc in ready_calls]
    state.streamed_tool_calls.extend(fixed)
    if state.stream_tool is not None:
        expected = expected_arguments_for_stream_tool(fixed, state.stream_tool)
        if not await _flush_open_stream_tool(
            resp,
            parser,
            state.stream_tool,
            disconnected,
            expected_arguments=expected,
        ):
            return False
        state.stream_tool = None
        state.block_type = None
        state.pending_tc_count += len(fixed)
        return True
    (
        state.block_idx,
        state.block_type,
        state.pending_tc_count,
        ok,
    ) = await _emit_ready_tool_calls(
        resp,
        parser,
        state.block_idx,
        state.block_type,
        disconnected,
        state.pending_tc_count,
        ready=ready_calls,
    )
    return ok


async def _process_answer_content_event(
    resp,
    content: str,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    state.full_answer += content
    ready_calls = parser.feed(content)

    had_stream_tool = state.stream_tool is not None
    (
        state.block_idx,
        state.block_type,
        state.stream_tool,
        ok,
    ) = await _emit_streaming_tool_delta(
        resp,
        parser,
        state.block_idx,
        state.block_type,
        state.stream_tool,
        disconnected,
    )
    if not had_stream_tool and state.stream_tool is not None:
        state.stream_tool_blocks_sent += 1
    if not ok:
        return False

    if not await _emit_partial_thinking(resp, parser, state, disconnected):
        return False
    if not await _emit_safe_text_delta(resp, parser, state, disconnected):
        return False
    return await _handle_ready_tool_calls_in_event(
        resp,
        parser,
        state,
        disconnected,
        ready_calls,
    )


async def _process_anthropic_content_event(
    resp,
    proc_event: Dict[str, Any],
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    etype = proc_event.get("type")
    content = proc_event.get("content", "")
    if etype in ("response_created", "usage"):
        return True
    if etype == "thinking":
        return await _process_thinking_content_event(
            resp,
            content,
            state,
            disconnected,
            parser=parser,
        )
    if etype != "answer":
        return True
    return await _process_answer_content_event(
        resp,
        content,
        parser,
        state,
        disconnected,
    )


async def _process_anthropic_content_events(
    resp,
    events: List[Dict[str, Any]],
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    for proc_event in events:
        if not await _process_anthropic_content_event(
            resp,
            proc_event,
            parser,
            state,
            disconnected,
        ):
            return False
    return True

