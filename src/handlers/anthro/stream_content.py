from __future__ import annotations

import json
from contextlib import aclosing
from typing import Any, Dict, List, Optional, Tuple

from server.formats import UpstreamUsageTracker, _fix_tool_call_id, should_emit_anthropic_message_start
from server.retry import stream_with_session_retry

from handlers.openai import _chat_once
from handlers.openai.chat import _resolve_retry_client

from handlers.anthro.events import _close_block, _emit_anthropic_event
from handlers.anthro.events import AnthropicStreamState, expected_arguments_for_stream_tool
from handlers.anthro.stream_tools import (
    _emit_ready_tool_calls,
    _emit_streaming_tool_delta,
    _flush_open_stream_tool,
    _send_thinking_delta,
)


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
    state.block_idx, state.block_type, state.stream_tool, ok = await _send_thinking_delta(
        resp, content, state.block_idx, state.block_type, disconnected,
        parser=parser, stream_tool=state.stream_tool,
    )
    return ok


async def _emit_partial_thinking(
    resp,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    pt = parser.partial_thinking
    if len(pt) <= state.last_thinking_len:
        return True
    new_thinking = pt[state.last_thinking_len:]
    state.last_thinking_len = len(pt)
    if not new_thinking:
        return True
    state.block_idx, state.block_type, state.stream_tool, ok = await _send_thinking_delta(
        resp, new_thinking, state.block_idx, state.block_type, disconnected,
        parser=parser, stream_tool=state.stream_tool,
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
    if not await _emit_anthropic_event(resp, {
        "type": "content_block_start",
        "index": state.block_idx,
        "content_block": {"type": "text", "text": ""},
    }, disconnected):
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
    return await _emit_anthropic_event(resp, {
        "type": "content_block_delta",
        "index": state.block_idx,
        "delta": {"type": "text_delta", "text": new_text},
    }, disconnected)


async def _emit_safe_text_delta(
    resp,
    parser,
    state: AnthropicStreamState,
    disconnected: list,
) -> bool:
    safe_text = parser.partial_text
    if len(safe_text) <= state.last_safe_len:
        return True
    new_text = safe_text[state.last_safe_len:]
    state.last_safe_len = len(safe_text)
    if not new_text:
        return True
    if state.stream_tool is not None:
        if not await _flush_open_stream_tool(resp, parser, state.stream_tool, disconnected):
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
            resp, parser, state.stream_tool, disconnected,
            expected_arguments=expected,
        ):
            return False
        state.stream_tool = None
        state.block_type = None
        state.pending_tc_count += len(fixed)
        return True
    state.block_idx, state.block_type, state.pending_tc_count, ok = await _emit_ready_tool_calls(
        resp, parser, state.block_idx, state.block_type, disconnected,
        state.pending_tc_count, ready=ready_calls,
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
    state.block_idx, state.block_type, state.stream_tool, ok = await _emit_streaming_tool_delta(
        resp, parser, state.block_idx, state.block_type, state.stream_tool, disconnected,
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
        resp, parser, state, disconnected, ready_calls,
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
            resp, content, state, disconnected, parser=parser,
        )
    if etype != "answer":
        return True
    return await _process_answer_content_event(
        resp, content, parser, state, disconnected,
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
            resp, proc_event, parser, state, disconnected,
        ):
            return False
    return True


async def _make_anthropic_chat_stream(state, messages, model, tools, req_id, protocol_options):
    async for event in _chat_once(
        state, messages, model, tools, req_id, protocol_options=protocol_options,
        prompt_api="anthropic",
    ):
        yield event


async def _events_to_process(
    event: Dict[str, Any],
    stream_state: AnthropicStreamState,
) -> Optional[List[Dict[str, Any]]]:
    etype = event.get("type")
    if not stream_state.message_started:
        if etype in ("response_created",):
            return None
        if should_emit_anthropic_message_start(event, False) or etype in ("thinking", "answer"):
            to_process = stream_state.deferred_content + [event]
            stream_state.deferred_content = []
            return to_process
        return None
    return [event]


async def _ingest_stream_event(
    resp,
    event: Dict[str, Any],
    *,
    model: str,
    msg_id: str,
    stream_state: AnthropicStreamState,
    usage_tracker: UpstreamUsageTracker,
    raw_recorder,
    disconnected: list,
) -> Optional[bool]:
    """处理单条上游事件：True=继续下一条，None=进入内容解析。"""
    from handlers.anthro.stream_core import _ensure_anthropic_message_start

    etype = event.get("type")
    if etype == "prompt_meta":
        usage_tracker.set_estimated_input_from_prompt_chars(int(event.get("prompt_chars") or 0))
        await _ensure_anthropic_message_start(
            resp, model, msg_id, usage_tracker, stream_state, disconnected,
        )
        return True

    usage_tracker.ingest_event(event)
    raw_recorder.ingest_event(event)

    content = event.get("content", "")
    if content and etype in ("thinking", "answer"):
        usage_tracker.add_output_chars(len(content))

    if not stream_state.message_started:
        if etype in ("response_created",):
            return True
        if should_emit_anthropic_message_start(event, False) or etype in ("thinking", "answer"):
            await _ensure_anthropic_message_start(
                resp, model, msg_id, usage_tracker, stream_state, disconnected,
            )
    return None


async def _stream_event_loop(
    resp,
    state_obj,
    messages,
    model,
    tools,
    req_id,
    disconnected,
    protocol_options,
    msg_id,
    stream_state: AnthropicStreamState,
    parser,
    usage_tracker,
    raw_recorder,
) -> None:
    retry_client = _resolve_retry_client(state_obj, model, messages, tools)
    async with aclosing(
        stream_with_session_retry(
            req_id,
            state_obj,
            lambda: _make_anthropic_chat_stream(
                state_obj, messages, model, tools, req_id, protocol_options,
            ),
            client=retry_client,
        ),
    ) as event_stream:
        async for event in event_stream:
            if disconnected[0]:
                break
            ingest = await _ingest_stream_event(
                resp, event, model=model, msg_id=msg_id, stream_state=stream_state,
                usage_tracker=usage_tracker, raw_recorder=raw_recorder, disconnected=disconnected,
            )
            if ingest is True:
                continue
            to_process = await _events_to_process(event, stream_state)
            if to_process is None:
                continue
            if not await _process_anthropic_content_events(
                resp, to_process, parser, stream_state, disconnected,
            ):
                break
