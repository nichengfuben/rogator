from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from server.formats import (
    UpstreamUsageTracker,
    _fix_tool_call_id,
    log_qwen_upstream_usage,
)
from server.records.response_record import record_raw_response
from echotools.fncall import FncallStreamParser
from echotools.logger import get_logger

from handlers.api_errors import classify_stream_error
from handlers.fncall_inject import (
    finalize_parser_tool_calls,
    reconcile_pending_tool_index,
    resolve_streamed_tool_calls,
)
from handlers.anthro.stream_content import _process_anthropic_content_events, _stream_event_loop
from handlers.anthro.events import (
    AnthropicStreamState,
    _close_block,
    _emit_anthropic_event,
    _message_start_event,
    _write_stream_error,
    expected_arguments_for_stream_tool,
    stream_result_tuple,
)
from handlers.anthro.stream_tools import (
    _emit_ready_tool_calls,
    _flush_open_stream_tool,
    _send_thinking_delta,
)

logger = get_logger("rogator")


async def _ensure_anthropic_message_start(
    resp,
    model: str,
    msg_id: str,
    usage_tracker: UpstreamUsageTracker,
    state: AnthropicStreamState,
    disconnected: list,
) -> None:
    if state.message_started or disconnected[0]:
        return
    await _emit_anthropic_event(
        resp,
        _message_start_event(model, msg_id, usage_tracker.anthropic_message_start_usage),
        disconnected,
    )
    state.message_started = True


async def _maybe_emit_message_start(
    resp,
    model: str,
    msg_id: str,
    usage_tracker: UpstreamUsageTracker,
    state: AnthropicStreamState,
    disconnected: list,
) -> None:
    await _ensure_anthropic_message_start(
        resp, model, msg_id, usage_tracker, state, disconnected,
    )


async def _handle_classified_stream_error(
    resp, state, usage_tracker, e, disconnected,
) -> Tuple[int, Optional[str], str, bool, int, List[Dict[str, Any]], UpstreamUsageTracker]:
    info = classify_stream_error(e)
    if info.kind == "rate_limited":
        logger.warning("Anthropic stream token expired: %s", e)
    elif info.kind == "timeout":
        logger.warning("Anthropic stream upstream timeout: %s", e)
    else:
        logger.error("Anthropic stream error: %s", e, exc_info=True)
    err_body: Dict[str, Any] = {"message": info.message}
    if info.kind != "server_error":
        err_body["type"] = info.kind
    await _write_stream_error(resp, {"type": "error", "error": err_body}, disconnected)
    return stream_result_tuple(state, usage_tracker, early_return=True)


async def _flush_remaining_thinking(
    resp, parser, state: AnthropicStreamState, disconnected: list,
) -> bool:
    if disconnected[0]:
        return True
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


async def _emit_final_text_delta(
    resp, new_text: str, state: AnthropicStreamState, disconnected: list,
) -> bool:
    if state.stream_tool is not None:
        return False
    if state.block_type != "text":
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
    if state.block_type != "text":
        return True
    await _emit_anthropic_event(resp, {
        "type": "content_block_delta",
        "index": state.block_idx,
        "delta": {"type": "text_delta", "text": new_text},
    }, disconnected)
    return True


async def _flush_remaining_text(
    resp, parser, state: AnthropicStreamState, final_text: str, disconnected: list,
) -> bool:
    if disconnected[0]:
        return True
    safe_text = parser.partial_text if parser.has_calls else (final_text or parser.partial_text)
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
    return await _emit_final_text_delta(resp, new_text, state, disconnected)


async def _flush_ready_tool_calls_with_stream_tool(
    resp, parser, state: AnthropicStreamState, fixed: List[Dict[str, Any]], disconnected: list,
) -> bool:
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


async def _handle_late_ready_tools(
    resp, parser, state: AnthropicStreamState, disconnected: list,
) -> bool:
    late_ready = parser.get_ready_tool_calls()
    if not late_ready:
        return True
    fixed = [_fix_tool_call_id(tc) for tc in late_ready]
    state.streamed_tool_calls.extend(fixed)
    if state.stream_tool is not None:
        return await _flush_ready_tool_calls_with_stream_tool(
            resp, parser, state, fixed, disconnected,
        )
    state.block_idx, state.block_type, state.pending_tc_count, ok = await _emit_ready_tool_calls(
        resp, parser, state.block_idx, state.block_type, disconnected,
        state.pending_tc_count, ready=late_ready,
    )
    return ok


async def _handle_open_stream_tool_at_end(
    resp, parser, state: AnthropicStreamState, req_id: str, disconnected: list,
) -> bool:
    if state.stream_tool is None:
        return True
    expected = None
    if state.all_tool_calls:
        expected = state.all_tool_calls[0]["function"]["arguments"]
    if not await _flush_open_stream_tool(
        resp, parser, state.stream_tool, disconnected,
        expected_arguments=expected,
    ):
        return False
    state.stream_tool = None
    state.block_type = None
    if parser.streaming_invoke_closed or state.all_tool_calls:
        state.pending_tc_count += 1
    else:
        logger.warning("Anthropic stream ended with incomplete invoke %s", req_id)
    return True


async def _finalize_open_tools(
    resp, parser, state: AnthropicStreamState, req_id: str, disconnected: list,
) -> bool:
    if disconnected[0]:
        return True
    late_ready = parser.get_ready_tool_calls()
    if late_ready:
        return await _handle_late_ready_tools(resp, parser, state, disconnected)
    if state.stream_tool is not None:
        return await _handle_open_stream_tool_at_end(
            resp, parser, state, req_id, disconnected,
        )
    state.block_idx, state.block_type, state.pending_tc_count, ok = await _emit_ready_tool_calls(
        resp, parser, state.block_idx, state.block_type, disconnected,
        state.pending_tc_count,
    )
    return ok


def _reconcile_tool_calls(state: AnthropicStreamState) -> None:
    state.all_tool_calls = resolve_streamed_tool_calls(
        state.all_tool_calls, state.streamed_tool_calls,
    )
    state.pending_tc_count = reconcile_pending_tool_index(
        state.pending_tc_count, state.all_tool_calls, state.stream_tool_blocks_sent,
    )


async def _complete_anthropic_stream(
    resp, parser, stream_state, usage_tracker, model, msg_id, req_id, disconnected,
) -> Tuple[int, Optional[str], str, bool, int, List[Dict[str, Any]], UpstreamUsageTracker]:
    final_text, parsed_calls = finalize_parser_tool_calls(
        parser,
        warn=logger.warning,
        warn_prefix="anthropic stream parser.finalize failed",
    )
    stream_state.all_tool_calls = parsed_calls

    if not await _flush_remaining_thinking(resp, parser, stream_state, disconnected):
        return stream_result_tuple(stream_state, usage_tracker, early_return=True)
    if not await _flush_remaining_text(resp, parser, stream_state, final_text, disconnected):
        return stream_result_tuple(stream_state, usage_tracker, early_return=True)
    if not await _finalize_open_tools(resp, parser, stream_state, req_id, disconnected):
        return stream_result_tuple(stream_state, usage_tracker, early_return=True)

    _reconcile_tool_calls(stream_state)
    await _maybe_emit_message_start(
        resp, model, msg_id, usage_tracker, stream_state, disconnected,
    )

    if stream_state.deferred_content and not disconnected[0]:
        if not await _process_anthropic_content_events(
            resp, stream_state.deferred_content, parser, stream_state, disconnected,
        ):
            return stream_result_tuple(stream_state, usage_tracker, early_return=True)
        stream_state.deferred_content = []

    return stream_result_tuple(stream_state, usage_tracker, early_return=False)


async def _stream_anthropic(
    resp, state, messages, model, tools, req_id, disconnected, protocol_options=None,
    *, msg_id: str,
) -> Tuple[int, Optional[str], str, bool, int, List[Dict[str, Any]], UpstreamUsageTracker]:
    stream_state = AnthropicStreamState()
    parser = FncallStreamParser(protocol=state.protocol, tools=tools, protocol_options=protocol_options)
    usage_tracker = UpstreamUsageTracker()

    with record_raw_response(req_id) as raw_recorder:
        try:
            await _stream_event_loop(
                resp, state, messages, model, tools, req_id, disconnected,
                protocol_options, msg_id, stream_state, parser, usage_tracker, raw_recorder,
            )
        except asyncio.CancelledError:
            logger.info("Stream cancelled %s", req_id)
            raise
        except Exception as e:
            return await _handle_classified_stream_error(
                resp, stream_state, usage_tracker, e, disconnected,
            )
        finally:
            log_qwen_upstream_usage(req_id, usage_tracker)

        return await _complete_anthropic_stream(
            resp, parser, stream_state, usage_tracker, model, msg_id, req_id, disconnected,
        )
