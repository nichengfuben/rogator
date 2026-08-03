from __future__ import annotations

from typing import Any, Dict, List, Optional

from handlers.anthropic.events import AnthropicStreamState
from handlers.anthropic.stream_content import _process_anthropic_content_events
from handlers.chat_request import iter_retried_chat_events
from handlers.openai import _chat_once
from server.formats import UpstreamUsageTracker, should_emit_anthropic_message_start
from server.records.sse_record import record_sse_stream


async def _make_anthropic_chat_stream(
    state, messages, model, tools, req_id, protocol_options
):
    async for event in _chat_once(
        state,
        messages,
        model,
        tools,
        req_id,
        protocol_options=protocol_options,
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
        if should_emit_anthropic_message_start(event, False) or etype in (
            "thinking",
            "answer",
        ):
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
    from handlers.anthropic.stream_core import _ensure_anthropic_message_start

    etype = usage_tracker.ingest_upstream_event(event)
    if etype == "prompt_meta":
        await _ensure_anthropic_message_start(
            resp,
            model,
            msg_id,
            usage_tracker,
            stream_state,
            disconnected,
        )
        return True

    raw_recorder.ingest_event(event)

    if not stream_state.message_started:
        if etype in ("response_created",):
            return True
        if should_emit_anthropic_message_start(event, False) or etype in (
            "thinking",
            "answer",
        ):
            await _ensure_anthropic_message_start(
                resp,
                model,
                msg_id,
                usage_tracker,
                stream_state,
                disconnected,
            )
    return None


async def _dispatch_stream_event(
    resp,
    event: Dict[str, Any],
    *,
    model: str,
    msg_id: str,
    stream_state: AnthropicStreamState,
    usage_tracker: UpstreamUsageTracker,
    raw_recorder,
    disconnected: list,
    parser,
) -> bool:
    ingest = await _ingest_stream_event(
        resp,
        event,
        model=model,
        msg_id=msg_id,
        stream_state=stream_state,
        usage_tracker=usage_tracker,
        raw_recorder=raw_recorder,
        disconnected=disconnected,
    )
    if ingest is True:
        return True
    to_process = await _events_to_process(event, stream_state)
    if to_process is None:
        return True
    return await _process_anthropic_content_events(
        resp,
        to_process,
        parser,
        stream_state,
        disconnected,
    )


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
    async def _on_event(event: Dict[str, Any]) -> bool:
        return await _dispatch_stream_event(
            resp,
            event,
            model=model,
            msg_id=msg_id,
            stream_state=stream_state,
            usage_tracker=usage_tracker,
            raw_recorder=raw_recorder,
            disconnected=disconnected,
            parser=parser,
        )

    with record_sse_stream(req_id):
        await iter_retried_chat_events(
            req_id,
            state_obj,
            lambda: _make_anthropic_chat_stream(
                state_obj,
                messages,
                model,
                tools,
                req_id,
                protocol_options,
            ),
            model=model,
            messages=messages,
            tools=tools,
            disconnected=disconnected,
            on_event=_on_event,
        )
