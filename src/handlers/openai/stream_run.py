from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from aiohttp import web
from echotools import FncallStreamParser
from echotools.base.logger import get_logger

from handlers.openai.chat import _chat_once
from handlers.openai.stream_tools import (
    OpenAIStreamState,
    _emit_chunk,
    _emit_openai_streaming_tool_argument_pieces,
    _emit_partial_thinking,
    _emit_ready_tool_calls,
    _process_openai_stream_event,
    _safe_write,
    _send_stream_finish,
    _write_openai_stream_error,
)
from handlers.shared.api_errors import handler_error_response, log_classified_stream_error
from handlers.shared.fncall_inject import (
    advance_partial_buffer,
    finalize_parser_tool_calls,
    reconcile_pending_tool_index,
    resolve_streamed_tool_calls,
    take_parser_final_delta,
)
from server.formats import (
    _gen_chatcmpl_id,
    log_qwen_upstream_usage,
)
from server.records.response_record import record_raw_response
from server.records.sse_record import record_sse_stream
from state import QueueFullError, tracked_request

logger = get_logger("rogator")


async def _flush_remaining_thinking_and_text(
    st: OpenAIStreamState,
    final_text: str,
) -> None:
    if st.disconnected[0]:
        return
    await _emit_partial_thinking(st)

    if st.disconnected[0]:
        return
    safe_text = (
        st.parser.partial_text
        if st.parser.has_calls
        else (final_text or st.parser.partial_text)
    )
    new_text, st.last_safe_len = advance_partial_buffer(st.last_safe_len, safe_text)
    if new_text:
        chunk = st.stream_chunk(content=new_text)
        await _emit_chunk(st.resp, chunk, st.disconnected)


async def _finalize_openai_stream_tool(
    st: OpenAIStreamState, all_tool_calls: List[Dict[str, Any]]
) -> None:
    if st.disconnected[0]:
        return
    late_ready = st.parser.get_ready_tool_calls()
    if late_ready:
        await _emit_ready_tool_calls(st, late_ready)
        return
    if st.stream_tool is None:
        return
    final_delta = take_parser_final_delta(st.parser)
    if final_delta:
        _, piece = final_delta
        if piece:
            await _emit_openai_streaming_tool_argument_pieces(
                st.resp,
                st.model,
                st.chunk_id,
                st.stream_tool,
                piece,
                st.disconnected,
                include_usage=st.include_usage,
            )
    st.stream_tool = None
    if st.parser.streaming_invoke_closed or all_tool_calls:
        st.pending_tc_index += 1
    else:
        logger.warning("OpenAI stream ended with incomplete invoke %s", st.req_id)


def _resolve_all_tool_calls(
    st: OpenAIStreamState,
    parsed_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_tool_calls = resolve_streamed_tool_calls(parsed_calls, st.streamed_tool_calls)
    st.pending_tc_index = reconcile_pending_tool_index(
        st.pending_tc_index,
        all_tool_calls,
        st.stream_tool_blocks_sent,
    )
    return all_tool_calls


async def _run_openai_event_stream(
    st: OpenAIStreamState, state, messages, model, tools, protocol_options
) -> None:
    # 惰性导入：避免 chat_request ↔ openai 包初始化循环
    from handlers.chat_request import iter_retried_chat_events

    async def _make_stream():
        async for event in _chat_once(
            state,
            messages,
            model,
            tools,
            st.req_id,
            protocol_options=protocol_options,
            prompt_api="openai",
        ):
            yield event

    with record_sse_stream(st.req_id), record_raw_response(st.req_id) as raw_recorder:

        async def _on_event(event: Dict[str, Any]) -> bool:
            raw_recorder.ingest_event(event)
            return await _process_openai_stream_event(st, event)

        await iter_retried_chat_events(
            st.req_id,
            state,
            _make_stream,
            model=model,
            messages=messages,
            tools=tools,
            disconnected=st.disconnected,
            on_event=_on_event,
        )


async def _write_classified_openai_stream_error(
    resp, exc: BaseException, disconnected: list
) -> None:
    info = log_classified_stream_error(exc, label="OpenAI stream")
    await _write_openai_stream_error(
        resp,
        info.message,
        disconnected,
        error_type=info.kind,
        code=info.code,
    )


async def _run_openai_stream_guarded(
    st: OpenAIStreamState,
    state,
    messages,
    model,
    tools,
    protocol_options,
    req_id,
    resp,
) -> Optional[web.StreamResponse]:
    try:
        await _run_openai_event_stream(
            st, state, messages, model, tools, protocol_options
        )
    except asyncio.CancelledError:
        logger.info("OpenAI stream cancelled %s", req_id)
        await _safe_write(resp, b"data: [DONE]\n\n", st.disconnected)
        raise
    except Exception as e:
        await _write_classified_openai_stream_error(resp, e, st.disconnected)
        return resp
    return None


async def _finish_openai_stream(
    st: OpenAIStreamState, resp, model, include_usage
) -> web.StreamResponse:
    final_text, parsed_calls = finalize_parser_tool_calls(
        st.parser,
        warn=logger.warning,
    )
    await _flush_remaining_thinking_and_text(st, final_text)
    await _finalize_openai_stream_tool(st, parsed_calls)
    all_tool_calls = _resolve_all_tool_calls(st, parsed_calls)
    usage = st.usage_tracker.openai_stream_usage()
    emit_usage = include_usage and usage is not None
    await _send_stream_finish(
        resp,
        model,
        st.chunk_id,
        all_tool_calls,
        st.disconnected,
        already_sent_tc_count=st.pending_tc_index,
        usage=usage,
        include_usage=emit_usage,
    )
    return resp


def _new_openai_stream_state(
    *,
    model: str,
    resp: web.StreamResponse,
    include_usage: bool,
    state,
    tools,
    protocol_options,
    req_id: str,
) -> OpenAIStreamState:
    return OpenAIStreamState(
        model=model,
        chunk_id=_gen_chatcmpl_id(),
        resp=resp,
        disconnected=[False],
        include_usage=include_usage,
        parser=FncallStreamParser(
            protocol=state.protocol,
            tools=tools,
            protocol_options=protocol_options,
        ),
        req_id=req_id,
    )


async def _prepare_openai_stream_response(request) -> web.StreamResponse:
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)
    return resp


async def _handle_stream(
    request,
    state,
    messages,
    model,
    req_id,
    tools,
    protocol_options=None,
    *,
    include_usage: bool = False,
):
    try:
        async with tracked_request(state, req_id):
            resp = await _prepare_openai_stream_response(request)
            st = _new_openai_stream_state(
                model=model,
                resp=resp,
                include_usage=include_usage,
                state=state,
                tools=tools,
                protocol_options=protocol_options,
                req_id=req_id,
            )
            try:
                early = await _run_openai_stream_guarded(
                    st, state, messages, model, tools, protocol_options, req_id, resp,
                )
                if early is not None:
                    return early
                if st.disconnected[0]:
                    logger.info("Client disconnected during stream %s", req_id)
                    return resp
                return await _finish_openai_stream(st, resp, model, include_usage)
            except asyncio.CancelledError:
                logger.info("OpenAI stream cancelled during shutdown %s", req_id)
                raise
            except Exception as e:
                logger.error(
                    "OpenAI stream error (uncaught path) %s: %s",
                    req_id,
                    e,
                    exc_info=True,
                )
                await _write_classified_openai_stream_error(resp, e, st.disconnected)
                return resp
            finally:
                log_qwen_upstream_usage(req_id, st.usage_tracker)
    except QueueFullError as exc:
        return handler_error_response(exc, label="OpenAI stream")
