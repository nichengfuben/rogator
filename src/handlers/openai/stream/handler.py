from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web
from echotools.fncall import FncallStreamParser
from echotools.logger import get_logger

from handlers.api_errors import handler_error_response
from server.formats import as_upstream_connection_error
from handlers.openai.chat import _chat_once, _resolve_retry_client
from handlers.openai.stream.tools import (
    _emit_chunk,
    _emit_openai_streaming_tool_argument_pieces,
    _emit_openai_streaming_tool_delta,
    _emit_tool_call_chunks,
    _handle_uncaught_openai_stream_error,
    _safe_write,
    _send_stream_finish,
    _write_openai_stream_error,
)
from server.formats import (
    TokenExpiredError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    UpstreamConnectionError,
    UpstreamUsageTracker,
    log_qwen_upstream_usage,
    _fix_tool_call_id,
    _gen_chatcmpl_id,
    build_openai_chunk,
)
from server.model.model_registry import ModelRegistryEntry, is_native_upstream_event
from server.records.response_record import record_raw_response
from server.retry import stream_with_session_retry
from state import QueueFullError, tracked_request

logger = get_logger("rogator")


@dataclass
class OpenAIStreamState:
    """流式 OpenAI 响应的 mutable 状态容器。"""

    model: str
    chunk_id: str
    resp: web.StreamResponse
    disconnected: list
    include_usage: bool
    parser: FncallStreamParser
    req_id: str
    full_answer: str = ""
    full_thinking: str = ""
    last_safe_len: int = 0
    last_thinking_len: int = 0
    pending_tc_index: int = 0
    streamed_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stream_tool: Optional[Dict[str, Any]] = None
    stream_tool_blocks_sent: int = 0
    native_upstream: bool = False
    registry_entry: Optional[ModelRegistryEntry] = None
    usage_tracker: UpstreamUsageTracker = field(default_factory=UpstreamUsageTracker)

    def stream_chunk(self, **kwargs: Any) -> Dict[str, Any]:
        if self.include_usage:
            usage = self.usage_tracker.openai_stream_usage()
            if usage is not None:
                kwargs["usage"] = usage
            else:
                kwargs["usage_null"] = True
        return build_openai_chunk(self.model, chunk_id=self.chunk_id, **kwargs)


async def _emit_ready_tool_calls(
    st: OpenAIStreamState, ready_calls: List[Dict[str, Any]],
) -> None:
    fixed = [_fix_tool_call_id(tc) for tc in ready_calls]
    st.streamed_tool_calls.extend(fixed)
    if st.stream_tool is not None:
        st.stream_tool = None
        st.pending_tc_index += len(fixed)
        return
    st.pending_tc_index = await _emit_tool_call_chunks(
        st.resp, st.model, st.chunk_id, fixed, st.pending_tc_index, st.disconnected,
        include_usage=st.include_usage,
    )


async def _emit_partial_thinking(st: OpenAIStreamState) -> bool:
    pt = st.parser.partial_thinking
    if len(pt) <= st.last_thinking_len:
        return True
    new_thinking = pt[st.last_thinking_len:]
    st.last_thinking_len = len(pt)
    st.full_thinking += new_thinking
    chunk = st.stream_chunk(reasoning=new_thinking)
    return await _emit_chunk(st.resp, chunk, st.disconnected)


async def _emit_partial_text(st: OpenAIStreamState) -> bool:
    safe_text = st.parser.partial_text
    if len(safe_text) <= st.last_safe_len:
        return True
    new_text = safe_text[st.last_safe_len:]
    st.last_safe_len = len(safe_text)
    chunk = st.stream_chunk(content=new_text)
    return await _emit_chunk(st.resp, chunk, st.disconnected)


async def _process_openai_stream_thinking(st: OpenAIStreamState, content: str) -> bool:
    st.full_thinking += content
    if not content:
        return True
    chunk = st.stream_chunk(reasoning=content)
    return await _emit_chunk(st.resp, chunk, st.disconnected)


async def _process_openai_stream_answer(st: OpenAIStreamState, content: str) -> bool:
    st.full_answer += content
    ready_calls = st.parser.feed(content)

    had_stream_tool = st.stream_tool is not None
    st.stream_tool, st.pending_tc_index, ok = await _emit_openai_streaming_tool_delta(
        st.resp, st.parser, st.model, st.chunk_id, st.stream_tool, st.pending_tc_index, st.disconnected,
        include_usage=st.include_usage,
    )
    if not had_stream_tool and st.stream_tool is not None:
        st.stream_tool_blocks_sent += 1
    if not ok:
        return False

    if not await _emit_partial_thinking(st):
        return False
    if not await _emit_partial_text(st):
        return False

    if ready_calls:
        await _emit_ready_tool_calls(st, ready_calls)
    return True


async def _process_native_stream_event(st: OpenAIStreamState, event: Dict[str, Any]) -> bool:
    etype = event.get("type")
    if etype == "prompt_meta":
        st.usage_tracker.set_estimated_input_from_prompt_chars(int(event.get("prompt_chars") or 0))
        return True
    st.usage_tracker.ingest_event(event)
    if etype in ("response_created", "usage"):
        return True
    if etype == "thinking":
        content = event.get("content", "")
        if content:
            st.usage_tracker.add_output_chars(len(content))
        return await _process_openai_stream_thinking(st, content)
    if etype == "answer":
        content = event.get("content", "")
        if content:
            st.usage_tracker.add_output_chars(len(content))
            st.full_answer += content
            chunk = st.stream_chunk(content=content)
            return await _emit_chunk(st.resp, chunk, st.disconnected)
        return True
    if etype == "tool_call":
        tc = event.get("tool_call")
        if tc:
            await _emit_ready_tool_calls(st, [tc])
        return True
    return True


async def _process_openai_stream_event(st: OpenAIStreamState, event: Dict[str, Any]) -> bool:
    """处理单个上游事件；返回 False 表示应中断流。"""
    if is_native_upstream_event(st.registry_entry, event):
        st.native_upstream = True
        return await _process_native_stream_event(st, event)

    etype = event.get("type")
    if etype == "prompt_meta":
        st.usage_tracker.set_estimated_input_from_prompt_chars(int(event.get("prompt_chars") or 0))
        return True
    st.usage_tracker.ingest_event(event)
    if etype in ("response_created", "usage"):
        return True
    content = event.get("content", "")
    if content and etype in ("thinking", "answer"):
        st.usage_tracker.add_output_chars(len(content))
    if etype == "thinking":
        return await _process_openai_stream_thinking(st, content)
    if etype != "answer":
        return True
    return await _process_openai_stream_answer(st, content)


def _finalize_parser_tool_calls(st: OpenAIStreamState) -> tuple[str, List[Dict[str, Any]]]:
    if st.native_upstream:
        return st.full_answer, list(st.streamed_tool_calls)
    final_text = st.parser.partial_text
    try:
        final_text, parsed_calls = st.parser.finalize()
        return final_text, [_fix_tool_call_id(tc) for tc in parsed_calls]
    except Exception as e:
        logger.warning("stream parser.finalize failed: %s", e)
        return final_text, []


async def _flush_remaining_thinking_and_text(
    st: OpenAIStreamState, final_text: str,
) -> None:
    if st.disconnected[0] or st.native_upstream:
        return
    await _emit_partial_thinking(st)
    if st.disconnected[0]:
        return
    safe_text = st.parser.partial_text if st.parser.has_calls else (final_text or st.parser.partial_text)
    if len(safe_text) > st.last_safe_len:
        new_text = safe_text[st.last_safe_len:]
        st.last_safe_len = len(safe_text)
        if new_text:
            chunk = st.stream_chunk(content=new_text)
            await _emit_chunk(st.resp, chunk, st.disconnected)


async def _finalize_openai_stream_tool(st: OpenAIStreamState, all_tool_calls: List[Dict[str, Any]]) -> None:
    if st.disconnected[0] or st.native_upstream:
        return
    late_ready = st.parser.get_ready_tool_calls()
    if late_ready:
        await _emit_ready_tool_calls(st, late_ready)
        return
    if st.stream_tool is None:
        return
    if not st.parser.streaming_invoke_closed:
        final_delta = st.parser.complete_stream_delta_if_needed()
        if final_delta:
            _, piece = final_delta
            if piece:
                await _emit_openai_streaming_tool_argument_pieces(
                    st.resp, st.model, st.chunk_id, st.stream_tool, piece, st.disconnected,
                    include_usage=st.include_usage,
                )
    st.stream_tool = None
    if st.parser.streaming_invoke_closed or all_tool_calls:
        st.pending_tc_index += 1
    else:
        logger.warning("OpenAI stream ended with incomplete invoke %s", st.req_id)


def _resolve_all_tool_calls(
    st: OpenAIStreamState, parsed_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_tool_calls = parsed_calls or st.streamed_tool_calls
    if all_tool_calls and st.stream_tool_blocks_sent:
        st.pending_tc_index = max(
            st.pending_tc_index,
            min(len(all_tool_calls), st.stream_tool_blocks_sent),
        )
    return all_tool_calls


async def _run_openai_event_stream(st: OpenAIStreamState, state, messages, model, tools, protocol_options) -> None:
    retry_client = _resolve_retry_client(state, model, messages, tools)

    async def _make_stream():
        async for event in _chat_once(
            state, messages, model, tools, st.req_id, protocol_options=protocol_options,
            prompt_api="openai",
        ):
            yield event

    with record_raw_response(st.req_id) as raw_recorder:
        async with aclosing(
            stream_with_session_retry(st.req_id, state, _make_stream, client=retry_client),
        ) as event_stream:
            async for event in event_stream:
                if st.disconnected[0]:
                    break
                raw_recorder.ingest_event(event)
                if not await _process_openai_stream_event(st, event):
                    break


async def _run_openai_stream_guarded(
    st: OpenAIStreamState, state, messages, model, tools, protocol_options, req_id, resp,
) -> Optional[web.StreamResponse]:
    try:
        await _run_openai_event_stream(st, state, messages, model, tools, protocol_options)
    except asyncio.CancelledError:
        logger.info("OpenAI stream cancelled %s", req_id)
        await _safe_write(resp, b"data: [DONE]\n\n", st.disconnected)
        raise
    except TokenExpiredError as e:
        logger.warning("OpenAI stream token expired: %s", e)
        await _write_openai_stream_error(resp, str(e), st.disconnected, error_type="rate_limited", code=429)
        return resp
    except UpstreamTimeoutError as e:
        logger.warning("OpenAI stream upstream timeout: %s", e)
        await _write_openai_stream_error(resp, str(e), st.disconnected, error_type="timeout", code=504)
        return resp
    except UpstreamUnavailableError as e:
        logger.debug("OpenAI stream upstream unavailable: %s", e.message)
        await _write_openai_stream_error(
            resp, e.message, st.disconnected, error_type=e.error_type, code=e.status,
        )
        return resp
    except Exception as e:
        conn_err = as_upstream_connection_error(e)
        if conn_err is not None:
            logger.warning("OpenAI stream upstream connection: %s", conn_err.message)
            await _write_openai_stream_error(
                resp, conn_err.message, st.disconnected,
                error_type=conn_err.error_type, code=conn_err.status,
            )
            return resp
        logger.error("OpenAI stream error: %s", e, exc_info=True)
        await _write_openai_stream_error(resp, str(e), st.disconnected)
        return resp
    return None


async def _finish_openai_stream(st: OpenAIStreamState, resp, model, include_usage) -> web.StreamResponse:
    final_text, parsed_calls = _finalize_parser_tool_calls(st)
    await _flush_remaining_thinking_and_text(st, final_text)
    await _finalize_openai_stream_tool(st, parsed_calls)
    all_tool_calls = _resolve_all_tool_calls(st, parsed_calls)
    usage = st.usage_tracker.openai_stream_usage()
    emit_usage = include_usage and usage is not None
    await _send_stream_finish(
        resp, model, st.chunk_id, all_tool_calls, st.disconnected,
        already_sent_tc_count=st.pending_tc_index,
        usage=usage,
        include_usage=emit_usage,
    )
    return resp


async def _execute_openai_stream_body(
    st: OpenAIStreamState,
    state,
    messages,
    model,
    tools,
    protocol_options,
    req_id: str,
    resp: web.StreamResponse,
    *,
    include_usage: bool,
) -> web.StreamResponse:
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
        await _handle_uncaught_openai_stream_error(resp, st, req_id, e, logger)
        return resp


async def handle_openai_stream(
    request, state, messages, model, req_id, tools, protocol_options=None, *,
    include_usage: bool = False,
    registry_entry: Optional[ModelRegistryEntry] = None,
):
    try:
        async with tracked_request(state, req_id):
            resp = web.StreamResponse(status=200, headers={
                "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                "Connection": "keep-alive", "X-Accel-Buffering": "no",
            })
            await resp.prepare(request)
            st = OpenAIStreamState(
                model=model,
                chunk_id=_gen_chatcmpl_id(),
                resp=resp,
                disconnected=[False],
                include_usage=include_usage,
                parser=FncallStreamParser(protocol=state.protocol, tools=tools, protocol_options=protocol_options),
                req_id=req_id,
                registry_entry=registry_entry,
            )

            try:
                return await _execute_openai_stream_body(
                    st, state, messages, model, tools, protocol_options, req_id, resp,
                    include_usage=include_usage,
                )
            finally:
                log_qwen_upstream_usage(req_id, st.usage_tracker)
    except QueueFullError as exc:
        return handler_error_response(exc, label="OpenAI stream")
