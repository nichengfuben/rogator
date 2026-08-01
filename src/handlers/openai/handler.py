from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web
from echotools.fncall import FncallStreamParser
from echotools.logger import get_logger

from handlers import get_state
from handlers.api_errors import apply_tool_choice, handler_error_response, log_classified_stream_error
from handlers.fncall_inject import (
    advance_partial_buffer,
    finalize_parser_tool_calls,
    reconcile_pending_tool_index,
    resolve_streamed_tool_calls,
    take_parser_final_delta,
)
from handlers.openai.chat import _chat_once, _process_openai_non_stream
from handlers.openai.protocol import _build_protocol_options
from handlers.openai.stream_tools import (
    _emit_chunk,
    _emit_openai_streaming_tool_argument_pieces,
    _emit_openai_streaming_tool_delta,
    _emit_tool_call_chunks,
    _safe_write,
    _send_stream_finish,
    _write_openai_stream_error,
)
from server.config import CONFIG
from server.formats import (
    UpstreamUsageTracker,
    log_qwen_upstream_usage,
    openai_stream_include_usage,
    _error_response,
    _fix_tool_call_id,
    _gen_chatcmpl_id,
    _json_response,
    build_openai_chunk,
)
from server.records.response_record import record_raw_response
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
    new_thinking, st.last_thinking_len = advance_partial_buffer(
        st.last_thinking_len, st.parser.partial_thinking,
    )
    if not new_thinking:
        return True
    st.full_thinking += new_thinking
    chunk = st.stream_chunk(reasoning=new_thinking)
    return await _emit_chunk(st.resp, chunk, st.disconnected)


async def _emit_partial_text(st: OpenAIStreamState) -> bool:
    new_text, st.last_safe_len = advance_partial_buffer(
        st.last_safe_len, st.parser.partial_text,
    )
    if not new_text:
        return True
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


async def _process_openai_stream_event(st: OpenAIStreamState, event: Dict[str, Any]) -> bool:
    """处理单个上游事件；返回 False 表示应中断流。"""
    etype = st.usage_tracker.ingest_upstream_event(event)
    if etype in ("prompt_meta", "response_created", "usage"):
        return True
    content = event.get("content", "")
    if etype == "thinking":
        return await _process_openai_stream_thinking(st, content)
    if etype != "answer":
        return True
    return await _process_openai_stream_answer(st, content)


async def _flush_remaining_thinking_and_text(
    st: OpenAIStreamState, final_text: str,
) -> None:
    if st.disconnected[0]:
        return
    await _emit_partial_thinking(st)

    if st.disconnected[0]:
        return
    safe_text = st.parser.partial_text if st.parser.has_calls else (final_text or st.parser.partial_text)
    new_text, st.last_safe_len = advance_partial_buffer(st.last_safe_len, safe_text)
    if new_text:
        chunk = st.stream_chunk(content=new_text)
        await _emit_chunk(st.resp, chunk, st.disconnected)


async def _finalize_openai_stream_tool(st: OpenAIStreamState, all_tool_calls: List[Dict[str, Any]]) -> None:
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
    all_tool_calls = resolve_streamed_tool_calls(parsed_calls, st.streamed_tool_calls)
    st.pending_tc_index = reconcile_pending_tool_index(
        st.pending_tc_index, all_tool_calls, st.stream_tool_blocks_sent,
    )
    return all_tool_calls


async def _run_openai_event_stream(st: OpenAIStreamState, state, messages, model, tools, protocol_options) -> None:
    # 惰性导入：避免 chat_request ↔ openai 包初始化循环
    from handlers.chat_request import iter_retried_chat_events

    async def _make_stream():
        async for event in _chat_once(
            state, messages, model, tools, st.req_id, protocol_options=protocol_options,
            prompt_api="openai",
        ):
            yield event

    with record_raw_response(st.req_id) as raw_recorder:
        async def _on_event(event: Dict[str, Any]) -> bool:
            raw_recorder.ingest_event(event)
            return await _process_openai_stream_event(st, event)

        await iter_retried_chat_events(
            st.req_id, state, _make_stream,
            model=model, messages=messages, tools=tools,
            disconnected=st.disconnected, on_event=_on_event,
        )


async def _write_classified_openai_stream_error(resp, exc: BaseException, disconnected: list) -> None:
    info = log_classified_stream_error(exc, label="OpenAI stream")
    await _write_openai_stream_error(
        resp, info.message, disconnected, error_type=info.kind, code=info.code,
    )


async def _run_openai_stream_guarded(
    st: OpenAIStreamState, state, messages, model, tools, protocol_options, req_id, resp,
) -> Optional[web.StreamResponse]:
    try:
        await _run_openai_event_stream(st, state, messages, model, tools, protocol_options)
    except asyncio.CancelledError:
        logger.info("OpenAI stream cancelled %s", req_id)
        await _safe_write(resp, b"data: [DONE]\n\n", st.disconnected)
        raise
    except Exception as e:
        await _write_classified_openai_stream_error(resp, e, st.disconnected)
        return resp
    return None


async def _finish_openai_stream(st: OpenAIStreamState, resp, model, include_usage) -> web.StreamResponse:
    final_text, parsed_calls = finalize_parser_tool_calls(
        st.parser, warn=logger.warning,
    )
    await _flush_remaining_thinking_and_text(st, final_text)
    await _finalize_openai_stream_tool(st, parsed_calls)
    all_tool_calls = _resolve_all_tool_calls(st, parsed_calls)
    usage = st.usage_tracker.openai_stream_usage()
    emit_usage = include_usage and usage is not None
    await _send_stream_finish(
        resp, model, st.chunk_id, all_tool_calls, st.disconnected,
        already_sent_tc_count=st.pending_tc_index,
        usage=usage if emit_usage else None,
        include_usage=emit_usage,
    )
    return resp


async def _handle_stream(
    request, state, messages, model, req_id, tools, protocol_options=None, *,
    include_usage: bool = False,
):
    try:
        async with tracked_request(state, req_id):
            resp = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                         "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
            await resp.prepare(request)
            st = OpenAIStreamState(
                model=model,
                chunk_id=_gen_chatcmpl_id(),
                resp=resp,
                disconnected=[False],
                include_usage=include_usage,
                parser=FncallStreamParser(protocol=state.protocol, tools=tools, protocol_options=protocol_options),
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
                logger.error("OpenAI stream error (uncaught path) %s: %s", req_id, e, exc_info=True)
                await _write_classified_openai_stream_error(resp, e, st.disconnected)
                return resp
            finally:
                log_qwen_upstream_usage(req_id, st.usage_tracker)
    except QueueFullError as exc:
        return handler_error_response(exc, label="OpenAI stream")


async def openai_chat_handler(request: web.Request) -> web.StreamResponse:
    from handlers.chat_request import (
        log_chat_request,
        new_request_id,
        read_chat_json,
        resolve_chat_model,
    )

    state = get_state()
    if state.is_shutting_down:
        return _error_response(503, "Shutting down", "server_error")
    if state.scheduler.pending >= CONFIG.max_queue_size:
        return _error_response(503, "Busy", "server_error")
    body = await read_chat_json(request, protocol="openai")
    if isinstance(body, web.Response):
        return body
    messages = body.get("messages", [])
    model = resolve_chat_model(state, body.get("model", state.model))
    if isinstance(model, web.Response):
        return model
    stream = body.get("stream", False)
    tools = apply_tool_choice(body.get("tools", []), body.get("tool_choice"))
    if not messages:
        return _error_response(400, "messages is required")
    protocol_options = _build_protocol_options(body)
    log_chat_request(
        protocol="openai",
        messages=messages,
        model=model,
        stream=stream,
        tools=tools,
        protocol_options=protocol_options,
    )
    req_id = new_request_id()
    if not stream:
        return await _handle_non_stream(state, messages, model, req_id, tools, protocol_options)
    return await _handle_stream(
        request, state, messages, model, req_id, tools, protocol_options,
        include_usage=openai_stream_include_usage(body),
    )


async def _handle_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(state, messages, model, req_id, tools, protocol_options))
        return _json_response(result)
    except Exception as e:
        return handler_error_response(e, label="OpenAI non-stream")
