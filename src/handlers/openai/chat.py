from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.exec.fncall.protocols.entml_think.parse import split_entml_thinking
from echotools.base.logger import get_logger

from core.dispatch import stream_openai_chat
from handlers import EmptyResponseError
from handlers.openai.tools import _parse_tool_calls
from server.formats import (
    UpstreamUsageTracker,
    build_openai_response,
    log_qwen_upstream_usage,
)
from server.records.response_record import record_raw_response
from server.records.sse_record import record_sse_stream
from server.retry import run_with_session_retry

logger = get_logger("rogator")


def _resolve_retry_client(state, model, messages, tools):
    """兼容旧路径；实现位于 chat_request（惰性导入避免循环）。"""
    from handlers.chat_request import resolve_retry_client
    return resolve_retry_client(state, model, messages, tools)


async def _chat_once(
    state,
    messages,
    model,
    tools,
    req_id,
    files=None,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> AsyncGenerator[Dict[str, Any], None]:
    """单次聊天（不含换号重试，由 session_retry 包装）。"""
    async for event in stream_openai_chat(
        state,
        messages,
        model,
        tools,
        req_id,
        protocol_options=protocol_options,
        prompt_api=prompt_api,
        files=files,
    ):
        yield event


async def _collect_non_stream_response(
    state, messages, model, tools, req_id, protocol_options,
    *,
    prompt_api: str = "openai",
) -> Dict[str, Any]:
    response_parts: List[str] = []
    think_parts: List[str] = []
    event_count = 0
    usage_tracker = UpstreamUsageTracker()
    # zen 等不使用 entml 的上游无需落盘 response/sse
    from contextlib import nullcontext
    _skip_record = (protocol_options or {}).get("_upstream_name") in {"zen", "cursor"}
    sse_ctx = nullcontext() if _skip_record else record_sse_stream(req_id)
    resp_ctx = nullcontext() if _skip_record else record_raw_response(req_id)
    with sse_ctx, resp_ctx as raw_recorder:
        async for event in _chat_once(
            state, messages, model, tools, req_id, protocol_options=protocol_options,
            prompt_api=prompt_api,
        ):
            event_count += 1
            # 非流式仅累积上游 usage 快照，不用流式 //4 估算（避免改变响应 usage）
            usage_tracker.ingest_event(event)
            if raw_recorder is not None:
                raw_recorder.ingest_event(event)
            etype = event.get("type")
            if etype in ("response_created", "usage", "prompt_meta"):
                continue
            if etype == "answer":
                response_parts.append(event.get("content", ""))
            elif etype == "thinking":
                think_parts.append(event.get("content", ""))
    if event_count == 0:
        logger.warning("No events received from upstream for req %s", req_id)
        raise EmptyResponseError(f"No events received from upstream for {req_id}")
    full_text = "".join(response_parts)
    reasoning = "".join(think_parts)
    display_text, tool_calls = _parse_tool_calls(state, full_text, tools)
    display_text, entml_thinking = split_entml_thinking(display_text)
    if entml_thinking:
        reasoning = f"{reasoning}\n{entml_thinking}".strip() if reasoning else entml_thinking
    log_qwen_upstream_usage(req_id, usage_tracker)
    return build_openai_response(
        model, display_text, reasoning=reasoning, tool_calls=tool_calls,
        usage=usage_tracker.openai_stream_usage(),
    )


async def _process_openai_non_stream(
    state, messages, model, req_id, tools, protocol_options=None, *,
    prompt_api: str = "openai",
):
    """非流式处理 - 含换号重试"""
    retry_client = _resolve_retry_client(state, model, messages, tools)

    async def _run():
        return await _collect_non_stream_response(
            state, messages, model, tools, req_id, protocol_options,
            prompt_api=prompt_api,
        )

    return await run_with_session_retry(req_id, state, _run, client=retry_client)
