from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from echotools.exec.fncall.protocols.entml_think.parse import split_entml_thinking
from echotools.logger import get_logger

from core.dispatch import resolve_upstream, stream_openai_chat
from handlers import EmptyResponseError
from handlers.openai.tools import _parse_tool_calls
from server.formats import (
    UpstreamUsageTracker,
    build_openai_response,
    log_qwen_upstream_usage,
)
from server.model.model_registry import ModelRegistryEntry, is_native_upstream_event
from server.records.response_record import record_raw_response
from server.retry import run_with_session_retry, stream_with_session_retry

logger = get_logger("rogator")


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


def _ingest_non_stream_event(
    event: Dict[str, Any],
    registry_entry: Optional[ModelRegistryEntry],
    response_parts: List[str],
    think_parts: List[str],
    tool_calls_native: List[Dict[str, Any]],
) -> None:
    if event.get("type") in ("response_created", "usage", "prompt_meta"):
        return
    if is_native_upstream_event(registry_entry, event):
        etype = event.get("type")
        if etype == "answer":
            response_parts.append(event.get("content", ""))
        elif etype == "thinking":
            think_parts.append(event.get("content", ""))
        elif etype == "tool_call" and event.get("tool_call"):
            tool_calls_native.append(event["tool_call"])
        return
    if event.get("type") == "answer":
        response_parts.append(event.get("content", ""))
    elif event.get("type") == "thinking":
        think_parts.append(event.get("content", ""))


async def _collect_non_stream_response(
    state, messages, model, tools, req_id, protocol_options,
    *,
    registry_entry: Optional[ModelRegistryEntry] = None,
) -> Dict[str, Any]:
    response_parts: List[str] = []
    think_parts: List[str] = []
    event_count = 0
    usage_tracker = UpstreamUsageTracker()
    tool_calls_native: List[Dict[str, Any]] = []
    native_tools = registry_entry is not None and not registry_entry.uses_entml_tools
    with record_raw_response(req_id) as raw_recorder:
        async for event in _chat_once(
            state, messages, model, tools, req_id, protocol_options=protocol_options,
            prompt_api="openai",
        ):
            event_count += 1
            usage_tracker.ingest_event(event)
            raw_recorder.ingest_event(event)
            _ingest_non_stream_event(
                event, registry_entry, response_parts, think_parts, tool_calls_native,
            )
    if event_count == 0:
        logger.warning("No events received from upstream for req %s", req_id)
        raise EmptyResponseError(f"No events received from upstream for {req_id}")
    full_text = "".join(response_parts)
    reasoning = "".join(think_parts)
    if native_tools or tool_calls_native:
        from server.formats import _fix_tool_call_id
        log_qwen_upstream_usage(req_id, usage_tracker)
        return build_openai_response(
            model, full_text, reasoning=reasoning,
            tool_calls=[_fix_tool_call_id(tc) for tc in tool_calls_native],
            usage=usage_tracker.openai_stream_usage(),
        )
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
    state, messages, model, req_id, tools, protocol_options=None, *, registry_entry=None,
):
    """非流式处理 - 含换号重试"""
    retry_client = _resolve_retry_client(state, model, messages, tools)

    async def _run():
        return await _collect_non_stream_response(
            state, messages, model, tools, req_id, protocol_options,
            registry_entry=registry_entry,
        )

    return await run_with_session_retry(req_id, state, _run, client=retry_client)


def _resolve_retry_client(state, model, messages, tools):
    _, client = resolve_upstream(state, model, messages, tools)
    return client
