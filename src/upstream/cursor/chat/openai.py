from __future__ import annotations

"""Cursor 上游 OpenAI 聊天流（原生 thinking / text / tool_call，无 entml）。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from echotools.base.logger import get_logger

from server.formats import TokenExpiredError, UpstreamUnavailableError
from upstream.cursor.stream.agent import stream_cursor_agent, stream_parked_agent
from upstream.cursor.auth.store import get_token_bundle
from upstream.cursor.chat import session_db
from upstream.cursor.chat.agent_session import (
    find_completed_for_workspace,
    find_parked_by_tool_ids,
    find_parked_for_workspace,
    resume_with_tool_results,
    trailing_tool_messages,
)
from upstream.cursor.chat.convert import (
    build_cursor_turn,
    map_model,
    openai_tools_to_mcp,
    original_tool_names,
    rewrite_tool_call_for_openai,
)
from upstream.cursor.stream.exec.tool_filter import tool_filter_for_openai

logger = get_logger("rogator")


def _is_cursor_rate_limit(message: str) -> bool:
    lower = (message or "").lower()
    return (
        "error_rate_limited" in lower
        or "rate_limited" in lower
        or "rate limit" in lower
        or "more agent usage" in lower
    )


def _raise_cursor_stream_error(message: str) -> None:
    msg = message or "Cursor upstream error"
    if _is_cursor_rate_limit(msg):
        raise TokenExpiredError(msg)
    raise UpstreamUnavailableError(msg, upstream="cursor")


def _map_tool_started(event, allowed_tool_names: Optional[Set[str]]) -> Optional[Dict[str, Any]]:
    from upstream.cursor.chat.convert import strip_mcp_prefix, _tool_name_match_keys

    name = (event.tool_name or "").strip()
    if not name:
        return None
    if name.startswith("mcp__"):
        if allowed_tool_names is not None:
            if not any(k in allowed_tool_names for k in _tool_name_match_keys(name)):
                return None
        name = strip_mcp_prefix(name)
    return {
        "type": "tool_started",
        "tool_name": name,
        "tool_call_id": event.tool_call_id,
        "native": True,
    }


def _map_tool_call(event, allowed_tool_names: Optional[Set[str]]) -> Optional[Dict[str, Any]]:
    if not event.tool_call:
        return None
    rewritten = rewrite_tool_call_for_openai(
        event.tool_call,
        allowed_originals=allowed_tool_names,
    )
    if not rewritten:
        return None
    return {"type": "tool_call", "tool_call": rewritten, "native": True}


def _map_cursor_event(
    event,
    client,
    *,
    allowed_tool_names: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    et = event.type
    if et == "checkpoint":
        return None
    if et == "awaiting_tools":
        return {"type": "awaiting_tools", "data": event.data or {}, "native": True}
    if et == "thinking":
        return {"type": "thinking", "content": event.text, "native": True}
    if et == "thinking_done":
        return {"type": "thinking_done", "native": True}
    if et == "text":
        return {"type": "answer", "content": event.text, "native": True}
    if et == "tool_call":
        return _map_tool_call(event, allowed_tool_names)
    if et == "tool_started":
        return _map_tool_started(event, allowed_tool_names)
    if et in ("tool_partial", "tool_result"):
        return None
    if et == "tool_completed":
        return {"type": "tool_completed", "native": True}
    if et == "summary" and event.text:
        return {"type": "summary", "content": event.text, "native": True}
    if et == "summary_started":
        return {"type": "summary_started", "native": True}
    if et == "usage" and event.usage:
        return {"type": "usage", "data": event.usage, "native": True}
    if et == "error":
        _raise_cursor_stream_error(event.error or "Cursor upstream error")
    if et == "done":
        if event.conversation_id:
            client._conversation_id = event.conversation_id  # noqa: SLF001
        return None
    return None


def _prepare_cursor_openai(
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    client: Any,
    *,
    reuse_conversation: bool = False,
) -> Tuple[
    str,
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    str,
    Optional[Set[str]],
    Optional[List[Dict[str, Any]]],
    Optional[List[str]],
    Optional[List[str]],
]:
    send_text, history, prepend = build_cursor_turn(messages, tools)
    if not reuse_conversation:
        client._conversation_id = None  # noqa: SLF001
    logger.info(
        "cursor turn req=%s prompt_chars=%d history=%d prepend=%d prompt_head=%r prompt_tail=%r",
        req_id, len(send_text), len(history), len(prepend),
        send_text[:160].replace("\n", "\\n"),
        send_text[-160:].replace("\n", "\\n"),
    )
    allowed_tools, exclude_tools = tool_filter_for_openai(bool(tools))
    return (
        send_text,
        history or None,
        prepend or None,
        map_model(model),
        original_tool_names(tools) if tools else None,
        openai_tools_to_mcp(tools) if tools else None,
        allowed_tools,
        exclude_tools,
    )


def _buffer_tool_call(
    mapped: Dict[str, Any],
    pending: List[Dict[str, Any]],
    seen: Set[str],
) -> bool:
    if mapped.get("type") != "tool_call":
        return False
    tc = mapped.get("tool_call") or {}
    tid = str(tc.get("id") or "").strip()
    if tid and tid in seen:
        return True
    if tid:
        seen.add(tid)
    pending.append(mapped)
    return True


async def _iter_openai_cursor_events(
    agent_events,
    client: Any,
    allowed_names: Optional[Set[str]],
) -> AsyncGenerator[Dict[str, Any], None]:
    pending: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _flush() -> List[Dict[str, Any]]:
        out = list(pending)
        pending.clear()
        return out

    async for event in agent_events:
        if event.type == "awaiting_tools":
            for item in _flush():
                yield item
            return
        mapped = _map_cursor_event(event, client, allowed_tool_names=allowed_names)
        if mapped is None:
            if event.type == "done":
                for item in _flush():
                    yield item
            continue
        if mapped.get("type") == "awaiting_tools":
            for item in _flush():
                yield item
            return
        if _buffer_tool_call(mapped, pending, seen):
            continue
        et = mapped.get("type")
        if et == "answer" and pending:
            for item in _flush():
                yield item
            return
        if et in ("thinking", "thinking_done") and pending:
            continue
        yield mapped
        if event.type == "done":
            for item in _flush():
                yield item
    for item in _flush():
        yield item


async def _resume_or_none(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    client: Any,
    workspace: str,
) -> Tuple[bool, Optional[AsyncGenerator[Dict[str, Any], None]]]:
    """返回 (handled_fully, generator_or_none)。handled_fully 表示勿再开新 Run。"""
    tool_msgs = trailing_tool_messages(messages)
    if not tool_msgs:
        return False, None
    allowed_names = original_tool_names(tools) if tools else None
    tids = [str(m.get("tool_call_id") or "").strip() for m in tool_msgs if m.get("tool_call_id")]
    parked = find_parked_by_tool_ids(tids)
    if parked is None and workspace:
        parked = find_parked_for_workspace(workspace)
        if parked is not None:
            logger.info(
                "cursor turn req=%s park_fallback workspace session=%s tools=%d",
                req_id, parked.session_id[:8], len(tool_msgs),
            )
    if parked is None:
        return False, None
    ok = resume_with_tool_results(parked, tool_msgs, req_id=req_id)
    logger.info(
        "cursor turn req=%s resume_exec ok=%s session=%s tools=%d",
        req_id, ok, parked.session_id[:8], len(tool_msgs),
    )

    async def _gen() -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "prompt_meta", "prompt_chars": 0, "native": True, "resume_exec": True}
        if ok:
            async for item in _iter_openai_cursor_events(
                stream_parked_agent(parked), client, allowed_names,
            ):
                yield item
        else:
            session_db.log_request(
                kind="fallback_text",
                req_id=req_id,
                session_id=parked.session_id,
                payload={"reason": "resume_send_failed"},
            )

    return ok, _gen()


async def _stream_fresh_agent(
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    client: Any,
    workspace: str,
    files: Optional[List[Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    tool_msgs = trailing_tool_messages(messages)
    completed = find_completed_for_workspace(workspace) if not tool_msgs else None
    reuse = bool(completed and completed.get("conversation_id") and completed.get("status") == "done")
    prepared = _prepare_cursor_openai(
        messages, model, tools, req_id, client, reuse_conversation=reuse,
    )
    send_text, history, prepend, cursor_model, allowed_names, mcp_tools, allowed_tools, exclude_tools = prepared
    yield {"type": "prompt_meta", "prompt_chars": len(send_text), "native": True}
    conv_id = conv_state = blob_store = session_id = None
    if reuse and completed:
        conv_id = completed.get("conversation_id") or None
        conv_state = completed.get("checkpoint") or {}
        blob_store = completed.get("blobs") or {}
        session_id = completed.get("session_id") or None
        client._conversation_id = conv_id  # noqa: SLF001
        logger.info(
            "cursor resume_session req=%s session=%s conv=%s blobs=%d",
            req_id, (session_id or "")[:8], (conv_id or "")[:8], len(blob_store or {}),
        )
        session_db.log_request(
            kind="resume_session", req_id=req_id, session_id=session_id or "",
            prompt_head=send_text[:160], history_len=len(history or []),
        )
    elif tool_msgs:
        session_db.log_request(
            kind="fallback_text", req_id=req_id, prompt_head=send_text[:160],
            history_len=len(history or []), payload={"reason": "no_parked_run"},
        )
    await client.ensure_token()
    agent_events = stream_cursor_agent(
        prompt=send_text, model=cursor_model, token=get_token_bundle(),
        conversation_id=conv_id, conversation_group_id=None,
        conversation_state=conv_state, mcp_tools=mcp_tools,
        conversation_history=history, workspace=workspace, files=files,
        prepend_user_messages=prepend, allowed_tools=allowed_tools,
        exclude_tools=exclude_tools, defer_mcp=True, session_id=session_id,
        blob_store=blob_store, req_id=req_id,
    )
    async for item in _iter_openai_cursor_events(agent_events, client, allowed_names):
        yield item


async def stream_openai_chat(
    state: Any,
    client: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    _ = protocol_options, prompt_api, state
    workspace = getattr(client, "_workspace", None) or ""
    handled, gen = await _resume_or_none(messages, tools, req_id, client, workspace)
    if gen is not None:
        async for item in gen:
            yield item
        if handled:
            return
    async for item in _stream_fresh_agent(
        messages, model, tools, req_id, client, workspace, files,
    ):
        yield item
