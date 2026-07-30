from __future__ import annotations

"""Cursor 上游 OpenAI 聊天流（原生 thinking / text / tool_call，无 entml）。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from echotools.logger import get_logger

from upstream.cursor.stream.agent import stream_cursor_agent
from upstream.cursor.auth.store import get_token_bundle
from upstream.cursor.chat.convert import (
    build_cursor_turn,
    map_model,
    openai_tools_to_mcp,
    original_tool_names,
    rewrite_tool_call_for_openai,
)
from upstream.cursor.stream.exec.tool_filter import tool_filter_for_openai

logger = get_logger("rogator")


def _map_tool_started(event, allowed_tool_names: Optional[Set[str]]) -> Optional[Dict[str, Any]]:
    name = (event.tool_name or "").strip()
    if not name:
        return None
    if allowed_tool_names is not None and name not in allowed_tool_names:
        return None
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
        raise RuntimeError(event.error or "Cursor upstream error")
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
    logger.info(
        "cursor turn req=%s prompt_chars=%d history=%d prepend=%d prompt_head=%r prompt_tail=%r",
        req_id, len(send_text), len(history), len(prepend),
        send_text[:160].replace("\n", "\\n"),
        send_text[-160:].replace("\n", "\\n"),
    )
    allowed_tools, exclude_tools = tool_filter_for_openai(bool(tools))
    client._conversation_id = None  # noqa: SLF001
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
    """若为 tool_call 则缓冲并返回 True。"""
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
        client._conversation_id = None  # noqa: SLF001
        return out

    async for event in agent_events:
        mapped = _map_cursor_event(event, client, allowed_tool_names=allowed_names)
        if mapped is None:
            if event.type == "done":
                for item in _flush():
                    yield item
                return
            continue
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
            return
    for item in _flush():
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
    prepared = _prepare_cursor_openai(messages, model, tools, req_id, client)
    send_text, history, prepend, cursor_model, allowed_names, mcp_tools, allowed_tools, exclude_tools = prepared
    yield {"type": "prompt_meta", "prompt_chars": len(send_text), "native": True}
    await client.ensure_token()
    workspace = getattr(client, "_workspace", None) or ""
    agent_events = stream_cursor_agent(
        prompt=send_text, model=cursor_model, token=get_token_bundle(),
        conversation_id=None, mcp_tools=mcp_tools, conversation_history=history,
        workspace=workspace, files=files, prepend_user_messages=prepend,
        allowed_tools=allowed_tools, exclude_tools=exclude_tools, defer_mcp=True,
    )
    async for item in _iter_openai_cursor_events(agent_events, client, allowed_names):
        yield item
