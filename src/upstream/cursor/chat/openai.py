from __future__ import annotations

"""Cursor 上游 OpenAI 聊天流（原生 thinking / text / tool_call，无 entml）。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Set

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
from upstream.cursor.stream.tool_filter import tool_filter_for_openai

logger = get_logger("rogator")


def _map_cursor_event(
    event,
    client,
    *,
    allowed_tool_names: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    if event.type == "thinking":
        return {"type": "thinking", "content": event.text, "native": True}
    if event.type == "thinking_done":
        return {"type": "thinking_done", "native": True}
    if event.type == "text":
        return {"type": "answer", "content": event.text, "native": True}
    if event.type == "tool_call" and event.tool_call:
        rewritten = rewrite_tool_call_for_openai(
            event.tool_call,
            allowed_originals=allowed_tool_names,
        )
        if not rewritten:
            return None
        return {"type": "tool_call", "tool_call": rewritten, "native": True}
    if event.type == "tool_started":
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
    if event.type == "tool_partial":
        return None
    if event.type == "tool_completed":
        return {"type": "tool_completed", "native": True}
    if event.type == "tool_result":
        # OpenAI 代理模式下 tool result 由客户端回灌；本地 stub 不转发
        return None
    if event.type == "summary" and event.text:
        return {"type": "summary", "content": event.text, "native": True}
    if event.type == "summary_started":
        return {"type": "summary_started", "native": True}
    if event.type == "usage" and event.usage:
        return {"type": "usage", "data": event.usage, "native": True}
    if event.type == "error":
        raise RuntimeError(event.error or "Cursor upstream error")
    if event.type == "done":
        if event.conversation_id:
            client._conversation_id = event.conversation_id  # noqa: SLF001
        return None
    return None


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
    _ = protocol_options, prompt_api
    send_text, history, prepend = build_cursor_turn(messages, tools)
    logger.info(
        "cursor turn req=%s prompt_chars=%d history=%d prepend=%d prompt_head=%r prompt_tail=%r",
        req_id,
        len(send_text),
        len(history),
        len(prepend),
        send_text[:160].replace("\n", "\\n"),
        send_text[-160:].replace("\n", "\\n"),
    )
    yield {"type": "prompt_meta", "prompt_chars": len(send_text), "native": True}

    await client.ensure_token()
    cursor_model = map_model(model)
    mcp_tools = openai_tools_to_mcp(tools) if tools else None
    allowed_names = original_tool_names(tools) if tools else None
    workspace = getattr(client, "_workspace", None) or ""
    allowed_tools, exclude_tools = tool_filter_for_openai(bool(tools))
    # OpenAI 代理每轮用 messages 重建 history；复用 conversation_id 会与
    # defer_mcp 空回执的服务端状态冲突，故始终开新会话。
    client._conversation_id = None  # noqa: SLF001

    pending_tool_calls: List[Dict[str, Any]] = []
    seen_tool_ids: Set[str] = set()

    def _flush_tools() -> List[Dict[str, Any]]:
        out = list(pending_tool_calls)
        pending_tool_calls.clear()
        client._conversation_id = None  # noqa: SLF001
        return out

    async for event in stream_cursor_agent(
        prompt=send_text,
        model=cursor_model,
        token=get_token_bundle(),
        conversation_id=None,
        mcp_tools=mcp_tools,
        conversation_history=history or None,
        workspace=workspace,
        files=files,
        # 勿传 customSystemPrompt（agentn → unknown option '--system-prompt'）
        # IMPORTANT+system 走官方 prependUserMessages
        prepend_user_messages=prepend or None,
        allowed_tools=allowed_tools,
        exclude_tools=exclude_tools,
        defer_mcp=True,
    ):
        mapped = _map_cursor_event(event, client, allowed_tool_names=allowed_names)
        if mapped is None:
            if event.type == "done":
                for item in _flush_tools():
                    yield item
                return
            continue

        et = mapped.get("type")
        if et == "tool_call":
            tc = mapped.get("tool_call") or {}
            tid = str(tc.get("id") or "").strip()
            if tid and tid in seen_tool_ids:
                continue
            if tid:
                seen_tool_ids.add(tid)
            pending_tool_calls.append(mapped)
            continue

        if et == "answer" and pending_tool_calls:
            # OpenAI 语义：本轮以 tool_calls 结束，丢弃工具后的正文
            for item in _flush_tools():
                yield item
            return

        if et in ("thinking", "thinking_done") and pending_tool_calls:
            # 工具已出齐后的尾部 thinking 忽略
            continue

        yield mapped

        if event.type == "done":
            for item in _flush_tools():
                yield item
            return

    for item in _flush_tools():
        yield item
