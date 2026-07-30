from __future__ import annotations

"""Cursor 上游 OpenAI 聊天流（原生 thinking / text / tool_call，无 entml）。"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from echotools.logger import get_logger

from upstream.cursor.stream.agent import stream_cursor_agent
from upstream.cursor.auth.store import get_token_bundle
from upstream.cursor.chat.convert import map_model, openai_tools_to_mcp, split_prompt_and_history

logger = get_logger("rogator")


def _cursor_send_text(prompt: str, splitter) -> str:
    if splitter.send_full_prompt or len(prompt) <= splitter.max_chars:
        return prompt
    return prompt[-splitter.max_chars :]


def _map_cursor_event(event, client) -> Optional[Dict[str, Any]]:
    if event.type == "thinking":
        return {"type": "thinking", "content": event.text, "native": True}
    if event.type == "thinking_done":
        return {"type": "thinking_done", "native": True}
    if event.type == "text":
        return {"type": "answer", "content": event.text, "native": True}
    if event.type == "tool_call" and event.tool_call:
        return {"type": "tool_call", "tool_call": event.tool_call, "native": True}
    if event.type == "tool_started":
        return {"type": "tool_started", "tool_name": event.tool_name, "tool_call_id": event.tool_call_id, "native": True}
    if event.type == "tool_partial":
        return {"type": "tool_partial", "tool_name": event.tool_name, "native": True}
    if event.type == "tool_completed":
        return {"type": "tool_completed", "native": True}
    if event.type == "tool_result" and event.tool_result:
        return {"type": "tool_result", "content": event.tool_result, "native": True}
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
    prompt, history = split_prompt_and_history(messages)
    send_text = _cursor_send_text(prompt, state.splitter)
    yield {"type": "prompt_meta", "prompt_chars": len(send_text), "native": True}

    await client.ensure_token()
    cursor_model = map_model(model)
    mcp_tools = openai_tools_to_mcp(tools) if tools else None
    workspace = getattr(client, "_workspace", None)

    async for event in stream_cursor_agent(
        prompt=send_text,
        model=cursor_model,
        token=get_token_bundle(),
        conversation_id=getattr(client, "_conversation_id", None),
        mcp_tools=mcp_tools,
        conversation_history=history or None,
        workspace=workspace,
        files=files,
    ):
        mapped = _map_cursor_event(event, client)
        if mapped is not None:
            yield mapped
        elif event.type == "done":
            return
