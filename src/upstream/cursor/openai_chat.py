from __future__ import annotations

"""Cursor 上游 OpenAI 聊天流（原生 thinking / text / tool_call，无 entml）。"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from echotools.logger import get_logger

from upstream.cursor.agent_stream import stream_cursor_agent
from upstream.cursor.auth_store import get_token_bundle
from upstream.cursor.converter import map_model, openai_tools_to_mcp, split_prompt_and_history

logger = get_logger("rogator")


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
    if files:
        logger.debug("Cursor: ignoring %d uploaded file(s) for req %s", len(files), req_id)

    prompt, history = split_prompt_and_history(messages)
    if state.splitter.send_full_prompt or len(prompt) <= state.splitter.max_chars:
        send_text = prompt
    else:
        send_text = prompt[-state.splitter.max_chars :]

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
    ):
        if event.type == "thinking":
            yield {"type": "thinking", "content": event.text, "native": True}
        elif event.type == "text":
            yield {"type": "answer", "content": event.text, "native": True}
        elif event.type == "tool_call" and event.tool_call:
            yield {"type": "tool_call", "tool_call": event.tool_call, "native": True}
        elif event.type == "usage" and event.usage:
            yield {"type": "usage", "data": event.usage, "native": True}
        elif event.type == "error":
            raise RuntimeError(event.error or "Cursor upstream error")
        elif event.type == "done":
            if event.conversation_id:
                client._conversation_id = event.conversation_id  # noqa: SLF001
            return
