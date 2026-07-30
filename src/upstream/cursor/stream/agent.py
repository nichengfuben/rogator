from __future__ import annotations

import asyncio
import queue
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.worker import stream_worker


async def stream_cursor_agent(
    *,
    prompt: str,
    model: str,
    token: Dict[str, str],
    conversation_id: Optional[str] = None,
    mcp_tools: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    workspace: Optional[str] = None,
    tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    images: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
    custom_system_prompt: Optional[str] = None,
    harness: Optional[Any] = None,
    exclude_workspace_context: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    q: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    def _run() -> None:
        stream_worker(
            q, token, prompt, model, conversation_id, mcp_tools, conversation_history, workspace or "",
            tool_handlers=tool_handlers,
            images=images,
            files=files,
            custom_system_prompt=custom_system_prompt,
            harness=harness,
            exclude_workspace_context=exclude_workspace_context,
        )

    loop.run_in_executor(None, _run)
    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is None:
            break
        yield event
        if event.type in ("done", "error"):
            break
