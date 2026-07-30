from __future__ import annotations

import asyncio
import queue
import threading
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
    prepend_user_messages: Optional[List[Dict[str, Any]]] = None,
    harness: Optional[Any] = None,
    exclude_workspace_context: bool = False,
    allowed_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    defer_mcp: bool = False,
) -> AsyncGenerator[StreamEvent, None]:
    q: queue.Queue = queue.Queue()

    def _run() -> None:
        stream_worker(
            q, token, prompt, model, conversation_id, mcp_tools, conversation_history, workspace or "",
            tool_handlers=tool_handlers,
            images=images,
            files=files,
            custom_system_prompt=custom_system_prompt,
            prepend_user_messages=prepend_user_messages,
            harness=harness,
            exclude_workspace_context=exclude_workspace_context,
            allowed_tools=allowed_tools,
            exclude_tools=exclude_tools,
            defer_mcp=defer_mcp,
        )

    worker = threading.Thread(
        target=_run,
        name="cursor-stream-worker",
        daemon=True,
    )
    worker.start()
    while True:
        try:
            event = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.02)
            continue
        if event is None:
            break
        yield event
        if event.type in ("done", "error"):
            break
