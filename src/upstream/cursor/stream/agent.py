from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from upstream.cursor.chat.agent_session import ParkedRun
from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.worker import stream_worker


def _start_stream_worker(
    q: queue.Queue,
    *,
    prompt: str,
    model: str,
    token: Dict[str, str],
    conversation_id: Optional[str],
    conversation_group_id: Optional[str],
    conversation_state: Optional[Dict[str, Any]],
    mcp_tools: Optional[List[Dict[str, Any]]],
    conversation_history: Optional[List[Dict[str, Any]]],
    workspace: str,
    tool_handlers: Optional[Dict[str, Callable[..., Any]]],
    images: Optional[List[Any]],
    files: Optional[List[Any]],
    custom_system_prompt: Optional[str],
    prepend_user_messages: Optional[List[Dict[str, Any]]],
    harness: Optional[Any],
    exclude_workspace_context: bool,
    allowed_tools: Optional[List[str]],
    exclude_tools: Optional[List[str]],
    defer_mcp: bool,
    session_id: Optional[str] = None,
    blob_store: Optional[Dict[str, str]] = None,
    req_id: str = "",
) -> threading.Thread:
    def _run() -> None:
        stream_worker(
            q, token, prompt, model, conversation_id, mcp_tools, conversation_history, workspace,
            tool_handlers=tool_handlers, images=images, files=files,
            custom_system_prompt=custom_system_prompt,
            prepend_user_messages=prepend_user_messages, harness=harness,
            exclude_workspace_context=exclude_workspace_context,
            allowed_tools=allowed_tools, exclude_tools=exclude_tools, defer_mcp=defer_mcp,
            conversation_group_id=conversation_group_id,
            conversation_state=conversation_state,
            session_id=session_id,
            blob_store=blob_store,
            req_id=req_id,
        )

    worker = threading.Thread(target=_run, name="cursor-stream-worker", daemon=True)
    worker.start()
    return worker


async def _drain_queue(q: queue.Queue) -> AsyncGenerator[StreamEvent, None]:
    """排空 worker 队列。

    ``awaiting_tools``：停车等客户端回灌，立刻结束给上层。
    ``done``/``error``：继续读到 ``None``，确保 worker 跑完 ``complete_run`` 再返回。
    """
    while True:
        try:
            event = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.02)
            continue
        if event is None:
            break
        yield event
        if event.type == "awaiting_tools":
            break


async def stream_cursor_agent(
    *,
    prompt: str,
    model: str,
    token: Dict[str, str],
    conversation_id: Optional[str] = None,
    conversation_group_id: Optional[str] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
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
    session_id: Optional[str] = None,
    blob_store: Optional[Dict[str, str]] = None,
    req_id: str = "",
) -> AsyncGenerator[StreamEvent, None]:
    q: queue.Queue = queue.Queue()
    _start_stream_worker(
        q, prompt=prompt, model=model, token=token, conversation_id=conversation_id,
        conversation_group_id=conversation_group_id,
        conversation_state=conversation_state,
        mcp_tools=mcp_tools, conversation_history=conversation_history,
        workspace=workspace or "", tool_handlers=tool_handlers, images=images, files=files,
        custom_system_prompt=custom_system_prompt, prepend_user_messages=prepend_user_messages,
        harness=harness, exclude_workspace_context=exclude_workspace_context,
        allowed_tools=allowed_tools, exclude_tools=exclude_tools, defer_mcp=defer_mcp,
        session_id=session_id, blob_store=blob_store, req_id=req_id,
    )
    async for event in _drain_queue(q):
        yield event


async def stream_parked_agent(run: ParkedRun) -> AsyncGenerator[StreamEvent, None]:
    """续读已挂起 Run 的事件队列（resume_with_tool_results 之后调用）。"""
    async for event in _drain_queue(run.event_q):
        yield event
