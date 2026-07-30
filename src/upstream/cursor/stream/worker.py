from __future__ import annotations

import queue
import socket
import ssl
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import h2.connection
    import h2.config
    import h2.events
except ImportError as exc:
    raise ImportError("h2 required for cursor upstream: pip install h2") from exc

from upstream.cursor.stream.handlers import AgentRunContext
from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.loop import run_agent_loop
from upstream.cursor.stream.proto import (
    agent_config,
    agent_host,
    build_selected_context,
    encode_frame,
    safe_send_data,
)


def _open_h2_socket(host: str):
    ctx = ssl.create_default_context()
    ctx.set_alpn_protocols(["h2"])
    raw_sock = socket.create_connection((host, 443), timeout=15)
    sock = ctx.wrap_socket(raw_sock, server_hostname=host)
    if sock.selected_alpn_protocol() != "h2":
        raise RuntimeError("ALPN h2 not negotiated")
    conn = h2.connection.H2Connection(
        config=h2.config.H2Configuration(client_side=True, header_encoding="utf-8"),
    )
    conn.initiate_connection()
    sock.sendall(conn.data_to_send())
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        events = conn.receive_data(chunk)
        sock.sendall(conn.data_to_send())
        if any(isinstance(e, h2.events.SettingsAcknowledged) for e in events):
            break
    return sock, conn


def _build_run_request(
    *,
    prompt: str,
    model: str,
    conv_id: str,
    msg_id: str,
    group_id: str,
    workspace: str,
    mcp_tools: Optional[List[Dict[str, Any]]],
    conversation_history: Optional[List[Dict[str, Any]]],
    images: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
    custom_system_prompt: Optional[str] = None,
    harness: Optional[Any] = None,
    exclude_workspace_context: bool = False,
) -> Dict[str, Any]:
    user_message: Dict[str, Any] = {"text": prompt, "messageId": msg_id, "mode": 1}
    selected = build_selected_context(images, files)
    if selected:
        user_message["selectedContext"] = selected
    user_action: Dict[str, Any] = {
        "userMessage": user_message,
        "requestContext": {"workspacePath": workspace},
    }
    if conversation_history:
        user_action["conversationHistory"] = {"messages": conversation_history}
    run_request: Dict[str, Any] = {
        "conversationState": {},
        "action": {"userMessageAction": user_action},
        "modelDetails": {"modelId": model},
        "requestedModel": {"modelId": model, "builtInModel": True},
        "conversationId": conv_id,
        "conversationGroupId": group_id,
        "suggestNextPrompt": False,
    }
    if custom_system_prompt:
        run_request["customSystemPrompt"] = custom_system_prompt
    if harness:
        run_request["harness"] = harness
    if exclude_workspace_context:
        run_request["excludeWorkspaceContext"] = True
    payload: Dict[str, Any] = {"runRequest": run_request}
    if mcp_tools:
        run_request["mcpTools"] = {"mcpTools": mcp_tools}
    return payload


def _agent_headers(
    host: str,
    token: Dict[str, str],
    client_version: str,
    timezone: str,
    session_id: str,
    request_id: str,
) -> List[Tuple[str, str]]:
    _ = timezone
    # 对齐 cursor_mvp：Agent Run 请求头不带 checksum / timezone。
    return [
        (":method", "POST"),
        (":path", "/agent.v1.AgentService/Run"),
        (":scheme", "https"),
        (":authority", host),
        ("content-type", "application/connect+json"),
        ("connect-protocol-version", "1"),
        ("authorization", f"Bearer {token['accessToken']}"),
        ("te", "trailers"),
        ("x-cursor-client-version", client_version),
        ("x-cursor-platform", "cli"),
        ("x-cursor-session-id", session_id),
        ("x-request-id", request_id),
    ]


def _run_agent_stream(
    q: queue.Queue,
    token: Dict[str, str],
    prompt: str,
    model: str,
    conversation_id: Optional[str],
    mcp_tools: Optional[List[Dict[str, Any]]],
    conversation_history: Optional[List[Dict[str, Any]]],
    workspace: str,
    *,
    tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    images: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
    custom_system_prompt: Optional[str] = None,
    harness: Optional[Any] = None,
    exclude_workspace_context: bool = False,
) -> None:
    cfg = agent_config()
    host = agent_host()
    timeout = int(cfg.get("request_timeout") or 300)
    heartbeat_interval = float(cfg.get("heartbeat_interval") or 5)
    client_version = str(cfg.get("client_version") or "cli-2026.07.23-e383d2b")
    timezone = str(cfg.get("timezone") or "Asia/Shanghai")
    conv_id = conversation_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    sock, conn = _open_h2_socket(host)
    stream_id = conn.get_next_available_stream_id()
    conn.send_headers(
        stream_id,
        _agent_headers(host, token, client_version, timezone, session_id, request_id),
        end_stream=False,
    )
    sock.sendall(conn.data_to_send())
    run_request = _build_run_request(
        prompt=prompt, model=model, conv_id=conv_id, msg_id=str(uuid.uuid4()),
        group_id=str(uuid.uuid4()), workspace=workspace, mcp_tools=mcp_tools,
        conversation_history=conversation_history, images=images, files=files,
        custom_system_prompt=custom_system_prompt, harness=harness,
        exclude_workspace_context=exclude_workspace_context,
    )
    safe_send_data(conn, sock, stream_id, encode_frame(run_request))
    ctx = AgentRunContext(
        q=q, conv_id=conv_id, start=time.time(), tool_handlers=tool_handlers or {},
        send_frame=lambda _obj: None, finish=lambda *_a, **_k: None, touch=lambda: None,
    )
    run_agent_loop(sock, conn, stream_id, ctx, timeout=float(timeout), heartbeat_interval=heartbeat_interval)


def stream_worker(
    q: queue.Queue,
    token: Dict[str, str],
    prompt: str,
    model: str,
    conversation_id: Optional[str],
    mcp_tools: Optional[List[Dict[str, Any]]],
    conversation_history: Optional[List[Dict[str, Any]]],
    workspace: str,
    *,
    tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    images: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
    custom_system_prompt: Optional[str] = None,
    harness: Optional[Any] = None,
    exclude_workspace_context: bool = False,
) -> None:
    try:
        _run_agent_stream(
            q, token, prompt, model, conversation_id, mcp_tools, conversation_history, workspace,
            tool_handlers=tool_handlers, images=images, files=files,
            custom_system_prompt=custom_system_prompt, harness=harness,
            exclude_workspace_context=exclude_workspace_context,
        )
    except Exception as exc:
        q.put(StreamEvent(type="error", error=str(exc)))
    q.put(None)
