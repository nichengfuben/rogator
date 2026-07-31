from __future__ import annotations

import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import h2.events

from upstream.cursor.chat.agent_session import (
    PARK_TTL_SEC,
    ParkedRun,
    PendingMcp,
    complete_run,
    park_run,
)
from upstream.cursor.stream.handlers import AgentRunContext, process_agent_message
from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.proto import encode_frame, safe_send_data


def _read_frames(buffer: bytes) -> Tuple[bytes, List[Dict[str, Any]]]:
    import json
    import struct

    messages: List[Dict[str, Any]] = []
    while len(buffer) >= 5:
        length = struct.unpack_from(">I", buffer, 1)[0]
        if len(buffer) < 5 + length:
            break
        payload = buffer[5 : 5 + length]
        buffer = buffer[5 + length :]
        try:
            messages.append(json.loads(payload.decode("utf-8")))
        except Exception:
            continue
    return buffer, messages


def _wire_ctx_callbacks(ctx: AgentRunContext, sock, conn, stream_id: int, sock_lock: threading.Lock) -> None:
    def send_frame(obj: Dict[str, Any]) -> None:
        safe_send_data(conn, sock, stream_id, encode_frame(obj), sock_lock=sock_lock)

    def finish(elapsed: float, *, usage=None) -> None:
        if usage:
            ctx.q.put(StreamEvent(type="usage", usage=usage, conversation_id=ctx.conv_id, elapsed=elapsed))
        ctx.q.put(StreamEvent(type="done", conversation_id=ctx.conv_id, elapsed=elapsed))
        ctx._loop_finished = True  # noqa: SLF001

    def touch() -> None:
        ctx.last_activity = time.time()

    ctx.send_frame = send_frame
    ctx.finish = finish
    ctx.touch = touch


def _process_messages(messages: List[Dict[str, Any]], ctx: AgentRunContext) -> bool:
    for msg in messages:
        if process_agent_message(msg, ctx):
            return True
    return False


def _handle_data_event(
    event: h2.events.DataReceived,
    conn,
    buffer: bytes,
    ctx: AgentRunContext,
) -> Tuple[bytes, bool]:
    conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
    buffer += event.data
    buffer, messages = _read_frames(buffer)
    return buffer, _process_messages(messages, ctx)


def _run_heartbeat(
    sock, conn, stream_id: int, sock_lock: threading.Lock,
    stop: threading.Event, interval: float, counter: List[int],
) -> None:
    while not stop.wait(interval):
        try:
            counter[0] += 1
            safe_send_data(
                conn, sock, stream_id,
                encode_frame({"clientHeartbeat": {"id": counter[0]}}),
                sock_lock=sock_lock,
            )
        except Exception:
            break


def _poll_agent_stream(sock, conn, buffer: bytes, ctx: AgentRunContext, start: float, timeout: float) -> Tuple[bytes, bool]:
    if time.time() - start > timeout:
        ctx.q.put(StreamEvent(type="error", error="Request timeout"))
        return buffer, True
    try:
        chunk = sock.recv(65536)
    except socket.timeout:
        return buffer, False
    if not chunk:
        return buffer, True
    done = False
    for event in conn.receive_data(chunk):
        if isinstance(event, h2.events.DataReceived):
            buffer, done = _handle_data_event(event, conn, buffer, ctx)
        elif isinstance(event, h2.events.StreamEnded):
            ctx.finish(time.time() - start)
            done = True
        if done:
            break
    pending = conn.data_to_send()
    if pending:
        sock.sendall(pending)
    return buffer, done


def _enter_park(
    *,
    sock,
    conn,
    stream_id: int,
    sock_lock: threading.Lock,
    ctx: AgentRunContext,
    hb_stop: threading.Event,
) -> ParkedRun:
    pending: Dict[str, PendingMcp] = {}
    for tid, pm in (getattr(ctx, "pending_mcp", None) or {}).items():
        if isinstance(pm, PendingMcp):
            pending[tid] = pm
        elif isinstance(pm, dict):
            pending[tid] = PendingMcp(
                tool_call_id=tid,
                base_msg=dict(pm.get("base_msg") or pm),
                exec_id=int(pm.get("exec_id") or 0),
            )
    run = ParkedRun(
        session_id=ctx.session_id or ctx.conv_id,
        workspace=getattr(ctx, "workspace", "") or "",
        ctx=ctx,
        sock=sock,
        conn=conn,
        stream_id=stream_id,
        sock_lock=sock_lock,
        event_q=ctx.q,
        pending=pending,
        last_checkpoint=dict(getattr(ctx, "last_checkpoint", None) or {}),
        hb_stop=hb_stop,
    )
    ctx.q.put(StreamEvent(
        type="awaiting_tools",
        data={"tool_call_ids": list(pending.keys()), "session_id": run.session_id},
        conversation_id=ctx.conv_id,
        elapsed=time.time() - ctx.start,
    ))
    park_run(run)
    return run


def _park_wait_tick(
    *,
    parked: ParkedRun,
    sock,
    conn,
    buffer: bytes,
    ctx: AgentRunContext,
    start: float,
    timeout: float,
) -> Tuple[bytes, Optional[str]]:
    """一次 park 等待轮询。返回 (buffer, error_or_None)；error 为 'stop'/'ttl'/'timeout'/'closed'/'ended'。"""
    if parked.stop_event.is_set():
        return buffer, "stop"
    if time.time() - parked.parked_at > PARK_TTL_SEC:
        return buffer, "ttl"
    if time.time() - start > timeout:
        return buffer, "timeout"
    try:
        chunk = sock.recv(65536)
    except socket.timeout:
        return buffer, None
    if not chunk:
        return buffer, "closed"
    for event in conn.receive_data(chunk):
        if isinstance(event, h2.events.DataReceived):
            buffer, _ = _handle_data_event(event, conn, buffer, ctx)
        elif isinstance(event, h2.events.StreamEnded):
            return buffer, "ended"
    pending_bytes = conn.data_to_send()
    if pending_bytes:
        sock.sendall(pending_bytes)
    return buffer, None


def _wait_while_parked(
    *,
    parked: ParkedRun,
    sock,
    conn,
    buffer: bytes,
    ctx: AgentRunContext,
    start: float,
    timeout: float,
) -> Tuple[bytes, bool]:
    """等待 resume。返回 (buffer, should_return_from_loop)。"""
    from upstream.cursor.chat.agent_session import drop_parked

    while not parked.resume_event.is_set():
        buffer, err = _park_wait_tick(
            parked=parked, sock=sock, conn=conn, buffer=buffer,
            ctx=ctx, start=start, timeout=timeout,
        )
        if err is None:
            continue
        if err == "stop":
            return buffer, True
        if err == "ttl":
            ctx.q.put(StreamEvent(type="error", error="Parked tool session timed out"))
            drop_parked(parked.session_id, reason="ttl")
            return buffer, True
        if err == "timeout":
            ctx.q.put(StreamEvent(type="error", error="Request timeout while parked"))
            drop_parked(parked.session_id, reason="timeout")
            return buffer, True
        if err == "closed":
            ctx.q.put(StreamEvent(type="error", error="Connection closed while parked"))
            return buffer, True
        if err == "ended":
            ctx.q.put(StreamEvent(type="error", error="Stream ended while parked"))
            return buffer, True
    parked.resume_event.clear()
    ctx.should_park = False
    return buffer, False


def _finish_session_if_needed(
    *,
    sock,
    conn,
    stream_id: int,
    sock_lock: threading.Lock,
    ctx: AgentRunContext,
) -> None:
    if not getattr(ctx, "session_id", ""):
        return
    try:
        complete_run(ParkedRun(
            session_id=ctx.session_id,
            workspace=getattr(ctx, "workspace", "") or "",
            ctx=ctx,
            sock=sock,
            conn=conn,
            stream_id=stream_id,
            sock_lock=sock_lock,
            event_q=ctx.q,
            pending={},
            last_checkpoint=dict(getattr(ctx, "last_checkpoint", None) or {}),
        ))
    except Exception:
        pass


def _start_loop_infra(
    sock,
    conn,
    stream_id: int,
    ctx: AgentRunContext,
    heartbeat_interval: float,
) -> Tuple[threading.Lock, threading.Event, threading.Thread]:
    sock.setblocking(True)
    sock.settimeout(1.0)
    sock_lock = threading.Lock()
    hb_stop = threading.Event()
    hb_counter = [0]
    ctx._loop_finished = False  # noqa: SLF001
    _wire_ctx_callbacks(ctx, sock, conn, stream_id, sock_lock)
    hb_thread = threading.Thread(
        target=_run_heartbeat,
        args=(sock, conn, stream_id, sock_lock, hb_stop, heartbeat_interval, hb_counter),
        daemon=True,
    )
    hb_thread.start()
    return sock_lock, hb_stop, hb_thread


def run_agent_loop(
    sock,
    conn,
    stream_id: int,
    ctx: AgentRunContext,
    *,
    timeout: float,
    heartbeat_interval: float,
) -> None:
    buffer = b""
    start = time.time()
    sock_lock, hb_stop, _hb = _start_loop_infra(sock, conn, stream_id, ctx, heartbeat_interval)
    parked: Optional[ParkedRun] = None
    try:
        done = False
        while not done:
            if getattr(ctx, "should_park", False) and ctx.pending_mcp:
                parked = _enter_park(
                    sock=sock, conn=conn, stream_id=stream_id, sock_lock=sock_lock,
                    ctx=ctx, hb_stop=hb_stop,
                )
                buffer, should_return = _wait_while_parked(
                    parked=parked, sock=sock, conn=conn, buffer=buffer,
                    ctx=ctx, start=start, timeout=timeout,
                )
                if should_return:
                    parked = None
                    return
                parked = None
                continue
            buffer, done = _poll_agent_stream(sock, conn, buffer, ctx, start, timeout)
            if getattr(ctx, "_loop_finished", False):
                done = True
        _finish_session_if_needed(
            sock=sock, conn=conn, stream_id=stream_id, sock_lock=sock_lock, ctx=ctx,
        )
    finally:
        if parked is None:
            hb_stop.set()
            try:
                sock.close()
            except Exception:
                pass
