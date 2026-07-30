from __future__ import annotations

import socket
import threading
import time
from typing import Any, Dict, List, Tuple

import h2.events

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
        with sock_lock:
            safe_send_data(conn, sock, stream_id, encode_frame(obj), sock_lock=sock_lock)

    def finish(elapsed: float, *, usage=None) -> None:
        if usage:
            ctx.q.put(StreamEvent(type="usage", usage=usage, conversation_id=ctx.conv_id, elapsed=elapsed))
        ctx.q.put(StreamEvent(type="done", conversation_id=ctx.conv_id, elapsed=elapsed))

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
            with sock_lock:
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
    sock.setblocking(True)
    sock.settimeout(1.0)
    sock_lock = threading.Lock()
    hb_stop = threading.Event()
    hb_counter = [0]
    _wire_ctx_callbacks(ctx, sock, conn, stream_id, sock_lock)

    hb_thread = threading.Thread(
        target=_run_heartbeat,
        args=(sock, conn, stream_id, sock_lock, hb_stop, heartbeat_interval, hb_counter),
        daemon=True,
    )
    hb_thread.start()
    try:
        done = False
        while not done:
            buffer, done = _poll_agent_stream(sock, conn, buffer, ctx, start, timeout)
    finally:
        hb_stop.set()
        try:
            sock.close()
        except Exception:
            pass
