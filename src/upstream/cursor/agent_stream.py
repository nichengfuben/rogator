from __future__ import annotations

"""Cursor Agent HTTP/2 双向流。"""

import asyncio
import base64
import json
import platform
import queue
import socket
import ssl
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import h2.connection
    import h2.config
    import h2.events
except ImportError as exc:
    raise ImportError("h2 required for cursor upstream: pip install h2") from exc

from upstream.cursor.config import cursor_agent_config


@dataclass
class StreamEvent:
    type: str
    text: str = ""
    error: str = ""
    tool_call: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    elapsed: float = 0.0


def _cfg() -> Dict[str, Any]:
    return cursor_agent_config()


def _host() -> str:
    base = str(_cfg().get("base_url") or "https://api2.cursor.sh")
    return base.replace("https://", "").replace("http://", "").rstrip("/")


def generate_checksum(machine_id: str, mac_machine_id: str) -> str:
    k = 165
    t = int(time.time() * 1000) // 1_000_000
    b = bytearray([
        (t >> 40) & 255, (t >> 32) & 255, (t >> 24) & 255,
        (t >> 16) & 255, (t >> 8) & 255, t & 255,
    ])
    for i, range_len in enumerate(range(len(b))):
        b[i] = ((b[i] ^ k) + (range_len % 256)) & 0xFF
        k = b[i]
    prefix = base64.b64encode(bytes(b)).decode()
    if mac_machine_id:
        return f"{prefix}{machine_id}/{mac_machine_id}"
    return f"{prefix}{machine_id}"


def encode_frame(obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return bytes([0]) + struct.pack(">I", len(payload)) + payload


def build_exec_reply(exec_msg: Dict[str, Any]) -> Dict[str, Any]:
    exec_id = exec_msg.get("id", 0)
    exec_uuid = exec_msg.get("execId", "")
    base = {"id": exec_id, "execId": exec_uuid}
    shell = "powershell" if platform.system().lower() == "windows" else "bash"
    os_name = "windows" if platform.system().lower() == "windows" else platform.system().lower()

    if "requestContextArgs" in exec_msg:
        return {"execClientMessage": {**base, "requestContextResult": {"success": {"requestContext": {
            "env": {"operatingSystem": os_name, "defaultShell": shell},
        }}}}}
    if "readArgs" in exec_msg:
        return {"execClientMessage": {**base, "readResult": {"fileNotFound": {}}}}
    if "lsArgs" in exec_msg:
        return {"execClientMessage": {**base, "lsResult": {"error": {"path": "", "error": "Headless"}}}}
    if "shellStreamArgs" in exec_msg:
        return {"execClientMessage": {**base, "shellStreamResult": {"rejected": {"reason": "Headless mode"}}}}
    if "shellArgs" in exec_msg:
        return {"execClientMessage": {**base, "shellResult": {"rejected": {"reason": "Headless mode"}}}}
    if "grepArgs" in exec_msg:
        return {"execClientMessage": {**base, "grepResult": {"error": {"error": "Headless"}}}}
    if "writeArgs" in exec_msg:
        return {"execClientMessage": {**base, "writeResult": {}}}
    if "deleteArgs" in exec_msg:
        return {"execClientMessage": {**base, "deleteResult": {"error": {"path": "", "error": "Headless"}}}}
    if "diagnosticsArgs" in exec_msg:
        return {"execClientMessage": {**base, "diagnosticsResult": {"diagnostics": []}}}
    if "mcpArgs" in exec_msg:
        args = exec_msg.get("mcpArgs") or {}
        name = args.get("toolName") or args.get("name") or "unknown"
        return {"execClientMessage": {**base, "mcpResult": {
            "success": {"content": [{"text": {"text": f"Tool {name} deferred to client"}}], "isError": False},
        }}}
    return {"execClientMessage": {**base, "requestContextResult": {"error": {"error": "Unknown exec type"}}}}


def build_interaction_reply(iq: Dict[str, Any]) -> Dict[str, Any]:
    iq_id = iq.get("id", 0)
    response: Dict[str, Any] = {"interactionResponse": {"id": iq_id}}
    if "askQuestionInteractionQuery" in iq:
        response["interactionResponse"]["askQuestionInteractionResponse"] = {}
    elif "webSearchRequestQuery" in iq:
        response["interactionResponse"]["webSearchRequestResponse"] = {
            "approved": False,
            "reason": "Auto-rejected",
        }
    elif "switchModeRequestQuery" in iq:
        response["interactionResponse"]["switchModeRequestResponse"] = {}
    elif "createPlanRequestQuery" in iq:
        response["interactionResponse"]["createPlanRequestResponse"] = {}
    else:
        response["interactionResponse"]["askQuestionInteractionResponse"] = {}
    return response


def _text_delta(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("delta") or "")
    return ""


def _parse_error(msg: Dict[str, Any]) -> str:
    err = msg.get("error") or {}
    detail = ""
    for det in err.get("details") or []:
        val = det.get("value")
        if val:
            try:
                detail = base64.b64decode(val).decode("utf-8", errors="replace")
            except Exception:
                detail = str(val)
            break
        dg = det.get("debug") or {}
        if dg.get("error"):
            detail = f"{dg['error']}: {(dg.get('details') or {}).get('detail', '')}"
            break
    return detail or err.get("message") or "Unknown error"


def _stream_worker(
    q: queue.Queue,
    token: Dict[str, str],
    prompt: str,
    model: str,
    conversation_id: Optional[str],
    mcp_tools: Optional[List[Dict[str, Any]]],
    conversation_history: Optional[List[Dict[str, Any]]],
    workspace: str,
) -> None:
    cfg = _cfg()
    host = _host()
    timeout = int(cfg.get("request_timeout") or 120)
    heartbeat_interval = float(cfg.get("heartbeat_interval") or 5)
    client_version = str(cfg.get("client_version") or "3.12.17")
    timezone = str(cfg.get("timezone") or "Asia/Shanghai")

    conv_id = conversation_id or str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())

    try:
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(["h2"])
        raw_sock = socket.create_connection((host, 443), timeout=15)
        sock = ctx.wrap_socket(raw_sock, server_hostname=host)
        if sock.selected_alpn_protocol() != "h2":
            q.put(StreamEvent(type="error", error="ALPN h2 not negotiated"))
            q.put(None)
            return

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

        headers = [
            (":method", "POST"),
            (":path", "/agent.v1.AgentService/Run"),
            (":scheme", "https"),
            (":authority", host),
            ("content-type", "application/connect+json"),
            ("connect-protocol-version", "1"),
            ("authorization", f"Bearer {token['accessToken']}"),
            ("x-cursor-checksum", generate_checksum(token["machineId"], token["macMachineId"])),
            ("x-cursor-client-version", client_version),
            ("x-cursor-timezone", timezone),
            ("x-request-id", str(uuid.uuid4())),
        ]
        stream_id = conn.get_next_available_stream_id()
        conn.send_headers(stream_id, headers, end_stream=False)

        user_action: Dict[str, Any] = {
            "userMessage": {"text": prompt, "messageId": msg_id, "mode": 1},
            "requestContext": {"workspacePath": workspace},
        }
        if conversation_history:
            user_action["conversationHistory"] = {"messages": conversation_history}

        run_request: Dict[str, Any] = {
            "runRequest": {
                "conversationState": {},
                "action": {"userMessageAction": user_action},
                "modelDetails": {
                    "modelId": model,
                    "displayName": model,
                    "displayNameShort": model,
                },
                "requestedModel": {"modelId": model, "builtInModel": True},
                "conversationId": conv_id,
                "conversationGroupId": group_id,
                "suggestNextPrompt": False,
                "workspacePath": workspace,
            }
        }
        if mcp_tools:
            run_request["runRequest"]["mcpTools"] = {"mcpTools": mcp_tools}

        conn.send_data(stream_id, encode_frame(run_request))
        sock.sendall(conn.data_to_send())

        buffer = b""
        start = time.time()
        last_activity = start
        heartbeat_count = 0
        text_received = False
        tool_completed_time: Optional[float] = None
        done = False
        sock.setblocking(True)
        sock.settimeout(1.0)
        sock_lock = threading.Lock()
        hb_stop = threading.Event()

        def send_frame(obj: Dict[str, Any]) -> None:
            with sock_lock:
                conn.send_data(stream_id, encode_frame(obj))
                sock.sendall(conn.data_to_send())

        def heartbeat_loop() -> None:
            while not hb_stop.wait(heartbeat_interval):
                try:
                    send_frame({"clientHeartbeat": {}})
                except Exception:
                    break

        hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        hb_thread.start()

        def finish(elapsed: float, *, usage: Optional[Dict[str, Any]] = None) -> None:
            nonlocal done
            if usage:
                q.put(StreamEvent(type="usage", usage=usage, conversation_id=conv_id, elapsed=elapsed))
            q.put(StreamEvent(type="done", conversation_id=conv_id, elapsed=elapsed))
            done = True

        def touch() -> None:
            nonlocal last_activity
            last_activity = time.time()

        try:
            while not done:
                if time.time() - start > timeout:
                    q.put(StreamEvent(type="error", error="Request timeout"))
                    break

                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break

                events = conn.receive_data(chunk)
                for event in events:
                    if isinstance(event, h2.events.DataReceived):
                        conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                        buffer += event.data
                        while True:
                            if len(buffer) < 5:
                                break
                            length = struct.unpack_from(">I", buffer, 1)[0]
                            if len(buffer) < 5 + length:
                                break
                            payload = buffer[5 : 5 + length]
                            buffer = buffer[5 + length :]
                            try:
                                msg = json.loads(payload.decode("utf-8"))
                            except Exception:
                                continue

                            elapsed = time.time() - start

                            if "error" in msg:
                                q.put(StreamEvent(type="error", error=_parse_error(msg), elapsed=elapsed))
                                done = True
                                break

                            if "execServerControlMessage" in msg:
                                escm = msg["execServerControlMessage"]
                                if "abort" in escm:
                                    q.put(StreamEvent(type="error", error="Server abort", elapsed=elapsed))
                                    done = True
                                    break
                                touch()
                                continue

                            if "interactionQuery" in msg:
                                send_frame(build_interaction_reply(msg["interactionQuery"]))
                                touch()
                                continue

                            if "execServerMessage" in msg:
                                exec_msg = msg["execServerMessage"]
                                if "mcpArgs" in exec_msg:
                                    mcp = exec_msg.get("mcpArgs") or {}
                                    name = mcp.get("toolName") or mcp.get("name") or ""
                                    args = mcp.get("args") or {}
                                    tc_id = str(exec_msg.get("execId") or exec_msg.get("id") or uuid.uuid4())
                                    q.put(StreamEvent(
                                        type="tool_call",
                                        tool_call={
                                            "id": tc_id,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args, ensure_ascii=False),
                                            },
                                        },
                                        elapsed=elapsed,
                                    ))
                                send_frame(build_exec_reply(exec_msg))
                                touch()
                                heartbeat_count = 0
                                continue

                            if "kvServerMessage" in msg or "conversationCheckpointUpdate" in msg:
                                touch()
                                continue

                            iu = msg.get("interactionUpdate") or {}
                            if "heartbeat" in iu:
                                heartbeat_count += 1
                                idle = time.time() - last_activity
                                if text_received and heartbeat_count >= 15 and idle > 120:
                                    finish(elapsed)
                                    break
                                if (
                                    not text_received
                                    and tool_completed_time is not None
                                    and (time.time() - tool_completed_time) > 15
                                ):
                                    finish(elapsed)
                                    break
                                if not text_received and heartbeat_count >= 5 and idle > 30:
                                    finish(elapsed)
                                    break
                                continue

                            if "textDelta" in iu:
                                t = _text_delta(iu["textDelta"])
                                if t:
                                    text_received = True
                                    q.put(StreamEvent(type="text", text=t, elapsed=elapsed))
                                    touch()
                                continue

                            if "thinkingDelta" in iu:
                                t = _text_delta(iu.get("thinkingDelta"))
                                if t:
                                    q.put(StreamEvent(type="thinking", text=t, elapsed=elapsed))
                                    touch()
                                continue

                            if "thinkingCompleted" in iu:
                                q.put(StreamEvent(type="thinking_done", elapsed=elapsed))
                                touch()
                                continue

                            if "partialToolCall" in iu:
                                touch()
                                continue

                            if "toolCallStarted" in iu:
                                tcs = iu.get("toolCallStarted") or {}
                                tc = tcs.get("toolCall") or {}
                                q.put(StreamEvent(
                                    type="tool_call",
                                    tool_call={
                                        "id": tc.get("toolCallId") or str(uuid.uuid4()),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("toolName") or "",
                                            "arguments": tc.get("argsJson") or "{}",
                                        },
                                    },
                                    elapsed=elapsed,
                                ))
                                touch()
                                continue

                            if "toolCallCompleted" in iu:
                                tool_completed_time = time.time()
                                touch()
                                continue

                            if "turnEnded" in iu:
                                te = iu.get("turnEnded") or {}
                                finish(
                                    elapsed,
                                    usage={
                                        "prompt_tokens": int(te.get("inputTokens") or 0),
                                        "completion_tokens": int(te.get("outputTokens") or 0),
                                    },
                                )
                                break

                            nested = iu.get("message") or {}
                            if nested:
                                if "textDelta" in nested:
                                    t = _text_delta(nested["textDelta"])
                                    if t:
                                        text_received = True
                                        q.put(StreamEvent(type="text", text=t, elapsed=elapsed))
                                        touch()
                                if "turnEnded" in nested:
                                    te = nested.get("turnEnded") or {}
                                    finish(
                                        elapsed,
                                        usage={
                                            "prompt_tokens": int(te.get("inputTokens") or 0),
                                            "completion_tokens": int(te.get("outputTokens") or 0),
                                        },
                                    )
                                    break

                    elif isinstance(event, h2.events.StreamEnded):
                        finish(time.time() - start)
                        break

                pending = conn.data_to_send()
                if pending:
                    sock.sendall(pending)

        finally:
            hb_stop.set()
            try:
                sock.close()
            except Exception:
                pass

    except Exception as exc:
        q.put(StreamEvent(type="error", error=str(exc)))
    q.put(None)


async def stream_cursor_agent(
    *,
    prompt: str,
    model: str,
    token: Dict[str, str],
    conversation_id: Optional[str] = None,
    mcp_tools: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    workspace: Optional[str] = None,
) -> AsyncGenerator[StreamEvent, None]:
    q: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _stream_worker,
        q,
        token,
        prompt,
        model,
        conversation_id,
        mcp_tools,
        conversation_history,
        workspace or "",
    )
    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is None:
            break
        yield event
        if event.type in ("done", "error"):
            break
