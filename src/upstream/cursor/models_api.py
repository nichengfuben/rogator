from __future__ import annotations

"""Cursor Agent GetUsableModels（HTTP/2 unary）。"""

import json
import socket
import ssl
import uuid
from typing import Any, Dict, List

try:
    import h2.connection
    import h2.config
    import h2.events
except ImportError as exc:
    raise ImportError("h2 required for cursor upstream: pip install h2") from exc

from upstream.cursor.agent_stream import _host, generate_checksum
from upstream.cursor.config import cursor_agent_config


def fetch_usable_models(token: Dict[str, str]) -> List[Dict[str, Any]]:
    """调用 ``/agent.v1.AgentService/GetUsableModels``，返回 models 数组。"""
    cfg = cursor_agent_config()
    host = _host()
    client_version = str(cfg.get("client_version") or "3.12.17")
    timeout = int(cfg.get("request_timeout") or 120)

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
            raise RuntimeError("HTTP/2 handshake failed")
        events = conn.receive_data(chunk)
        sock.sendall(conn.data_to_send())
        if any(isinstance(e, h2.events.SettingsAcknowledged) for e in events):
            break

    stream_id = conn.get_next_available_stream_id()
    headers = [
        (":method", "POST"),
        (":path", "/agent.v1.AgentService/GetUsableModels"),
        (":scheme", "https"),
        (":authority", host),
        ("content-type", "application/json"),
        ("connect-protocol-version", "1"),
        ("authorization", f"Bearer {token['accessToken']}"),
        ("x-cursor-checksum", generate_checksum(token["machineId"], token["macMachineId"])),
        ("x-cursor-client-version", client_version),
        ("x-request-id", str(uuid.uuid4())),
    ]
    conn.send_headers(stream_id, headers, end_stream=False)
    conn.send_data(stream_id, json.dumps({}).encode("utf-8"), end_stream=True)
    sock.sendall(conn.data_to_send())

    body = b""
    sock.settimeout(float(timeout))
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            events = conn.receive_data(chunk)
            for event in events:
                if isinstance(event, h2.events.DataReceived):
                    conn.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                    body += event.data
                elif isinstance(event, h2.events.StreamEnded):
                    sock.close()
                    if not body:
                        return []
                    data = json.loads(body.decode("utf-8"))
                    models = data.get("models")
                    return list(models) if isinstance(models, list) else []
            pending = conn.data_to_send()
            if pending:
                sock.sendall(pending)
    finally:
        try:
            sock.close()
        except Exception:
            pass

    if not body:
        return []
    data = json.loads(body.decode("utf-8"))
    models = data.get("models")
    return list(models) if isinstance(models, list) else []
