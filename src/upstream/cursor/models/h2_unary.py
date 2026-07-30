from __future__ import annotations

import json
import socket
import ssl
from typing import Any, Dict, List, Tuple

try:
    import h2.connection
    import h2.config
    import h2.events
except ImportError as exc:
    raise ImportError("h2 required for cursor upstream: pip install h2") from exc


def open_h2_connection(host: str) -> Tuple[Any, Any]:
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
    return sock, conn


def read_h2_json_body(sock, conn, *, timeout: float) -> bytes:
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
                    return body
            pending = conn.data_to_send()
            if pending:
                sock.sendall(pending)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return body


def parse_models_body(body: bytes) -> List[Dict[str, Any]]:
    if not body:
        return []
    data = json.loads(body.decode("utf-8"))
    models = data.get("models")
    return list(models) if isinstance(models, list) else []
