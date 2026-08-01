from __future__ import annotations

"""Cursor Agent GetUsableModels（HTTP/2 unary）。"""

import json
import uuid
from typing import Any, Dict, List

from upstream.cursor.setup.config import cursor_agent_config
from upstream.cursor.models.h2_unary import open_h2_connection, parse_models_body, read_h2_json_body
from upstream.cursor.stream.proto import agent_host, generate_checksum


def fetch_usable_models(token: Dict[str, str]) -> List[Dict[str, Any]]:
    """调用 ``/agent.v1.AgentService/GetUsableModels``，返回 models 数组。"""
    cfg = cursor_agent_config()
    host = agent_host()
    client_version = str(cfg.get("client_version") or "3.12.17")
    timeout = int(cfg.get("request_timeout") or 120)
    sock, conn = open_h2_connection(host)
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
    body = read_h2_json_body(sock, conn, timeout=float(timeout))
    return parse_models_body(body)
