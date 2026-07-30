from __future__ import annotations

import base64
import json
import os
import platform
import socket
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from upstream.cursor.setup.config import cursor_agent_config


@dataclass
class StreamEvent:
    type: str
    text: str = ""
    error: str = ""
    tool_call: Dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    tool_result: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    elapsed: float = 0.0


def agent_config() -> Dict[str, Any]:
    return cursor_agent_config()


def agent_host() -> str:
    base = str(agent_config().get("base_url") or "https://api2.cursor.sh")
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


def _send_h2_chunk(conn, sock, stream_id: int, piece: bytes, sock_lock) -> None:
    # 调用方若传入 sock_lock，须已持有；此处不再二次加锁（避免非重入 Lock 死锁）。
    _ = sock_lock
    conn.send_data(stream_id, piece, end_stream=False)
    sock.sendall(conn.data_to_send())


def _recv_flow_window(conn, sock, sock_lock) -> None:
    # 同上：不在此处抢锁，避免 send_frame/heartbeat 持锁调用时自死锁。
    _ = sock_lock
    chunk = sock.recv(65536)
    if not chunk:
        raise ConnectionError("Connection closed while waiting for flow control")
    conn.receive_data(chunk)
    pending = conn.data_to_send()
    if pending:
        sock.sendall(pending)


def safe_send_data(conn, sock, stream_id: int, data: bytes, *, sock_lock=None) -> None:
    """按 H2 流控分块发送，避免大 payload 阻塞。

    若传入 ``sock_lock``，则在整个发送过程持有该锁；调用方不要再外包一层同锁，
    除非使用可重入锁。
    """
    max_chunk = 16384

    def _send_all() -> None:
        offset = 0
        while offset < len(data):
            window = conn.local_flow_control_window(stream_id)
            if window <= 0:
                try:
                    _recv_flow_window(conn, sock, None)
                except socket.timeout:
                    time.sleep(0.05)
                continue
            chunk_size = min(max_chunk, window, len(data) - offset)
            _send_h2_chunk(conn, sock, stream_id, data[offset:offset + chunk_size], None)
            offset += chunk_size

    if sock_lock is not None:
        with sock_lock:
            _send_all()
    else:
        _send_all()


def build_selected_image(img_path: Any) -> Dict[str, Any]:
    if isinstance(img_path, str):
        img_data = open(img_path, "rb").read()
        ext = os.path.splitext(img_path)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime = mime_map.get(ext, "image/png")
        return {
            "uuid": str(uuid.uuid4()),
            "path": img_path,
            "mimeType": mime,
            "blobIdWithData": {
                "blobId": base64.b64encode(os.urandom(16)).decode(),
                "data": base64.b64encode(img_data).decode(),
            },
        }
    data = img_path.get("data", "")
    if not data and img_path.get("path"):
        p = img_path["path"]
        if os.path.isfile(p):
            data = base64.b64encode(open(p, "rb").read()).decode()
    return {
        "uuid": str(uuid.uuid4()),
        "path": img_path.get("path", ""),
        "mimeType": img_path.get("mimeType", "image/png"),
        "blobIdWithData": {
            "blobId": base64.b64encode(os.urandom(16)).decode(),
            "data": data,
        },
    }


def build_selected_file(file_spec: Any) -> Dict[str, Any]:
    if isinstance(file_spec, str):
        with open(file_spec, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"content": content, "path": file_spec}
    content = file_spec.get("content", "")
    if not content and file_spec.get("path"):
        p = file_spec["path"]
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
    result: Dict[str, Any] = {"content": content, "path": file_spec.get("path", "")}
    if file_spec.get("relativePath"):
        result["relativePath"] = file_spec["relativePath"]
    return result


def build_selected_context(
    images: Optional[List[Any]] = None,
    files: Optional[List[Any]] = None,
) -> Optional[Dict[str, Any]]:
    ctx: Dict[str, Any] = {}
    if images:
        ctx["selectedImages"] = [build_selected_image(img) for img in images]
    if files:
        ctx["files"] = [build_selected_file(f) for f in files]
    return ctx or None


def build_exec_reply(exec_msg: Dict[str, Any]) -> Dict[str, Any]:
    from upstream.cursor.stream.exec import execute_tool

    results = execute_tool(exec_msg)
    return {"execClientMessage": results[0] if results else {}}


def build_interaction_reply(iq: Dict[str, Any]) -> Dict[str, Any]:
    iq_id = iq.get("id", 0)
    response: Dict[str, Any] = {"interactionResponse": {"id": iq_id}}
    ir = response["interactionResponse"]
    if "askQuestionInteractionQuery" in iq:
        ir["askQuestionInteractionResponse"] = {}
    elif "webSearchRequestQuery" in iq:
        ir["webSearchRequestResponse"] = {"approved": False, "reason": "Auto-rejected"}
    elif "switchModeRequestQuery" in iq:
        ir["switchModeRequestResponse"] = {}
    elif "createPlanRequestQuery" in iq:
        ir["createPlanRequestResponse"] = {}
    elif "mcpAuthRequestQuery" in iq:
        ir["mcpAuthRequestResponse"] = {"approved": {}}
    elif "connectScmRequestQuery" in iq:
        ir["connectScmRequestResponse"] = {"approved": {}}
    elif "setupVmEnvironmentQuery" in iq:
        ir["setupVmEnvironmentResponse"] = {"approved": {}}
    elif "generateImageRequestQuery" in iq:
        ir["generateImageRequestResponse"] = {"rejected": {"reason": "Image generation not supported"}}
    elif "webFetchRequestQuery" in iq:
        ir["webFetchRequestResponse"] = {"approved": {}}
    elif "prManagementRequestQuery" in iq:
        ir["prManagementRequestResponse"] = {"rejected": {"reason": "PR management not supported"}}
    elif "subagentStartRequestQuery" in iq:
        ir["subagentStartRequestResponse"] = {"rejected": {"reason": "Subagents not supported"}}
    elif "subagentStopRequestQuery" in iq:
        ir["subagentStopRequestResponse"] = {}
    elif "beforeSubmitPromptRequestQuery" in iq:
        ir["beforeSubmitPromptRequestResponse"] = {"continue": True}
    elif "afterAgentResponseRequestQuery" in iq:
        ir["afterAgentResponseRequestResponse"] = {}
    elif "afterAgentThoughtRequestQuery" in iq:
        ir["afterAgentThoughtRequestResponse"] = {}
    elif "preToolUseRequestQuery" in iq:
        ir["preToolUseRequestResponse"] = {"permission": "allow"}
    elif "postToolUseRequestQuery" in iq:
        ir["postToolUseRequestResponse"] = {}
    elif "postToolUseFailureRequestQuery" in iq:
        ir["postToolUseFailureRequestResponse"] = {}
    elif "stopRequestQuery" in iq:
        ir["stopRequestResponse"] = {}
    elif "preCompactRequestQuery" in iq:
        ir["preCompactRequestResponse"] = {}
    else:
        ir["askQuestionInteractionResponse"] = {}
    return response


def text_delta(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("delta") or "")
    return ""


def parse_error(msg: Dict[str, Any]) -> str:
    err = msg.get("error") or {}
    mt = err.get("message", "")
    for det in err.get("details") or []:
        dg = det.get("debug") or {}
        if dg.get("error"):
            return f"{dg['error']}: {(dg.get('details') or {}).get('detail', '')}"
        val = det.get("value")
        if val:
            try:
                return base64.b64decode(val).decode("utf-8", errors="replace")
            except Exception:
                return str(val)
    return mt or "Unknown error"
