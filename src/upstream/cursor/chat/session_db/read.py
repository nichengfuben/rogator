from __future__ import annotations

"""session_db 读路径（lru_cache）。"""

import functools
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.chat.session_db.schema import (
    _connect,
    _ensure_db,
    _lock,
    cache_gen,
)


@functools.lru_cache(maxsize=256)
def get_session_row(session_id: str, gen: int = 0) -> Optional[Tuple[Any, ...]]:
    _ = gen
    path = _ensure_db()
    with _lock:
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT session_id, conversation_id, workspace, status, checkpoint_json, updated_at, expires_at "
                "FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return tuple(row) if row else None
        finally:
            conn.close()


@functools.lru_cache(maxsize=512)
def get_session_id_by_tool_call(tool_call_id: str, gen: int = 0) -> str:
    _ = gen
    path = _ensure_db()
    with _lock:
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT session_id FROM pending_tools WHERE tool_call_id=?",
                (tool_call_id,),
            ).fetchone()
            return str(row["session_id"]) if row else ""
        finally:
            conn.close()


@functools.lru_cache(maxsize=128)
def get_latest_session_for_workspace(workspace: str, gen: int = 0) -> str:
    _ = gen
    path = _ensure_db()
    now = time.time()
    with _lock:
        conn = _connect(path)
        try:
            row = conn.execute(
                """
                SELECT session_id FROM sessions
                WHERE workspace=? AND expires_at>? AND status IN ('parked','done','running')
                ORDER BY updated_at DESC LIMIT 1
                """,
                (workspace or "", now),
            ).fetchone()
            return str(row["session_id"]) if row else ""
        finally:
            conn.close()


@functools.lru_cache(maxsize=256)
def list_pending_tools(session_id: str, gen: int = 0) -> Tuple[Tuple[str, str, int], ...]:
    """返回 (tool_call_id, base_msg_json, exec_id) 元组列表。"""
    _ = gen
    path = _ensure_db()
    with _lock:
        conn = _connect(path)
        try:
            rows = conn.execute(
                "SELECT tool_call_id, base_msg_json, exec_id FROM pending_tools WHERE session_id=?",
                (session_id,),
            ).fetchall()
            return tuple((str(r["tool_call_id"]), str(r["base_msg_json"]), int(r["exec_id"])) for r in rows)
        finally:
            conn.close()


@functools.lru_cache(maxsize=1024)
def get_blob(session_id: str, blob_id: str, gen: int = 0) -> str:
    _ = gen
    path = _ensure_db()
    with _lock:
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT blob_data FROM blobs WHERE session_id=? AND blob_id=?",
                (session_id, blob_id),
            ).fetchone()
            return str(row["blob_data"]) if row else ""
        finally:
            conn.close()


@functools.lru_cache(maxsize=64)
def load_blobs_for_session(session_id: str, gen: int = 0) -> Tuple[Tuple[str, str], ...]:
    _ = gen
    path = _ensure_db()
    with _lock:
        conn = _connect(path)
        try:
            rows = conn.execute(
                "SELECT blob_id, blob_data FROM blobs WHERE session_id=?",
                (session_id,),
            ).fetchall()
            return tuple((str(r["blob_id"]), str(r["blob_data"])) for r in rows)
        finally:
            conn.close()


def session_dict(session_id: str) -> Optional[Dict[str, Any]]:
    row = get_session_row(session_id, cache_gen())
    if not row:
        return None
    return {
        "session_id": row[0],
        "conversation_id": row[1],
        "workspace": row[2],
        "status": row[3],
        "checkpoint": json.loads(row[4] or "{}"),
        "updated_at": row[5],
        "expires_at": row[6],
    }


def pending_tools_dicts(session_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tid, base_json, exec_id in list_pending_tools(session_id, cache_gen()):
        try:
            base = json.loads(base_json)
        except Exception:
            base = {}
        out.append({"tool_call_id": tid, "base_msg": base, "exec_id": exec_id})
    return out


def blobs_dict(session_id: str) -> Dict[str, str]:
    return {bid: data for bid, data in load_blobs_for_session(session_id, cache_gen())}


def new_session_id() -> str:
    return str(uuid.uuid4())
