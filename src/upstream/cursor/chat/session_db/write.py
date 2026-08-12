from __future__ import annotations

"""session_db 写路径。"""

import json
import time
from typing import Any, Dict, Optional

from upstream.cursor.chat.session_db.schema import (
    DEFAULT_TTL_SEC,
    _bump_gen,
    _connect,
    _ensure_db,
    _lock,
)


def upsert_session(
    *,
    session_id: str,
    conversation_id: str = "",
    workspace: str = "",
    status: str = "running",
    checkpoint: Optional[Dict[str, Any]] = None,
    ttl_sec: float = DEFAULT_TTL_SEC,
    db_path: Optional[str] = None,
) -> None:
    now = time.time()
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                INSERT INTO sessions(
                    session_id, conversation_id, workspace, status,
                    checkpoint_json, updated_at, expires_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    workspace=excluded.workspace,
                    status=excluded.status,
                    checkpoint_json=excluded.checkpoint_json,
                    updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at
                """,
                (
                    session_id,
                    conversation_id or "",
                    workspace or "",
                    status,
                    json.dumps(checkpoint if checkpoint is not None else {}, ensure_ascii=False),
                    now,
                    now + float(ttl_sec),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def update_checkpoint(
    session_id: str,
    checkpoint: Dict[str, Any],
    *,
    ttl_sec: float = DEFAULT_TTL_SEC,
    db_path: Optional[str] = None,
) -> None:
    now = time.time()
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                UPDATE sessions SET checkpoint_json=?, updated_at=?, expires_at=?
                WHERE session_id=?
                """,
                (json.dumps(checkpoint or {}, ensure_ascii=False), now, now + float(ttl_sec), session_id),
            )
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def set_session_status(
    session_id: str,
    status: str,
    *,
    ttl_sec: float = DEFAULT_TTL_SEC,
    db_path: Optional[str] = None,
) -> None:
    now = time.time()
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                UPDATE sessions SET status=?, updated_at=?, expires_at=?
                WHERE session_id=?
                """,
                (status, now, now + float(ttl_sec), session_id),
            )
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def upsert_pending_tool(
    *,
    tool_call_id: str,
    session_id: str,
    base_msg: Dict[str, Any],
    exec_id: int = 0,
    db_path: Optional[str] = None,
) -> None:
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                INSERT INTO pending_tools(tool_call_id, session_id, base_msg_json, exec_id, created_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(tool_call_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    base_msg_json=excluded.base_msg_json,
                    exec_id=excluded.exec_id,
                    created_at=excluded.created_at
                """,
                (
                    tool_call_id,
                    session_id,
                    json.dumps(base_msg or {}, ensure_ascii=False),
                    int(exec_id or 0),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def clear_pending_tools(session_id: str, *, db_path: Optional[str] = None) -> None:
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute("DELETE FROM pending_tools WHERE session_id=?", (session_id,))
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def delete_pending_tool(tool_call_id: str, *, db_path: Optional[str] = None) -> None:
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute("DELETE FROM pending_tools WHERE tool_call_id=?", (tool_call_id,))
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def upsert_blob(
    session_id: str,
    blob_id: str,
    blob_data: str,
    *,
    db_path: Optional[str] = None,
) -> None:
    if not blob_id:
        return
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                INSERT INTO blobs(session_id, blob_id, blob_data, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(session_id, blob_id) DO UPDATE SET
                    blob_data=excluded.blob_data,
                    updated_at=excluded.updated_at
                """,
                (session_id, blob_id, blob_data or "", time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        _bump_gen()


def log_request(
    *,
    kind: str,
    req_id: str = "",
    session_id: str = "",
    prompt_head: str = "",
    history_len: int = 0,
    payload: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    path = _ensure_db(db_path)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                INSERT INTO requests(
                    req_id, session_id, kind, prompt_head, history_len, payload_json, created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    req_id or "",
                    session_id or "",
                    kind,
                    (prompt_head or "")[:500],
                    int(history_len or 0),
                    json.dumps(payload or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def purge_expired(*, db_path: Optional[str] = None) -> int:
    path = _ensure_db(db_path)
    now = time.time()
    with _lock:
        conn = _connect(path)
        try:
            ids = [
                str(r["session_id"])
                for r in conn.execute(
                    "SELECT session_id FROM sessions WHERE expires_at<?", (now,)
                ).fetchall()
            ]
            for sid in ids:
                conn.execute("DELETE FROM pending_tools WHERE session_id=?", (sid,))
                conn.execute("DELETE FROM blobs WHERE session_id=?", (sid,))
                conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
            conn.commit()
            n = len(ids)
        finally:
            conn.close()
        if n:
            _bump_gen()
        return n
