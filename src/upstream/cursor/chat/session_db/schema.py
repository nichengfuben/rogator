from __future__ import annotations

"""SQLite 连接 / schema / 写世代。"""

import functools
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from core.persist.paths import upstream_dir

DB_FILENAME = "agent_sessions.db"
DEFAULT_TTL_SEC = 900

_lock = threading.RLock()
_write_gen = 0


@functools.lru_cache(maxsize=1)
def resolve_db_path(root_key: str = "") -> str:
    """缓存 DB 绝对路径字符串（root_key 预留测试注入）。"""
    root = Path(root_key) if root_key else None
    path = upstream_dir("cursor", root) / DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def bump_gen() -> None:
    global _write_gen
    _write_gen += 1
    from upstream.cursor.chat.session_db import read as _read

    _read.get_session_row.cache_clear()
    _read.get_session_id_by_tool_call.cache_clear()
    _read.get_latest_session_for_workspace.cache_clear()
    _read.list_pending_tools.cache_clear()
    _read.get_blob.cache_clear()
    _read.load_blobs_for_session.cache_clear()


# 兼容旧私有名
_bump_gen = bump_gen


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or resolve_db_path()
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


_connect = connect


def init_db(db_path: Optional[str] = None) -> None:
    with _lock:
        conn = connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    workspace TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_tools (
                    tool_call_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    base_msg_json TEXT NOT NULL,
                    exec_id INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pending_session
                    ON pending_tools(session_id);
                CREATE TABLE IF NOT EXISTS blobs (
                    session_id TEXT NOT NULL,
                    blob_id TEXT NOT NULL,
                    blob_data TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, blob_id)
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    req_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    prompt_head TEXT NOT NULL DEFAULT '',
                    history_len INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_workspace
                    ON sessions(workspace, updated_at DESC);
                """
            )
            conn.commit()
        finally:
            conn.close()


def ensure_db(db_path: Optional[str] = None) -> str:
    path = db_path or resolve_db_path()
    init_db(path)
    return path


_ensure_db = ensure_db


def cache_gen() -> int:
    return _write_gen
