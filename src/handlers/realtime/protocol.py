from __future__ import annotations

"""Realtime ASR 共享：语言解析、item/session id。"""

import uuid
from typing import Any, Dict, Optional


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def new_item_id() -> str:
    return f"item_{uuid.uuid4().hex}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex}"


def _dig(data: dict, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def parse_transcription_language(session: dict) -> str:
    """从 session.update / transcription_session.update 提取语言。"""
    paths = (
        ("input_audio_transcription", "language"),
        ("audio", "input", "transcription", "language"),
        ("input_audio_transcription", "language"),
    )
    for path in paths:
        raw = _dig(session, *path)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "zh-CN"


def parse_turn_detection_manual(session: dict) -> bool:
    td = session.get("turn_detection")
    if td is None:
        return True
    if isinstance(td, dict) and td.get("type") in (None, "", "none", "disabled"):
        return True
    return False
