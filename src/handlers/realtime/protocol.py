from __future__ import annotations

"""Realtime ASR 共享：语言解析、item/session id。"""

import uuid
from typing import Any, Dict, Optional

from upstream.qwen.chat.routes import ASR_SAMPLE_RATE

DEFAULT_TRANSCRIPTION_MODEL: str = "qwen-asr"


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
        ("audio", "input", "transcription", "language"),
        ("input_audio_transcription", "language"),
    )
    for path in paths:
        raw = _dig(session, *path)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "zh-CN"


def parse_transcription_model(session: dict) -> str:
    paths = (
        ("audio", "input", "transcription", "model"),
        ("input_audio_transcription", "model"),
    )
    for path in paths:
        raw = _dig(session, *path)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return DEFAULT_TRANSCRIPTION_MODEL


def build_transcription_session(
    session_id: str,
    *,
    language: str,
    model: str = DEFAULT_TRANSCRIPTION_MODEL,
    sample_rate: int = ASR_SAMPLE_RATE,
) -> Dict[str, Any]:
    """OpenAI Realtime 转写会话标准结构（session.type=transcription）。"""
    return {
        "type": "transcription",
        "object": "realtime.session",
        "id": session_id,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": sample_rate},
                "transcription": {
                    "model": model,
                    "language": language,
                },
                "turn_detection": None,
            },
        },
    }


def extract_session_update_payload(data: dict) -> dict:
    """从 session.update / transcription_session.update 取出 session 对象。"""
    etype = str(data.get("type") or "")
    session = data.get("session")
    if isinstance(session, dict):
        return session
    if etype == "transcription_session.update":
        return {k: v for k, v in data.items() if k not in ("type", "event_id")}
    return {}


def parse_turn_detection_manual(session: dict) -> bool:
    td = _dig(session, "audio", "input", "turn_detection")
    if td is None:
        td = session.get("turn_detection")
    if td is None:
        return True
    if isinstance(td, dict) and td.get("type") in (None, "", "none", "disabled"):
        return True
    return False
