from __future__ import annotations

"""Realtime ASR 协议与 handler 单元测试。"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handlers.realtime.protocol import (
    build_transcription_session,
    parse_transcription_language,
)
from handlers.realtime.oai import OaiRealtimeAsrConnection


def test_parse_transcription_language_nested() -> None:
    lang = parse_transcription_language({
        "audio": {"input": {"transcription": {"language": "en"}}},
    })
    assert lang == "en"


def test_build_transcription_session_standard_shape() -> None:
    session = build_transcription_session("sess_test", language="zh-CN")
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["format"]["rate"] == 16000
    assert session["audio"]["input"]["transcription"]["language"] == "zh-CN"
    assert session["audio"]["input"]["turn_detection"] is None


@pytest.mark.asyncio
async def test_oai_realtime_connection_commit_flow() -> None:
    sent: list[dict] = []

    ws = MagicMock()
    ws.send_str = AsyncMock(side_effect=lambda s: sent.append(json.loads(s)))

    fake_events = [
        type("E", (), {"kind": "delta", "text": "你好", "delta": "你"})(),
        type("E", (), {"kind": "delta", "text": "你好", "delta": "好"})(),
        type("E", (), {"kind": "completed", "text": "你好", "delta": ""})(),
    ]

    class FakeUpstream:
        def __init__(self, *a, **k) -> None:
            self._commit = __import__("asyncio").Event()

        async def start(self) -> None:
            return None

        async def append_pcm(self, data: bytes) -> None:
            pass

        async def commit(self) -> None:
            self._commit.set()

        async def close(self) -> None:
            return None

        async def iter_events(self):
            await self._commit.wait()
            for evt in fake_events:
                yield evt

    mock_qwen = MagicMock()
    mock_qwen._ensure_http_session = AsyncMock(return_value=MagicMock())
    mock_session = MagicMock(token="tok")

    with patch("handlers.realtime.oai.AsrRealtimeSession", FakeUpstream):
        conn = OaiRealtimeAsrConnection(ws, mock_qwen, mock_session, model="qwen3.7-max")
        await conn.send_created()
        await conn.handle_client_event({
            "type": "session.update",
            "session": {"input_audio_transcription": {"language": "zh-CN"}},
        })
        import base64
        await conn.handle_client_event({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x00" * 320).decode(),
        })
        await conn.handle_client_event({"type": "input_audio_buffer.commit"})

    types = [m["type"] for m in sent]
    assert "session.created" in types
    created = next(m for m in sent if m["type"] == "session.created")
    assert created["session"]["type"] == "transcription"
    assert "session.updated" in types
    assert "conversation.item.input_audio_transcription.delta" in types
    assert "conversation.item.input_audio_transcription.completed" in types
    assert "input_audio_buffer.committed" in types
