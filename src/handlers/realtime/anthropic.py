from __future__ import annotations

"""Anthropic Realtime ASR WebSocket?/anthropic/v1/realtime??"""

import asyncio
import base64
import contextlib
import json
import logging
import uuid
from typing import Any, Optional

from aiohttp import web

from handlers import get_state
from handlers.realtime.protocol import parse_transcription_language
from handlers.shared.api_errors import resolve_handler_model
from server.model.model_registry import ModelResolveError
from upstream.qwen.media.asr_realtime import AsrRealtimeSession, AsrStreamEvent

logger = logging.getLogger("rogator")


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


async def _send(ws: web.WebSocketResponse, payload: dict) -> None:
    await ws.send_str(json.dumps(payload, ensure_ascii=False))


class _AntTurn:
    def __init__(self, message_id: str) -> None:
        self.message_id = message_id
        self.block_index = 0
        self.upstream: Optional[AsrRealtimeSession] = None
        self.pump_task: Optional[asyncio.Task[None]] = None
        self.started = False


class AnthropicRealtimeAsrConnection:
    def __init__(self, ws: web.WebSocketResponse, qwen: Any, qwen_session: Any, *, model: str) -> None:
        self._ws = ws
        self._qwen = qwen
        self._qwen_session = qwen_session
        self._model = model
        self._language = "zh-CN"
        self._configured = False
        self._closed = False
        self._turn: Optional[_AntTurn] = None
        self._http: Optional[Any] = None

    async def _http_session(self):
        if self._http is None:
            self._http = await self._qwen._ensure_http_session()
        return self._http

    async def send_session_start(self) -> None:
        await _send(self._ws, {
            "type": "session.start",
            "session": {"model": self._model, "modalities": ["text"]},
        })

    async def _begin_message(self, turn: _AntTurn) -> None:
        if turn.started:
            return
        turn.started = True
        await _send(self._ws, {
            "type": "message_start",
            "message": {
                "id": turn.message_id,
                "type": "message",
                "role": "assistant",
                "model": self._model,
                "content": [],
            },
        })
        await _send(self._ws, {
            "type": "content_block_start",
            "index": turn.block_index,
            "content_block": {"type": "text", "text": ""},
        })

    async def _emit_event(self, turn: _AntTurn, evt: AsrStreamEvent) -> None:
        if evt.kind == "delta" and evt.delta:
            await self._begin_message(turn)
            await _send(self._ws, {
                "type": "content_block_delta",
                "index": turn.block_index,
                "delta": {"type": "text_delta", "text": evt.delta},
            })
        elif evt.kind == "completed":
            await self._begin_message(turn)
            if evt.text and not turn.started:
                await _send(self._ws, {
                    "type": "content_block_delta",
                    "index": turn.block_index,
                    "delta": {"type": "text_delta", "text": evt.text},
                })
            await _send(self._ws, {"type": "content_block_stop", "index": turn.block_index})
            await _send(self._ws, {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            })
            await _send(self._ws, {"type": "message_stop"})
            turn.started = False
        elif evt.kind == "failed":
            await _send(self._ws, {
                "type": "error",
                "error": {"type": "api_error", "message": evt.text or "ASR failed"},
            })

    async def _pump(self, turn: _AntTurn) -> None:
        assert turn.upstream is not None
        try:
            async for evt in turn.upstream.iter_events():
                await self._emit_event(turn, evt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Anthropic realtime pump failed: %s", exc)
            await _send(self._ws, {
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            })

    async def _ensure_turn(self) -> _AntTurn:
        if self._turn is None:
            turn = _AntTurn(_msg_id())
            http = await self._http_session()
            turn.upstream = AsrRealtimeSession(
                http, self._qwen_session.token,
                username=self._qwen_session.username,
                language=self._language,
            )
            await turn.upstream.start()
            turn.pump_task = asyncio.create_task(self._pump(turn))
            self._turn = turn
        return self._turn

    async def _finish_turn(self) -> None:
        if self._turn is None:
            return
        turn = self._turn
        self._turn = None
        if turn.upstream:
            await turn.upstream.commit()
        if turn.pump_task:
            await turn.pump_task
        if turn.upstream:
            await turn.upstream.close()
        await _send(self._ws, {"type": "input_audio_buffer.committed"})

    async def handle_client_event(self, data: dict) -> None:
        etype = str(data.get("type") or "")
        if etype == "session.update":
            session = data.get("session")
            if isinstance(session, dict):
                self._language = parse_transcription_language(session)
            self._configured = True
            await _send(self._ws, {
                "type": "session.updated",
                "session": {
                    "input_audio_transcription": {"language": self._language},
                },
            })
            return
        if not self._configured:
            await _send(self._ws, {
                "type": "error",
                "error": {"type": "invalid_request", "message": "send session.update first"},
            })
            return
        if etype == "input_audio_buffer.append":
            raw = data.get("audio") or ""
            if not isinstance(raw, str):
                await _send(self._ws, {
                    "type": "error",
                    "error": {"type": "invalid_request", "message": "audio required"},
                })
                return
            try:
                pcm = base64.b64decode(raw)
            except Exception:
                await _send(self._ws, {
                    "type": "error",
                    "error": {"type": "invalid_request", "message": "invalid base64"},
                })
                return
            turn = await self._ensure_turn()
            assert turn.upstream is not None
            await turn.upstream.append_pcm(pcm)
            return
        if etype == "input_audio_buffer.commit":
            await self._finish_turn()
            return
        if etype == "session.finish":
            self._closed = True
            await _send(self._ws, {"type": "session.finished"})
            return

    async def run(self) -> None:
        await self.send_session_start()
        async for msg in self._ws:
            if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                break
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                await self.handle_client_event(data)
            if self._closed:
                break
        if self._turn:
            with contextlib.suppress(Exception):
                await self._finish_turn()


async def anthropic_realtime_ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    state = get_state()
    model = (request.query.get("model") or state.model or "").strip()
    try:
        resolved = resolve_handler_model(state, model)
    except ModelResolveError as exc:
        await ws.send_str(json.dumps({
            "type": "error",
            "error": {"type": "invalid_request", "message": str(exc)},
        }))
        await ws.close()
        return ws
    qwen = state.client_for(resolved, ("asr",), upstream_name="qwen")
    async with qwen.lease_valid_session() as session:
        if not session:
            await ws.send_str(json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": "No valid Qwen session"},
            }))
            await ws.close()
            return ws
        conn = AnthropicRealtimeAsrConnection(ws, qwen, session, model=resolved)
        try:
            await conn.run()
        except Exception as exc:
            logger.warning("Anthropic realtime WS failed: %s", exc)
    return ws
