from __future__ import annotations

"""OpenAI Realtime API ASR WebSocket /v1/realtime 端点。"""

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any, Optional

from aiohttp import web

from handlers import get_state
from handlers.realtime.protocol import (
    new_event_id,
    new_item_id,
    new_session_id,
    parse_transcription_language,
)
from handlers.shared.api_errors import resolve_handler_model
from server.model.model_registry import ModelResolveError
from upstream.qwen.media.asr_realtime import AsrRealtimeSession, AsrStreamEvent

logger = logging.getLogger("rogator")


def _oai_event(event_type: str, **fields: Any) -> dict:
    payload = {"type": event_type, "event_id": new_event_id()}
    payload.update(fields)
    return payload


async def _send_json(ws: web.WebSocketResponse, payload: dict) -> None:
    await ws.send_str(json.dumps(payload, ensure_ascii=False))


async def _send_error(
    ws: web.WebSocketResponse, message: str, *, code: str = "invalid_request"
) -> None:
    await _send_json(ws, _oai_event("error", error={"type": code, "message": message}))


class _OaiTranscriptionTurn:
    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        self.upstream: Optional[AsrRealtimeSession] = None
        self.pump_task: Optional[asyncio.Task[None]] = None

    async def close(self) -> None:
        if self.pump_task and not self.pump_task.done():
            self.pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pump_task
        if self.upstream:
            await self.upstream.close()
        self.upstream = None


class OaiRealtimeAsrConnection:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        qwen: Any,
        qwen_session: Any,
        *,
        model: str,
    ) -> None:
        self._ws = ws
        self._qwen = qwen
        self._qwen_session = qwen_session
        self._model = model
        self._session_id = new_session_id()
        self._language = "zh-CN"
        self._configured = False
        self._closed = False
        self._turn: Optional[_OaiTranscriptionTurn] = None
        self._http: Optional[Any] = None

    async def _http_session(self):
        if self._http is None:
            self._http = await self._qwen._ensure_http_session()
        return self._http

    async def send_created(self) -> None:
        await _send_json(
            self._ws,
            _oai_event(
                "session.created",
                session={
                    "id": self._session_id,
                    "object": "realtime.session",
                    "model": self._model,
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                },
            ),
        )

    async def _pump_turn(self, turn: _OaiTranscriptionTurn) -> None:
        assert turn.upstream is not None
        try:
            async for evt in turn.upstream.iter_events():
                await self._emit_upstream_event(turn.item_id, evt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OAI realtime pump failed: %s", exc)
            await _send_error(self._ws, str(exc))

    async def _emit_upstream_event(self, item_id: str, evt: AsrStreamEvent) -> None:
        if evt.kind == "delta" and evt.delta:
            await _send_json(
                self._ws,
                _oai_event(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=evt.delta,
                ),
            )
        elif evt.kind == "completed":
            await _send_json(
                self._ws,
                _oai_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript=evt.text,
                ),
            )
        elif evt.kind == "failed":
            await _send_error(self._ws, evt.text or "ASR failed", code="server_error")

    async def _ensure_turn(self) -> _OaiTranscriptionTurn:
        if self._turn is None:
            self._turn = _OaiTranscriptionTurn(new_item_id())
            http = await self._http_session()
            self._turn.upstream = AsrRealtimeSession(
                http,
                self._qwen_session.token,
                language=self._language,
            )
            await self._turn.upstream.start()
            self._turn.pump_task = asyncio.create_task(self._pump_turn(self._turn))
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
        await _send_json(
            self._ws, _oai_event("input_audio_buffer.committed", item_id=turn.item_id)
        )
        if turn.upstream:
            await turn.upstream.close()

    async def _handle_session_update(self, data: dict) -> None:
        etype = str(data.get("type") or "")
        if etype == "session.update":
            session = data.get("session")
        else:
            session = {k: v for k, v in data.items() if k != "type"}
        if isinstance(session, dict):
            self._language = parse_transcription_language(session)
        self._configured = True
        await _send_json(
            self._ws,
            _oai_event(
                "session.updated",
                session={
                    "id": self._session_id,
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": self._language,
                    },
                    "turn_detection": None,
                },
            ),
        )

    async def handle_client_event(self, data: dict) -> None:
        etype = str(data.get("type") or "")
        if etype in ("session.update", "transcription_session.update"):
            await self._handle_session_update(data)
            return
        if not self._configured:
            await _send_error(self._ws, "send session.update before audio")
            return
        if etype == "input_audio_buffer.append":
            raw = data.get("audio") or ""
            if not isinstance(raw, str) or not raw:
                await _send_error(self._ws, "audio must be base64 string")
                return
            try:
                pcm = base64.b64decode(raw)
            except Exception:
                await _send_error(self._ws, "invalid base64 audio")
                return
            turn = await self._ensure_turn()
            assert turn.upstream is not None
            await turn.upstream.append_pcm(pcm)
            return
        if etype == "input_audio_buffer.commit":
            await self._finish_turn()
            return
        if etype in ("session.finish", "transcription_session.finish"):
            self._closed = True
            await _send_json(self._ws, _oai_event("session.finished"))
            return
        if etype:
            logger.debug("OAI realtime ignored client event: %s", etype)

    async def run(self) -> None:
        await self.send_created()
        async for msg in self._ws:
            if msg.type == web.WSMsgType.ERROR:
                break
            if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED):
                break
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                await _send_error(self._ws, "invalid JSON")
                continue
            if not isinstance(data, dict):
                await _send_error(self._ws, "event must be object")
                continue
            await self.handle_client_event(data)
            if self._closed:
                break
        if self._turn:
            with contextlib.suppress(Exception):
                await self._finish_turn()


async def oai_realtime_ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    state = get_state()
    model = (request.query.get("model") or state.model or "").strip()
    try:
        resolved = resolve_handler_model(state, model)
    except ModelResolveError as exc:
        await ws.send_str(
            json.dumps(
                {
                    "type": "error",
                    "error": {"message": str(exc), "type": "invalid_request"},
                }
            )
        )
        await ws.close()
        return ws
    qwen = state.client_for(resolved, ("asr",), upstream_name="qwen")
    async with qwen.lease_valid_session() as session:
        if not session:
            await ws.send_str(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "message": "No valid Qwen session",
                            "type": "server_error",
                        },
                    }
                )
            )
            await ws.close()
            return ws
        conn = OaiRealtimeAsrConnection(ws, qwen, session, model=resolved)
        try:
            await conn.run()
        except Exception as exc:
            logger.warning("OAI realtime WS failed: %s", exc)
            with contextlib.suppress(Exception):
                await _send_error(ws, str(exc))
    return ws
