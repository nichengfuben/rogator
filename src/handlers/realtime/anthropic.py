from __future__ import annotations

"""Anthropic Realtime ASR WebSocket /anthropic/v1/realtime 端点（OpenAI 转写事件格式）。"""

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any, Optional

from aiohttp import web

from handlers import get_state
from handlers.realtime.protocol import (
    build_transcription_session,
    extract_session_update_payload,
    new_event_id,
    new_item_id,
    new_session_id,
    parse_transcription_language,
    parse_transcription_model,
)
from handlers.shared.api_errors import resolve_handler_model
from server.model.platform_models import QWEN_ASR_EXTERNAL_ID
from server.model.model_registry import ModelResolveError
from upstream.qwen.media.asr_realtime import AsrRealtimeSession, AsrStreamEvent

logger = logging.getLogger("rogator")


async def _send(ws: web.WebSocketResponse, payload: dict) -> None:
    await ws.send_str(json.dumps(payload, ensure_ascii=False))


async def _send_invalid_request(ws: web.WebSocketResponse, message: str) -> None:
    await _send(
        ws,
        {
            "type": "error",
            "event_id": new_event_id(),
            "error": {"type": "invalid_request", "message": message},
        },
    )


def _oai_event(event_type: str, **fields: Any) -> dict:
    payload = {"type": event_type, "event_id": new_event_id()}
    payload.update(fields)
    return payload


class _AntTurn:
    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        self.upstream: Optional[AsrRealtimeSession] = None
        self.pump_task: Optional[asyncio.Task[None]] = None


class AnthropicRealtimeAsrConnection:
    def __init__(
        self,
        ws: web.WebSocketResponse,
        qwen: Any,
        qwen_session: Any,
        *,
        model: str,
        intent: str = "",
    ) -> None:
        self._ws = ws
        self._qwen = qwen
        self._qwen_session = qwen_session
        self._model = model
        self._intent = intent.strip().lower()
        self._session_id = new_session_id()
        self._language = "zh-CN"
        self._transcription_model = parse_transcription_model({})
        self._configured = self._intent == "transcription"
        self._closed = False
        self._turn: Optional[_AntTurn] = None
        self._http: Optional[Any] = None

    def _session_payload(self) -> dict:
        return build_transcription_session(
            self._session_id,
            language=self._language,
            model=self._transcription_model,
        )

    async def _http_session(self):
        if self._http is None:
            self._http = await self._qwen._ensure_http_session()
        return self._http

    async def send_session_start(self) -> None:
        await _send(self._ws, _oai_event("session.created", session=self._session_payload()))

    async def _emit_upstream_event(self, item_id: str, evt: AsrStreamEvent) -> None:
        if evt.kind == "delta" and evt.delta:
            await _send(
                self._ws,
                _oai_event(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=evt.delta,
                ),
            )
        elif evt.kind == "completed":
            await _send(
                self._ws,
                _oai_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript=evt.text,
                ),
            )
        elif evt.kind == "failed":
            await _send(
                self._ws,
                _oai_event(
                    "error",
                    error={"type": "server_error", "message": evt.text or "ASR failed"},
                ),
            )

    async def _pump(self, turn: _AntTurn) -> None:
        assert turn.upstream is not None
        try:
            async for evt in turn.upstream.iter_events():
                await self._emit_upstream_event(turn.item_id, evt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Anthropic realtime pump failed: %s", exc)
            await _send(
                self._ws,
                _oai_event("error", error={"type": "server_error", "message": str(exc)}),
            )

    async def _ensure_turn(self) -> _AntTurn:
        if self._turn is None:
            turn = _AntTurn(new_item_id())
            http = await self._http_session()
            turn.upstream = AsrRealtimeSession(
                http,
                self._qwen_session.token,
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
        await _send(
            self._ws,
            _oai_event("input_audio_buffer.committed", item_id=turn.item_id),
        )
        if turn.upstream:
            await turn.upstream.close()

    async def _handle_session_update(self, data: dict) -> None:
        session = extract_session_update_payload(data)
        if session:
            self._language = parse_transcription_language(session)
            self._transcription_model = parse_transcription_model(session)
        self._configured = True
        await _send(
            self._ws,
            _oai_event("session.updated", session=self._session_payload()),
        )

    async def _handle_audio_append(self, data: dict) -> None:
        raw = data.get("audio") or ""
        if not isinstance(raw, str):
            await _send_invalid_request(self._ws, "audio required")
            return
        try:
            pcm = base64.b64decode(raw)
        except Exception:
            await _send_invalid_request(self._ws, "invalid base64")
            return
        turn = await self._ensure_turn()
        assert turn.upstream is not None
        await turn.upstream.append_pcm(pcm)

    async def handle_client_event(self, data: dict) -> None:
        etype = str(data.get("type") or "")
        if etype in ("session.update", "transcription_session.update"):
            await self._handle_session_update(data)
            return
        if not self._configured:
            await _send_invalid_request(self._ws, "send session.update before audio")
            return
        if etype == "input_audio_buffer.append":
            await self._handle_audio_append(data)
            return
        if etype == "input_audio_buffer.commit":
            await self._finish_turn()
            return
        if etype in ("session.finish", "transcription_session.finish"):
            self._closed = True
            await _send(self._ws, _oai_event("session.finished"))
            return

    async def run(self) -> None:
        await self.send_session_start()
        async for msg in self._ws:
            if msg.type in (
                web.WSMsgType.CLOSE,
                web.WSMsgType.CLOSED,
                web.WSMsgType.ERROR,
            ):
                break
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                await _send_invalid_request(self._ws, "invalid JSON")
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
    model = (request.query.get("model") or QWEN_ASR_EXTERNAL_ID).strip()
    try:
        resolved = resolve_handler_model(state, model)
    except ModelResolveError as exc:
        await ws.send_str(
            json.dumps(
                {
                    "type": "error",
                    "event_id": new_event_id(),
                    "error": {"type": "invalid_request", "message": str(exc)},
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
                        "event_id": new_event_id(),
                        "error": {
                            "type": "server_error",
                            "message": "No valid Qwen session",
                        },
                    }
                )
            )
            await ws.close()
            return ws
        conn = AnthropicRealtimeAsrConnection(
            ws,
            qwen,
            session,
            model=resolved,
            intent=(request.query.get("intent") or "").strip(),
        )
        try:
            await conn.run()
        except Exception as exc:
            logger.warning("Anthropic realtime WS failed: %s", exc)
    return ws
