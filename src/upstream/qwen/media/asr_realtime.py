from __future__ import annotations

"""Qwen wsgu_asr 实时会话：边收 PCM chunk 边推送识别增量。"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Literal, Optional

import aiohttp

from core.transport.compat import removeprefix

from upstream.qwen.auth.crypto import build_asr_ws_headers
from upstream.qwen.chat.routes import (
    ASR_AUDIO_CHUNK_BYTES,
    ASR_WS_PATH,
    ASR_WS_TIMEOUT,
    BASE_URL,
)
from upstream.qwen.media.asr import (
    build_start_transcription,
    build_stop_transcription,
    normalize_asr_language,
)

logger = logging.getLogger("rogator")

AsrEventKind = Literal["started", "delta", "completed", "failed"]


@dataclass
class AsrStreamEvent:
    kind: AsrEventKind
    text: str = ""
    delta: str = ""


def _uuid_hex32() -> str:
    import uuid
    return uuid.uuid4().hex


class AsrRealtimeSession:
    """单次转写 turn：StartTranscription → 流式 PCM → StopTranscription。"""

    def __init__(
        self,
        http: aiohttp.ClientSession,
        token: str,
        *,
        username: str = "",
        language: str = "zh-CN",
        timeout: float = ASR_WS_TIMEOUT,
    ) -> None:
        self._http = http
        self._token = token
        self._username = username
        self._language = normalize_asr_language(language)
        self._timeout = timeout
        self._task_id = _uuid_hex32()
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._results: List[str] = []
        self._last_text = ""
        self._prev_emit = ""
        self._started = asyncio.Event()
        self._finished = asyncio.Event()
        self._event_queue: asyncio.Queue[AsrStreamEvent] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._pre_start: List[bytes] = []
        self._failed = False

    @property
    def last_text(self) -> str:
        return self._last_text

    def _apply_upstream(self, data: dict) -> bool:
        name = (data.get("header") or {}).get("name") or ""
        result = str((data.get("payload") or {}).get("result") or "")
        if name == "SentenceBegin":
            self._results.append("")
        elif name in ("TranscriptionResultChanged", "SentenceEnd", "TranscriptionResult"):
            if self._results:
                self._results[-1] = result
            elif result:
                self._results.append(result)
        self._last_text = "".join(self._results)
        return name in ("TranscriptionCompleted", "TaskFailed")

    async def _emit_text_delta(self) -> None:
        if self._last_text == self._prev_emit:
            return
        delta = self._last_text[len(self._prev_emit):]
        self._prev_emit = self._last_text
        if delta:
            await self._event_queue.put(
                AsrStreamEvent("delta", text=self._last_text, delta=delta),
            )

    async def _flush_pre_start(self) -> None:
        if not self._ws or self._ws.closed:
            return
        for blob in self._pre_start:
            for off in range(0, len(blob), ASR_AUDIO_CHUNK_BYTES):
                await self._ws.send_bytes(blob[off:off + ASR_AUDIO_CHUNK_BYTES])
        self._pre_start.clear()

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    self._failed = True
                    await self._event_queue.put(AsrStreamEvent("failed", text=str(self._ws.exception())))
                    break
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                name = (data.get("header") or {}).get("name") or ""
                if name == "TranscriptionStarted":
                    self._started.set()
                    await self._event_queue.put(AsrStreamEvent("started"))
                    await self._flush_pre_start()
                finished = self._apply_upstream(data)
                await self._emit_text_delta()
                if finished:
                    if name == "TaskFailed":
                        self._failed = True
                        await self._event_queue.put(AsrStreamEvent("failed", text=self._last_text))
                    else:
                        await self._event_queue.put(
                            AsrStreamEvent("completed", text=self._last_text.strip()),
                        )
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ASR realtime read failed: %s", exc)
            self._failed = True
            await self._event_queue.put(AsrStreamEvent("failed", text=str(exc)))
        finally:
            self._finished.set()

    async def start(self) -> None:
        host = removeprefix(removeprefix(BASE_URL, "https://"), "http://")
        ws_url = f"wss://{host}{ASR_WS_PATH}?token={self._token}"
        self._ws = await self._http.ws_connect(
            ws_url,
            ssl=False,
            heartbeat=30.0,
            proxy=None,
            headers=build_asr_ws_headers(self._token, username=self._username),
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )
        await self._ws.send_str(build_start_transcription(self._task_id, self._language))
        self._reader_task = asyncio.create_task(self._read_loop())

    async def append_pcm(self, data: bytes) -> None:
        if not data:
            return
        if self._started.is_set() and self._ws and not self._ws.closed:
            for off in range(0, len(data), ASR_AUDIO_CHUNK_BYTES):
                await self._ws.send_bytes(data[off:off + ASR_AUDIO_CHUNK_BYTES])
        else:
            self._pre_start.append(data)

    async def commit(self) -> None:
        try:
            await asyncio.wait_for(self._started.wait(), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            self._failed = True
            await self._event_queue.put(AsrStreamEvent("failed", text="TranscriptionStarted timeout"))
            self._finished.set()
            raise RuntimeError("ASR upstream did not start") from exc
        if self._ws and not self._ws.closed:
            await self._ws.send_str(build_stop_transcription(self._task_id))

    async def iter_events(self) -> AsyncGenerator[AsrStreamEvent, None]:
        while not self._finished.is_set() or not self._event_queue.empty():
            try:
                evt = await asyncio.wait_for(self._event_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            yield evt
            if evt.kind in ("completed", "failed"):
                break

    async def close(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
