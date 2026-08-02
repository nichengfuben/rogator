from __future__ import annotations

"""Qwen Chat ASR：wsgu_asr WebSocket + SpeechTranscriber 封装。"""

import asyncio
import contextlib
import io
import json
import logging
import shutil
import uuid
import wave
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from core.transport.compat import removeprefix

from upstream.qwen.auth.crypto import build_asr_ws_headers
from upstream.qwen.chat.routes import (
    ASR_AUDIO_CHUNK_BYTES,
    ASR_MAX_DURATION_SEC,
    ASR_SAMPLE_RATE,
    ASR_WS_PATH,
    ASR_WS_TIMEOUT,
    BASE_URL,
)

logger = logging.getLogger("rogator")

try:
    import audioop as _audioop
except ImportError:
    _audioop = None


def _uuid_hex32() -> str:
    return uuid.uuid4().hex


def normalize_asr_language(raw: str) -> str:
    lang = (raw or "").strip()
    if not lang:
        return "zh-CN"
    lower = lang.lower().replace("_", "-")
    if lower in ("zh", "zh-cn", "cmn", "chinese"):
        return "zh-CN"
    if lower in ("en", "en-us", "english"):
        return "en-US"
    if lower in ("ja", "ja-jp", "japanese"):
        return "ja-JP"
    if "-" in lower:
        a, b = lower.split("-", 1)
        return f"{a}-{b.upper()}"
    return lower


def _guess_format(filename: str, content_type: str) -> str:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if name.endswith(".wav") or "wav" in ctype:
        return "wav"
    if name.endswith(".mp3") or "mpeg" in ctype or "mp3" in ctype:
        return "mp3"
    if name.endswith(".webm") or "webm" in ctype:
        return "webm"
    if name.endswith((".m4a", ".mp4")) or "m4a" in ctype:
        return "m4a"
    if name.endswith(".ogg") or "ogg" in ctype:
        return "ogg"
    if name.endswith(".flac") or "flac" in ctype:
        return "flac"
    if "pcm" in ctype or name.endswith(".pcm"):
        return "pcm"
    return "unknown"


def _wav_to_pcm16_16k(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        if _audioop is None:
            raise ValueError("非 16-bit WAV，需 ffmpeg 或 Python<3.13")
        frames = _audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2
    if nchannels != 1:
        if _audioop is None:
            raise ValueError("多声道 WAV 需 ffmpeg 或 Python<3.13")
        frames = _audioop.tomono(
            frames, sampwidth, 0.5, 0.5,
        ) if nchannels == 2 else _audioop.tomono(frames, sampwidth, 1.0, 0.0)
    if framerate != ASR_SAMPLE_RATE:
        if _audioop is None:
            raise ValueError(f"采样率 {framerate}Hz 需 ffmpeg 重采样至 {ASR_SAMPLE_RATE}Hz")
        frames, _ = _audioop.ratecv(
            frames, sampwidth, 1, framerate, ASR_SAMPLE_RATE, None,
        )
    return frames


async def _ffmpeg_to_pcm16_16k(data: bytes) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("无法解析音频，请提供 16kHz mono WAV/PCM 或安装 ffmpeg")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
        "-ar", str(ASR_SAMPLE_RATE),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=data)
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"ffmpeg 转换失败: {err or proc.returncode}")
    if not stdout:
        raise RuntimeError("ffmpeg 输出为空")
    return stdout


async def aprepare_pcm16_16k_mono(
    audio_bytes: bytes,
    *,
    filename: str = "",
    content_type: str = "",
) -> bytes:
    if not audio_bytes:
        raise ValueError("音频为空")
    fmt = _guess_format(filename, content_type)
    if fmt == "pcm":
        return audio_bytes
    if fmt == "wav":
        try:
            return _wav_to_pcm16_16k(audio_bytes)
        except ValueError:
            return await _ffmpeg_to_pcm16_16k(audio_bytes)
    return await _ffmpeg_to_pcm16_16k(audio_bytes)


def build_start_transcription(task_id: str, language: str) -> str:
    return json.dumps({
        "header": {
            "message_id": _uuid_hex32(),
            "task_id": task_id,
            "namespace": "SpeechTranscriber",
            "name": "StartTranscription",
        },
        "context": {},
        "payload": {
            "sample_rate": ASR_SAMPLE_RATE,
            "format": "pcm",
            "enable_intermediate_result": True,
            "enable_inverse_text_normalization": True,
            "enable_punctuation_prediction": True,
            "language": normalize_asr_language(language),
        },
    }, ensure_ascii=False)


def build_stop_transcription(task_id: str) -> str:
    return json.dumps({
        "header": {
            "message_id": _uuid_hex32(),
            "task_id": task_id,
            "namespace": "SpeechTranscriber",
            "name": "StopTranscription",
        },
    }, ensure_ascii=False)


async def _consume_asr_ws(
    transcriber: AsrTranscriber,
    ws: aiohttp.ClientWebSocketResponse,
    started: asyncio.Event,
) -> AsyncGenerator[str, None]:
    prev_text = ""
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"ASR WebSocket 错误: {ws.exception()}")
        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            break
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        name = (data.get("header") or {}).get("name") or ""
        if name == "TranscriptionStarted":
            started.set()
        finished = transcriber._apply_event(data)
        if transcriber._last_text != prev_text:
            prev_text = transcriber._last_text
            yield transcriber._last_text
        if finished:
            if name == "TaskFailed":
                raise RuntimeError("ASR TaskFailed")
            break


class AsrTranscriber:
    """Qwen wsgu_asr WebSocket 音频为空"""

    def __init__(self, session: aiohttp.ClientSession, token: str) -> None:
        self._session = session
        self._token = token
        self._results: List[str] = []
        self._last_text: str = ""

    @property
    def last_text(self) -> str:
        return self._last_text

    def _apply_event(self, data: Dict[str, Any]) -> bool:
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

    async def transcribe(
        self,
        pcm_data: bytes,
        *,
        language: str = "zh-CN",
        timeout: float = ASR_WS_TIMEOUT,
    ) -> str:
        async for _ in self.transcribe_stream(pcm_data, language=language, timeout=timeout):
            pass
        text = self._last_text.strip()
        if not text:
            raise RuntimeError("ASR 音频为空???")
        return text

    async def transcribe_stream(
        self,
        pcm_data: bytes,
        *,
        language: str = "zh-CN",
        timeout: float = ASR_WS_TIMEOUT,
    ) -> AsyncGenerator[str, None]:
        if not pcm_data:
            raise ValueError("PCM 音频为空")
        max_bytes = ASR_SAMPLE_RATE * 2 * ASR_MAX_DURATION_SEC
        if len(pcm_data) > max_bytes:
            pcm_data = pcm_data[:max_bytes]

        host = removeprefix(removeprefix(BASE_URL, "https://"), "http://")
        ws_url = f"wss://{host}{ASR_WS_PATH}?token={self._token}"
        task_id = _uuid_hex32()
        self._results = []
        self._last_text = ""
        started = asyncio.Event()

        ws_headers = build_asr_ws_headers(self._token)
        async with self._session.ws_connect(
            ws_url, ssl=False, heartbeat=30.0, proxy=None,
            headers=ws_headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as ws:
            await ws.send_str(build_start_transcription(task_id, language))

            async def _sender() -> None:
                await started.wait()
                for off in range(0, len(pcm_data), ASR_AUDIO_CHUNK_BYTES):
                    await ws.send_bytes(pcm_data[off:off + ASR_AUDIO_CHUNK_BYTES])
                await ws.send_str(build_stop_transcription(task_id))

            send_task = asyncio.create_task(_sender())
            try:
                async for text in _consume_asr_ws(self, ws, started):
                    yield text
            finally:
                if not started.is_set():
                    started.set()
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
