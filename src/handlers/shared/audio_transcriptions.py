from __future__ import annotations

"""OpenAI / Anthropic ?? ASR?/v1/audio/transcriptions ??"""

import json
import logging
from typing import Any, Tuple

from aiohttp import web

from handlers import get_state
from handlers.shared.api_errors import (
    anthropic_error_response,
    model_resolve_error_response,
    resolve_handler_model,
)
from server.formats import ClientDisconnectedError, _error_response, _json_response, client_disconnected_response
from server.model.model_registry import ModelResolveError

logger = logging.getLogger("rogator")


def _parse_bool_field(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


async def _read_multipart_audio(
    request: web.Request,
) -> Tuple[bytes, str, str, str, str, bool, str]:
    if not request.content_type or "multipart" not in request.content_type:
        raise ValueError("Content-Type ??? multipart/form-data")
    audio = b""
    filename = ""
    content_type = ""
    model = ""
    language = ""
    response_format = "json"
    stream = False
    reader = await request.multipart()
    async for field in reader:
        if field.name == "file":
            filename = field.filename or "audio.wav"
            content_type = field.headers.get("Content-Type") or ""
            audio = await field.read()
        elif field.name == "model":
            model = (await field.read()).decode("utf-8", errors="replace").strip()
        elif field.name == "language":
            language = (await field.read()).decode("utf-8", errors="replace").strip()
        elif field.name == "response_format":
            response_format = (await field.read()).decode("utf-8", errors="replace").strip() or "json"
        elif field.name == "stream":
            stream = _parse_bool_field((await field.read()).decode("utf-8", errors="replace"))
    if not audio:
        raise ValueError("缺少 file 字段")
    return audio, filename, content_type, model, language, stream, response_format


def _openai_text_response(text: str, response_format: str) -> web.Response:
    if response_format == "text":
        return web.Response(text=text, content_type="text/plain")
    return _json_response({"text": text})


def _oai_sse_delta(text: str) -> bytes:
    return f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': text}, ensure_ascii=False)}\n\n".encode()


def _oai_sse_done(text: str) -> bytes:
    return f"data: {json.dumps({'type': 'transcript.text.done', 'text': text}, ensure_ascii=False)}\n\n".encode()


async def _stream_openai_asr(qwen: Any, session: Any, pcm: bytes, language: str) -> web.StreamResponse:
    from upstream.qwen.media.asr import AsrTranscriber

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(None)
    http = await qwen._ensure_http_session()
    asr = AsrTranscriber(http, session.token)
    prev = ""
    try:
        async for full in asr.transcribe_stream(pcm, language=language or "zh-CN"):
            delta = full[len(prev):]
            prev = full
            if delta:
                await resp.write(_oai_sse_delta(delta))
        await resp.write(_oai_sse_done(prev))
    except Exception as exc:
        err = {"type": "error", "error": {"message": str(exc)}}
        await resp.write(f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def _stream_anthropic_asr(qwen: Any, session: Any, pcm: bytes, language: str) -> web.StreamResponse:
    from upstream.qwen.media.asr import AsrTranscriber

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(None)
    http = await qwen._ensure_http_session()
    asr = AsrTranscriber(http, session.token)
    prev = ""
    try:
        async for full in asr.transcribe_stream(pcm, language=language or "zh-CN"):
            delta = full[len(prev):]
            prev = full
            if not delta:
                continue
            evt = {"type": "transcript.text.delta", "delta": {"text": delta}}
            await resp.write(
                f"event: content_block_delta\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n".encode(),
            )
        done = {"type": "transcript.text.done", "text": prev}
        await resp.write(
            f"event: message_stop\ndata: {json.dumps(done, ensure_ascii=False)}\n\n".encode(),
        )
    except Exception as exc:
        err = {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
        await resp.write(f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n".encode())
    return resp


async def audio_transcriptions_handler(request: web.Request) -> web.Response:
    state = get_state()
    try:
        audio, filename, ctype, model, language, stream, fmt = await _read_multipart_audio(request)
    except ClientDisconnectedError:
        return client_disconnected_response()
    except ValueError as exc:
        return _error_response(400, str(exc))

    try:
        resolved = resolve_handler_model(state, model or state.model)
    except ModelResolveError as exc:
        return model_resolve_error_response(exc)

    qwen = state.client_for(resolved, ("asr",), upstream_name="qwen")
    async with qwen.lease_valid_session() as session:
        if not session:
            return _error_response(503, "No valid Qwen session available")
        try:
            if stream:
                from upstream.qwen.media.asr import aprepare_pcm16_16k_mono
                pcm = await aprepare_pcm16_16k_mono(audio, filename=filename, content_type=ctype)
                return await _stream_openai_asr(qwen, session, pcm, language)
            text = await qwen.transcribe_audio(
                audio, session, filename=filename, content_type=ctype, language=language,
            )
        except Exception as exc:
            logger.warning("ASR failed: %s", exc)
            return _error_response(502, str(exc))
    return _openai_text_response(text, fmt)


async def anthropic_audio_transcriptions_handler(request: web.Request) -> web.Response:
    state = get_state()
    try:
        audio, filename, ctype, model, language, stream, fmt = await _read_multipart_audio(request)
    except ClientDisconnectedError:
        return client_disconnected_response()
    except ValueError as exc:
        return anthropic_error_response(400, str(exc))

    try:
        resolved = resolve_handler_model(state, model or state.model)
    except ModelResolveError as exc:
        return model_resolve_error_response(exc, protocol="anthropic")

    qwen = state.client_for(resolved, ("asr",), upstream_name="qwen")
    async with qwen.lease_valid_session() as session:
        if not session:
            return anthropic_error_response(503, "No valid Qwen session available")
        try:
            if stream:
                from upstream.qwen.media.asr import aprepare_pcm16_16k_mono
                pcm = await aprepare_pcm16_16k_mono(audio, filename=filename, content_type=ctype)
                return await _stream_anthropic_asr(qwen, session, pcm, language)
            text = await qwen.transcribe_audio(
                audio, session, filename=filename, content_type=ctype, language=language,
            )
        except Exception as exc:
            logger.warning("ASR failed: %s", exc)
            return anthropic_error_response(502, str(exc))
    if fmt == "text":
        return web.Response(text=text, content_type="text/plain")
    return _json_response({"text": text})
