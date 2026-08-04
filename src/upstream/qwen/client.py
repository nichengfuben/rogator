from __future__ import annotations

"""Qwen session and client implementation."""

import asyncio
import logging
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from upstream.qwen.auth.crypto import build_headers_async
from upstream.qwen.chat.session import ChatSession
from upstream.qwen.chat.routes import BASE_URL, CHAT_PATH
from upstream.qwen.account import ModelsFetchMixin, QwenLoginMixin
from upstream.qwen.chat.chat import (
    abort_upstream_on_cancel,
    create_chat_for_session,
    delete_upstream_chat,
    handle_chat_error,
    iter_sse_events,
    stop_upstream_generation,
)
from upstream.qwen.chat.upload.upstream_api import reconnect_sse_events_with_retry, warmup_session
from upstream.qwen.media.asr import AsrTranscriber, aprepare_pcm16_16k_mono
from upstream.qwen.media.tts import TtsService
from upstream.qwen.media.video import VideoService
from upstream.qwen.chat.store import (
    QwenSession,
    describe_sessions,
    mark_invalid as mark_invalid_in,
)
from upstream.qwen.chat.upload.files import UploadMixin
from upstream.qwen.auth.http import (
    map_connection_error,
    merge_session_cookies,
    run_with_connection_retry,
    sync_cookie_store,
)
from core.transport.http import upstream_timeout
from core.transport.owned import HttpTransportMixin
from upstream.qwen.chat.routes import DEFAULT_MODELS
from server.formats import (
    REQUEST_TOTAL_TIMEOUT,
    BaxiaSmBlockedError,
    UpstreamTimeoutError,
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
)
from server.config import CONFIG

logger = logging.getLogger("rogator")


async def _iter_qwen_sse_or_reconnect(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    resp: aiohttp.ClientResponse,
    response_id_box: List[str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    try:
        async for event in iter_sse_events(
            client, resp, session, response_id_out=response_id_box,
        ):
            yield event
    except UpstreamTimeoutError:
        rid = response_id_box[0] if response_id_box else ""
        if not rid:
            raise
        async for event in reconnect_sse_events_with_retry(
            client, session, chat_id, rid, cookies=cookies,
        ):
            yield event


def _note_sse_stats(stats: Dict[str, Any], event: Dict[str, Any]) -> None:
    now_ms = int(time.time() * 1000)
    if not stats["first_chunk_ms"]:
        stats["first_chunk_ms"] = now_ms
    stats["end_chunk_ms"] = now_ms
    content = event.get("content")
    if isinstance(content, str):
        stats["total_chars"] += len(content)


async def _stream_one_chat_post(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: Any,
    response_id_box: List[str],
    stats: Dict[str, Any],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    http = await client._ensure_http_session()
    async with http.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        absorb_fn = getattr(client, "absorb_cookies_for_session", None)
        if callable(absorb_fn):
            absorb_fn(session, resp, binding=cookies)
        elif cookies is not None:
            from upstream.qwen.auth.crypto import absorb_response_cookies
            absorb_response_cookies(cookies, resp)
        if resp.status != 200:
            await handle_chat_error(client, resp, session)
        async for event in _iter_qwen_sse_or_reconnect(
            client, session, chat_id, resp, response_id_box, cookies=cookies,
        ):
            _note_sse_stats(stats, event)
            yield event


def _reraise_or_prepare_retry(exc: Exception, attempt: int) -> Exception:
    """不可重试则抛出；可重试则返回待包装异常占位（调用方 reset）。"""
    conn_err = map_connection_error(exc)
    if conn_err is None or attempt >= 2:
        if conn_err is not None:
            raise conn_err from exc
        raise
    return exc


async def _iter_post_chat_sse_attempts(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: Any,
    response_id_box: List[str],
    stats: Dict[str, Any],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    for attempt in range(1, 3):
        try:
            async for event in _stream_one_chat_post(
                client, session, chat_id, url, payload, headers, timeout,
                response_id_box, stats, cookies=cookies,
            ):
                yield event
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _reraise_or_prepare_retry(exc, attempt)
            await client.reset_http_transport()
            await asyncio.sleep(0.6 * attempt)


async def _post_chat_sse(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    response_id_box: List[str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    from upstream.qwen.auth.report import (
        report_completions_request_id,
        report_streaming_statistics,
    )

    url = f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}"
    timeout = upstream_timeout(REQUEST_TOTAL_TIMEOUT)
    request_id = str(headers.get("X-Request-Id") or headers.get("x-request-id") or "")
    model = str(payload.get("model") or "")
    await report_completions_request_id(
        client, session, request_id=request_id, chat_id=chat_id,
    )
    stats: Dict[str, Any] = {
        "api_start_ms": int(time.time() * 1000),
        "first_chunk_ms": 0,
        "end_chunk_ms": 0,
        "total_chars": 0,
        "is_error": False,
    }
    try:
        async for event in _iter_post_chat_sse_attempts(
            client, session, chat_id, url, payload, headers, timeout,
            response_id_box, stats, cookies=cookies,
        ):
            yield event
    except Exception:
        stats["is_error"] = True
        raise
    finally:
        await report_streaming_statistics(
            client,
            session,
            chat_id=chat_id,
            model=model,
            request_id=request_id,
            response_id=response_id_box[0] if response_id_box else "",
            api_start_ms=stats["api_start_ms"],
            first_chunk_ms=stats["first_chunk_ms"],
            end_chunk_ms=stats["end_chunk_ms"] or int(time.time() * 1000),
            total_chars=stats["total_chars"],
            is_error=stats["is_error"],
        )


async def _chat_completion_stream(
    client: "QwenClient",
    session: QwenSession,
    chat_id: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    response_id_box: List[str] = []
    cancelled = False
    baxia_sm_retry = False
    try:
        async for event in _post_chat_sse(
            client, session, chat_id, payload, headers, response_id_box,
            cookies=cookies,
        ):
            yield event
    except BaxiaSmBlockedError:
        baxia_sm_retry = True
        raise
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        response_id = response_id_box[0] if response_id_box else ""
        await abort_upstream_on_cancel(client, session, chat_id, response_id)
        raise
    except Exception as exc:
        conn_err = map_connection_error(exc)
        if conn_err is not None:
            raise conn_err from exc
        raise
    finally:
        if not cancelled and not baxia_sm_retry:
            await client.cleanup_chat(session, chat_id)


class QwenClient(HttpTransportMixin, UploadMixin, QwenLoginMixin, ModelsFetchMixin):
    def __init__(self, splitter: Any) -> None:
        self._splitter = splitter
        self._cookie_jars: Dict[str, Dict[str, str]] = {}
        self._cookie_jars_lock = threading.Lock()
        self._init_session_pool()
        self._init_http_transport()
        self._init_models_cache(list(DEFAULT_MODELS))
        self._prelogin_target: int = CONFIG.prelogin
        self._login_interval: float = CONFIG.login_interval

    def _cookie_store_key(self, session: QwenSession) -> str:
        return session.username or str(session.user_id or "")

    def _account_cookie_store(self, session: QwenSession) -> Dict[str, str]:
        key = self._cookie_store_key(session)
        with self._cookie_jars_lock:
            return self._cookie_jars.setdefault(key, {})

    def begin_chat_cookies(
        self,
        session: QwenSession,
        *,
        thinking_mode: str = "Fast",
    ) -> Dict[str, str]:
        """单次 chat 请求绑定：create/completion 共用同一 cookie 快照。"""
        store = self._account_cookie_store(session)
        with self._cookie_jars_lock:
            merged = merge_session_cookies(
                session.token,
                store,
                user_id=str(session.user_id or ""),
            )
            sync_cookie_store(store, merged)
            binding = dict(merged)
        if thinking_mode:
            binding["qwen-thinking_mode"] = thinking_mode
        return binding

    def cookies_for_session(
        self,
        session: QwenSession,
        *,
        thinking_mode: str = "Fast",
    ) -> Dict[str, str]:
        return self.begin_chat_cookies(session, thinking_mode=thinking_mode)

    def absorb_cookies_for_session(
        self,
        session: QwenSession,
        response: Any,
        *,
        binding: Optional[Dict[str, str]] = None,
    ) -> None:
        from upstream.qwen.auth.crypto import absorb_response_cookies

        store = self._account_cookie_store(session)
        with self._cookie_jars_lock:
            absorb_response_cookies(store, response)
            if binding is not None:
                absorb_response_cookies(binding, response)

    def cookie_jar_for_session(self, session: QwenSession) -> Dict[str, str]:
        return self._account_cookie_store(session)

    @property
    def cookie_jar(self) -> Dict[str, str]:
        session = self.current_session
        if session is None:
            return {}
        return self._account_cookie_store(session)

    def mark_invalid(self, username: str) -> bool:
        found = mark_invalid_in(self._sessions, username)
        if found:
            self._persist_sessions()
        return found

    def mark_invalid_current(self) -> None:
        session = self.current_session
        if session:
            self.mark_invalid(session.username)

    def describe_sessions(self) -> Dict[str, Any]:
        return describe_sessions(self._sessions)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def _invalidate_session(self, session: QwenSession) -> None:
        session.is_valid = False
        self._persist_sessions()

    async def generate_video(
        self,
        prompt: str,
        image_url: str,
        token: str,
        user_id: str,
        model: str = "qwen-max-latest",
        size: str = "16:9",
        image_name: str = "source.png",
        download: bool = True,
    ) -> Dict[str, Any]:
        async def _run() -> Dict[str, Any]:
            s = await self._ensure_http_session()
            chat_session = ChatSession(s, lambda: None, lambda: {}, lambda: "")
            video_service = VideoService(s, lambda: None, lambda: {}, chat_session.create, chat_session.cleanup)
            return await video_service.generate(
                prompt, image_url, token, user_id, model=model, size=size,
                image_name=image_name, download=download,
            )

        return await run_with_connection_retry(
            "generate_video", _run, transport_owner=self,
        )

    async def synthesize_tts(
        self,
        text: str,
        token: str,
        model: str = "qwen3-max",
        save_dir: Optional[str] = None,
    ) -> Optional[str]:
        from upstream.qwen.chat.routes import TTS_DIR

        async def _run() -> Optional[str]:
            s = await self._ensure_http_session()
            chat_session = ChatSession(s, lambda: None, lambda: {}, lambda: "")
            tts_service = TtsService(
                s, lambda: None, lambda: {}, lambda: "",
                chat_session.create, chat_session.send_placeholder_message, chat_session.cleanup,
            )
            return await tts_service.synthesize(text, token, model=model, save_dir=save_dir or TTS_DIR)

        return await run_with_connection_retry(
            "synthesize_tts", _run, transport_owner=self,
        )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        session: QwenSession,
        *,
        filename: str = "",
        content_type: str = "",
        language: str = "",
    ) -> str:
        async def _run() -> str:
            pcm = await aprepare_pcm16_16k_mono(
                audio_bytes, filename=filename, content_type=content_type,
            )
            http = await self._ensure_http_session()
            asr = AsrTranscriber(http, session.token)
            return await asr.transcribe(pcm, language=language or "zh-CN")

        return await run_with_connection_retry(
            "transcribe_audio", _run, transport_owner=self,
        )

    async def create_chat(
        self,
        session: QwenSession,
        model: str,
        *,
        cookies: Optional[Dict[str, str]] = None,
    ) -> str:
        return await create_chat_for_session(self, session, model, cookies=cookies)

    async def stop_generation(
        self,
        session: QwenSession,
        chat_id: str,
        response_id: str = "",
    ) -> bool:
        return await stop_upstream_generation(self, session, chat_id, response_id)

    async def cleanup_chat(self, session: QwenSession, chat_id: str) -> bool:
        return await delete_upstream_chat(self, session, chat_id)

    async def chat_completion(
        self,
        session: QwenSession,
        chat_id: str,
        messages: List[Dict[str, Any]],
        model: str = "qwen3.7-max",
        files: Optional[List[Dict[str, Any]]] = None,
        *,
        qwen_thinking_enabled: bool = False,
        qwen_thinking_mode: str = "Fast",
        cookies: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not messages:
            raise ValueError("messages cannot be empty")
        user_content = messages[0].get("content", "")
        if not user_content:
            user_content = extract_last_user_content(messages)
        qwen_message = build_qwen_message(
            user_content, model, files,
            thinking_enabled=qwen_thinking_enabled,
            thinking_mode=qwen_thinking_mode,
        )
        payload = build_chat_payload(chat_id, model, qwen_message)
        if cookies is None:
            cookies = self.begin_chat_cookies(
                session, thinking_mode=qwen_thinking_mode,
            )
        headers = await build_headers_async(
            session.token,
            chat_id=chat_id,
            include_sse=True,
            api_path=CHAT_PATH,
            cookies=cookies,
        )
        async for event in _chat_completion_stream(
            self, session, chat_id, payload, headers, cookies=cookies,
        ):
            yield event

    async def shutdown(self) -> None:
        await self.close_http_transport()
