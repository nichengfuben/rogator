from __future__ import annotations

"""Qwen session and client implementation."""

import logging
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

from upstream.qwen.auth.crypto import build_headers_async
from upstream.qwen.chat.session import ChatSession
from upstream.qwen.chat.store import (
    QwenSession,
    describe_sessions,
    mark_invalid as mark_invalid_in,
)
from upstream.qwen.chat.routes import BASE_URL, CHAT_PATH, DEFAULT_MODELS
from upstream.qwen.account import ModelsFetchMixin, QwenLoginMixin
from upstream.qwen.chat.chat import (
    create_chat_for_session,
    delete_upstream_chat,
    stop_upstream_generation,
)
from upstream.qwen.completion_stream import chat_completion_stream
from upstream.qwen.media.asr import AsrTranscriber, aprepare_pcm16_16k_mono
from upstream.qwen.media.tts import TtsService
from upstream.qwen.media.video import VideoService
from upstream.qwen.chat.upload.files import UploadMixin
from upstream.qwen.auth.http import (
    merge_session_cookies,
    run_with_connection_retry,
    sync_cookie_store,
)
from core.transport.owned import HttpTransportMixin
from server.formats import (
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
)
from server.config import CONFIG

logger = logging.getLogger("rogator")


class QwenClient(HttpTransportMixin, UploadMixin, QwenLoginMixin, ModelsFetchMixin):
    def _client_session_kwargs(self) -> dict:
        from upstream.qwen.media.proxy_toggle import get_proxy_toggle
        if get_proxy_toggle().enabled:
            return {"use_env_proxy": True}
        return {}

    def _get_proxy_kwarg(self) -> Optional[str]:
        from upstream.qwen.media.proxy_toggle import get_proxy_toggle
        if not get_proxy_toggle().enabled:
            return None
        from server.retry.http_client import active_proxy_url
        return active_proxy_url()

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
        req_id: str = "",
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
        async for event in chat_completion_stream(
            self, session, chat_id, payload, headers,
            cookies=cookies, req_id=req_id,
        ):
            yield event

    async def shutdown(self) -> None:
        await self.close_http_transport()
