from __future__ import annotations

"""Qwen session and client implementation."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from upstream.qwen.auth.crypto import build_headers
from upstream.qwen.chat.session import ChatSession
from upstream.qwen.chat.routes import BASE_URL, CHAT_PATH
from upstream.qwen.account import ModelsFetchMixin, QwenLoginMixin
from upstream.qwen.chat.chat import create_chat_for_session, handle_chat_error, iter_sse_events
from upstream.qwen.media.tts import TtsService
from upstream.qwen.media.video import VideoService
from upstream.qwen.chat.store import (
    QwenSession,
    describe_sessions,
    mark_invalid as mark_invalid_in,
)
from upstream.qwen.chat.upload.files import UploadMixin
from upstream.qwen.auth.http import map_connection_error, run_with_connection_retry
from core.transport.http import upstream_timeout
from core.transport.owned import HttpTransportMixin
from server.formats import (
    DEFAULT_MODELS,
    REQUEST_TOTAL_TIMEOUT,
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
)
from server.config import CONFIG

logger = logging.getLogger("rogator")


class QwenClient(HttpTransportMixin, UploadMixin, QwenLoginMixin, ModelsFetchMixin):
    def __init__(self, splitter: Any) -> None:
        self._splitter = splitter
        self._init_session_pool()
        self._init_http_transport()
        self._init_models_cache(list(DEFAULT_MODELS))
        self._lock = asyncio.Lock()
        self._prelogin_target: int = CONFIG.prelogin
        self._login_interval: float = CONFIG.login_interval

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

    async def create_chat(self, session: QwenSession, model: str) -> str:
        return await create_chat_for_session(self, session, model)

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
        headers = build_headers(session.token, chat_id=chat_id, include_sse=True)
        try:
            s = await self._ensure_http_session()
            async with s.post(
                f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                headers=headers,
                timeout=upstream_timeout(REQUEST_TOTAL_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    await handle_chat_error(self, resp, session)
                async for event in iter_sse_events(self, resp, session):
                    yield event
        except Exception as exc:
            conn_err = map_connection_error(exc)
            if conn_err is not None:
                raise conn_err from exc
            raise

    async def shutdown(self) -> None:
        await self.close_http_transport()
