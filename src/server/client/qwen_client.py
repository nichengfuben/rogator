from __future__ import annotations

"""Qwen session and client implementation。"""

import asyncio
import logging
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp

from core.crypto.crypto import build_headers
from core.transport.chat_session import ChatSession
from core.transport.routes import BASE_URL, CHAT_PATH
from server.client.account import ModelsFetchMixin, SessionLoginMixin
from server.client.chat import create_chat_for_session, handle_chat_error, iter_sse_events
from core.media.tts import TtsService
from core.media.video import VideoService
from server.client.session_store import (
    CLEANUP_INTERVAL,
    QwenSession,
    SessionStoreMeta,
    clean_expired,
    describe_sessions,
    is_session_fatal_error,
    load_session_store,
    mark_invalid as mark_invalid_in,
    mask_username,
    save_sessions,
    valid_session_count,
)
from server.client.uploads import UploadMixin
from server.formats import (
    DEFAULT_MODELS,
    DEFAULT_USER_AGENT,
    REQUEST_TOTAL_TIMEOUT,
    TokenExpiredError,
    UpstreamTimeoutError,
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
)
from server.config import CONFIG

logger = logging.getLogger("rogator")


class QwenClient(UploadMixin, SessionLoginMixin, ModelsFetchMixin):
    def __init__(self, splitter: Any) -> None:
        self._splitter = splitter
        sessions, meta = load_session_store()
        self._sessions: List[QwenSession] = sessions
        self._current_index: int = meta.current_index
        self._account_index: int = meta.account_index
        self._blocked_accounts: Dict[str, float] = dict(meta.blocked_accounts)
        self._lock = asyncio.Lock()
        self._models: List[str] = list(DEFAULT_MODELS)
        self._models_fetch_time: float = 0
        self._models_cache_ttl: float = 300
        self._last_cleanup: float = 0.0
        self._prelogin_target: int = CONFIG.prelogin

    def _save_meta(self) -> List[str]:
        return save_sessions(
            self._sessions,
            current_index=self._current_index,
            account_index=self._account_index,
            blocked_accounts=self._blocked_accounts,
        )

    def block_account(self, username: str, block_seconds: float) -> None:
        """限流/耗尽账号：写入 blocked_accounts 并立即落盘。"""
        until = time.time() + max(block_seconds, 60.0)
        self._blocked_accounts[username] = until
        self._save_meta()
        logger.info(
            "Blocked account %s for %.0fs",
            mask_username(username), block_seconds,
        )

    def _is_account_blocked(self, username: str) -> bool:
        until = self._blocked_accounts.get(username)
        if until is None:
            return False
        if time.time() >= until:
            self._blocked_accounts.pop(username, None)
            return False
        return True

    @property
    def current_session_username(self) -> Optional[str]:
        return self._session_username_at_current()

    def _fix_current_index(self, previous_username: Optional[str] = None) -> None:
        """session 被移除后修正 current_index。"""
        username = previous_username
        if username is None and self._sessions and self._current_index < len(self._sessions):
            username = self._sessions[self._current_index].username
        if username and not any(s.username == username for s in self._sessions):
            valid_indices = [
                i for i, s in enumerate(self._sessions)
                if s.is_valid and not s.is_expired()
            ]
            self._current_index = random.choice(valid_indices) if valid_indices else 0
        elif self._current_index >= len(self._sessions):
            self._current_index = 0

    def _session_username_at_current(self) -> Optional[str]:
        if self._sessions and self._current_index < len(self._sessions):
            return self._sessions[self._current_index].username
        return None

    def prune_expired_sessions(self) -> List[str]:
        """按 JWT exp-30s 即时剔除过期/失效 session（内存）。"""
        previous_username = self._session_username_at_current()
        self._sessions, removed = clean_expired(self._sessions)
        if removed:
            self._fix_current_index(previous_username)
        return removed

    def cleanup_expired_sessions(self) -> List[str]:
        """按 JWT exp-30s 清理过期/失效 session 并落盘（后台任务用，无节流）。"""
        previous_username = self._session_username_at_current()
        removed = self._save_meta()
        if removed:
            self._fix_current_index(previous_username)
            logger.info("Session cleanup: removed %d expired/invalid session(s)", len(removed))
        return removed

    def _persist_sessions(self) -> List[str]:
        """清理并持久化 session 池，同步修正 current_index。"""
        previous_username = self._session_username_at_current()
        removed = self._save_meta()
        self._fix_current_index(previous_username)
        return removed

    def _invalidate_session(self, session: QwenSession) -> None:
        """标记 session 失效并立即清理落盘。"""
        session.is_valid = False
        self._persist_sessions()

    async def _ensure_cleanup(self) -> None:
        """按 CLEANUP_INTERVAL 节流落盘清理（热路径避免频繁写盘）。"""
        now = time.time()
        if now - self._last_cleanup < CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        self.cleanup_expired_sessions()

    @property
    def current_session(self) -> Optional[QwenSession]:
        if not self._sessions or self._current_index >= len(self._sessions):
            return None
        session = self._sessions[self._current_index]
        if not session.is_valid or session.is_expired():
            return None
        return session

    def mark_invalid(self, username: str) -> bool:
        """按 username 精确标记单个 session 失效并立即持久化。"""
        found = mark_invalid_in(self._sessions, username)
        if found:
            self._persist_sessions()
        return found

    def mark_invalid_current(self) -> None:
        session = self.current_session
        if session:
            self.mark_invalid(session.username)

    def describe_sessions(self) -> Dict[str, Any]:
        """汇总当前 session 池状态，供管理端点 / 日志排障使用。"""
        return describe_sessions(self._sessions)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def _index_of_username(self, username: str) -> Optional[int]:
        for i, s in enumerate(self._sessions):
            if s.username == username:
                return i
        return None

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
        """图生视频：委托给 core.media.video.VideoService，避免重复实现聊天/轮询逻辑。"""
        async with aiohttp.ClientSession() as s:
            chat_session = ChatSession(s, lambda: None, lambda: {}, lambda: "")
            video_service = VideoService(s, lambda: None, lambda: {}, chat_session.create, chat_session.cleanup)
            return await video_service.generate(
                prompt, image_url, token, user_id, model=model, size=size,
                image_name=image_name, download=download,
            )

    async def synthesize_tts(
        self,
        text: str,
        token: str,
        model: str = "qwen3-max",
        save_dir: Optional[str] = None,
    ) -> Optional[str]:
        """文本转语音：委托给 core.media.tts.TtsService，避免重复实现聊天/占位消息逻辑。"""
        from core.transport.routes import TTS_DIR

        async with aiohttp.ClientSession() as s:
            chat_session = ChatSession(s, lambda: None, lambda: {}, lambda: "")
            tts_service = TtsService(
                s, lambda: None, lambda: {}, lambda: "",
                chat_session.create, chat_session.send_placeholder_message, chat_session.cleanup,
            )
            return await tts_service.synthesize(text, token, model=model, save_dir=save_dir or TTS_DIR)

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
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TOTAL_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    await handle_chat_error(self, resp, session)
                async for event in iter_sse_events(self, resp, session):
                    yield event
