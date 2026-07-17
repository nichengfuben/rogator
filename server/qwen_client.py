from __future__ import annotations

"""Qwen session and client implementation。"""

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp

from accounts import ACCOUNTS, Account
from core.crypto.crypto import build_headers, build_login_headers, hash_password
from core.transport.endpoints import (
    AUTH_BASE_URL,
    BASE_URL,
    CHAT_PATH,
    MODELS_PATH,
    NEW_CHAT_PATH,
)
from core.transport.chat_session import ChatSession
from core.transport.sse import parse_sse_event
from core.media.tts import TtsService
from core.media.video import VideoService
from server.session_store import (
    CLEANUP_INTERVAL,
    QwenSession,
    clean_expired,
    describe_sessions,
    fetch_user_id,
    load_sessions,
    mark_invalid as mark_invalid_in,
    save_sessions,
)
from server.upload_mixin import UploadMixin
from server.formats import (
    DEFAULT_MODELS,
    DEFAULT_USER_AGENT,
    LOGIN_TIMEOUT,
    MODELS_CACHE_FILE,
    MODELS_FETCH_TIMEOUT,
    REQUEST_TOTAL_TIMEOUT,
    TokenExpiredError,
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
)

logger = logging.getLogger("rogator")


class QwenClient(UploadMixin):
    def __init__(self, splitter: Any) -> None:
        self._splitter = splitter
        self._sessions: List[QwenSession] = load_sessions()
        self._current_index: int = 0
        self._lock = asyncio.Lock()
        self._models: List[str] = list(DEFAULT_MODELS)
        self._models_fetch_time: float = 0
        self._models_cache_ttl: float = 300
        self._last_cleanup: float = 0.0

    async def _ensure_cleanup(self) -> None:
        """按 CLEANUP_INTERVAL 节流批量清理过期/失效 session。"""
        now = time.time()
        if now - self._last_cleanup < CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        self._sessions, removed = clean_expired(self._sessions)
        if removed:
            if self._current_index >= len(self._sessions):
                self._current_index = 0
            save_sessions(self._sessions)

    def mark_invalid(self, username: str) -> bool:
        """按 username 精确标记单个 session 失效并立即持久化。"""
        found = mark_invalid_in(self._sessions, username)
        if found:
            save_sessions(self._sessions)
        return found

    def mark_invalid_current(self) -> None:
        session = self.current_session
        if session:
            self.mark_invalid(session.username)

    def describe_sessions(self) -> Dict[str, Any]:
        """汇总当前 session 池状态，供管理端点 / 日志排障使用。"""
        return describe_sessions(self._sessions)

    @property
    def current_session(self) -> Optional[QwenSession]:
        if not self._sessions or self._current_index >= len(self._sessions):
            return None
        return self._sessions[self._current_index]

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    async def login_account(self, account: Account) -> Optional[QwenSession]:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"email": account.username, "password": hash_password(account.password), "remember_me": True}
                async with session.post(
                    f"{AUTH_BASE_URL}/api/v2/auths/signin", json=payload,
                    headers=build_login_headers(), ssl=False,
                    timeout=aiohttp.ClientTimeout(total=LOGIN_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Login %s HTTP %d", account.username[:6], resp.status)
                        return None
                    data = await resp.json()
                    if not data.get("success"):
                        logger.warning("Login %s failed: %s", account.username[:6], data.get("message", ""))
                        return None
                    token = str((data.get("data") or {}).get("access_token", ""))
                    if not token:
                        logger.warning("Login %s no token", account.username[:6])
                        return None
                    user_id = await fetch_user_id(session, token, AUTH_BASE_URL)
                    qs = QwenSession(account=account, token=token, user_id=user_id or account.username[:12])
                    logger.info("Logged in: %s", account.username[:6])
                    return qs
        except asyncio.TimeoutError:
            logger.warning("Login %s timed out", account.username[:6])
            return None
        except Exception as e:
            logger.warning("Login exception for %s: %s", account.username[:6], e)
            return None

    async def prelogin_accounts(self, count: int) -> None:
        await self._ensure_cleanup()
        if not ACCOUNTS:
            logger.warning("No accounts available")
            return
        existing_usernames = {s.username for s in self._sessions}
        shuffled = [a for a in ACCOUNTS if a.username not in existing_usernames]
        random.shuffle(shuffled)
        new_sessions: List[QwenSession] = []
        need = max(0, count - len(self._sessions))
        for i, account in enumerate(shuffled):
            if len(new_sessions) >= need:
                break
            logger.info("Trying account %d/%d (%s)...", i + 1, need, account.username[:6])
            qs = await self.login_account(account)
            if qs:
                new_sessions.append(qs)
                logger.info("Account %s OK", account.username[:6])
            else:
                logger.warning("Account %s failed, skipping", account.username[:6])
        if new_sessions:
            self._sessions.extend(new_sessions)
            self._current_index = 0
            save_sessions(self._sessions)
            logger.info("Prelogin done: %d new, %d total ready", len(new_sessions), len(self._sessions))
        elif not self._sessions:
            logger.error("All %d login attempts failed", count)

    async def switch_to_next(self) -> Optional[QwenSession]:
        async with self._lock:
            if not self._sessions:
                for account in ACCOUNTS:
                    qs = await self.login_account(account)
                    if qs:
                        self._sessions.append(qs)
                        self._current_index = 0
                        save_sessions(self._sessions)
                        return qs
                return None

            if self._current_index < len(self._sessions):
                self._sessions[self._current_index].is_valid = False

            valid = [i for i, s in enumerate(self._sessions) if s.is_valid]
            if valid:
                idx = random.choice(valid)
                self._current_index = idx
                save_sessions(self._sessions)
                return self._sessions[idx]

            tried_usernames = {s.account.username for s in self._sessions}
            for account in ACCOUNTS:
                if account.username in tried_usernames:
                    continue
                qs = await self.login_account(account)
                if qs:
                    self._sessions[self._current_index] = qs
                    save_sessions(self._sessions)
                    return qs

            current_account = self._sessions[self._current_index].account
            qs = await self.login_account(current_account)
            if qs:
                self._sessions[self._current_index] = qs
                save_sessions(self._sessions)
                return qs

            return None

    async def get_valid_session(self) -> Optional[QwenSession]:
        await self._ensure_cleanup()
        session = self.current_session
        if session and session.is_valid:
            return session
        return await self.switch_to_next()

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
        from core.transport.endpoints import TTS_DIR

        async with aiohttp.ClientSession() as s:
            chat_session = ChatSession(s, lambda: None, lambda: {}, lambda: "")
            tts_service = TtsService(
                s, lambda: None, lambda: {}, lambda: "",
                chat_session.create, chat_session.send_placeholder_message, chat_session.cleanup,
            )
            return await tts_service.synthesize(text, token, model=model, save_dir=save_dir or TTS_DIR)

    async def fetch_models(self, use_cache: bool = True) -> List[str]:
        now = time.time()
        if use_cache and self._models and (now - self._models_fetch_time) < self._models_cache_ttl:
            return self._models
        session = await self.get_valid_session()
        if not session:
            return list(DEFAULT_MODELS)
        try:
            async with aiohttp.ClientSession() as s:
                headers = build_headers(session.token)
                headers["Accept"] = "application/json"
                async with s.get(
                    f"{BASE_URL}{MODELS_PATH}", headers=headers, ssl=False,
                    timeout=aiohttp.ClientTimeout(total=MODELS_FETCH_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        return list(DEFAULT_MODELS)
                    data = await resp.json()
                    models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
                    if models:
                        self._models = models
                        self._models_fetch_time = now
                        return models
            return list(DEFAULT_MODELS)
        except Exception:
            return list(DEFAULT_MODELS)

    def load_models_cache(self) -> List[str]:
        try:
            from pathlib import Path
            p = Path(MODELS_CACHE_FILE)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                models = data.get("models", [])
                if models:
                    self._models = models
                    return models
        except Exception:
            pass
        return list(DEFAULT_MODELS)

    def _check_create_chat_error(self, session: QwenSession, data: Dict[str, Any]) -> None:
        data_obj = data.get("data") or {}
        if not isinstance(data_obj, dict):
            raise RuntimeError(f"Create chat failed: {data}")
        code = str(data_obj.get("code", "")).lower()
        details = str(data_obj.get("details", "")).lower()
        if code == "unauthorized" or "token" in details or "expired" in details or "log in" in details:
            session.is_valid = False
            raise TokenExpiredError(f"Token expired: {data_obj.get('details', '')}")
        raise RuntimeError(f"Create chat failed: {data}")

    async def create_chat(self, session: QwenSession, model: str) -> str:
        async with aiohttp.ClientSession() as s:
            payload = {
                "title": "新建对话", "models": [model], "chat_mode": "local",
                "chat_type": "t2t", "timestamp": int(time.time() * 1000), "project_id": "",
            }
            headers = build_headers(session.token, include_version=False)
            async with s.post(
                f"{BASE_URL}{NEW_CHAT_PATH}", json=payload, headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    if resp.status in (401, 403):
                        session.is_valid = False
                        raise TokenExpiredError(f"Token expired: HTTP {resp.status}")
                    raise RuntimeError(f"Create chat HTTP {resp.status}")
                data = await resp.json()
                if not data.get("success"):
                    self._check_create_chat_error(session, data)
                chat_id = str((data.get("data") or {}).get("id", ""))
                if not chat_id:
                    raise RuntimeError(f"Create chat failed: no chat_id in {data}")
                return chat_id

    async def chat_completion(
        self,
        session: QwenSession,
        chat_id: str,
        messages: List[Dict[str, Any]],
        model: str = "qwen3.7-max",
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not messages:
            raise ValueError("messages cannot be empty")
        user_content = messages[0].get("content", "")
        if not user_content:
            user_content = extract_last_user_content(messages)
        qwen_message = build_qwen_message(user_content, model, files)
        payload = build_chat_payload(chat_id, model, qwen_message)
        headers = build_headers(session.token, chat_id=chat_id, include_sse=True)
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TOTAL_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    await _handle_chat_error(resp, session)
                async for event in _iter_sse_events(resp, session):
                    yield event


async def _handle_chat_error(resp, session):
    if resp.status in (401, 403):
        session.is_valid = False
        raise TokenExpiredError(f"Token expired: HTTP {resp.status}")
    body = await resp.text()
    if "RateLimited" in body or "daily usage" in body:
        session.is_valid = False
        logger.warning("Session %s rate limited", session.username[:6])
        raise TokenExpiredError(f"Rate limited: {body[:200]}")
    logger.error("Chat HTTP %d: %s", resp.status, body[:500])
    raise RuntimeError(f"Chat HTTP {resp.status}: {body[:200]}")


def _check_error_line(line: str, session) -> None:
    if not (line.startswith("{") and "success" in line):
        return
    err = json.loads(line)
    if err.get("success", True):
        return
    msg = json.dumps(err, ensure_ascii=False)
    if "RateLimited" in msg or "daily usage" in msg:
        session.is_valid = False
        raise TokenExpiredError(f"Rate limited: {msg}")
    raise RuntimeError(f"Qwen API error: {msg}")


async def _iter_sse_events(resp, session):
    async for raw in resp.content:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            _check_error_line(line, session)
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        event = parse_sse_event(data_str)
        if event:
            yield event
