from __future__ import annotations

"""Qwen session and client implementation."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
from core.transport.sse import parse_sse_event
from server.oss_upload import upload_to_oss
from server.formats import (
    DEFAULT_MODELS,
    DEFAULT_USER_AGENT,
    LOGIN_TIMEOUT,
    MODELS_CACHE_FILE,
    MODELS_FETCH_TIMEOUT,
    REQUEST_TOTAL_TIMEOUT,
    TOKEN_EXPIRE_SECONDS,
    TokenExpiredError,
    extract_last_user_content,
)

logger = logging.getLogger("rogator")


@dataclass
class QwenSession:
    account: Account
    token: str
    user_id: str
    login_time: float = field(default_factory=time.time)
    is_valid: bool = True

    @property
    def username(self) -> str:
        return self.account.username

    def is_expired(self) -> bool:
        """检查 token 是否��过 12 小时"""
        return time.time() - self.login_time > TOKEN_EXPIRE_SECONDS


class QwenClient:
    def __init__(self, splitter: Any) -> None:
        self._splitter = splitter
        self._sessions: List[QwenSession] = []
        self._current_index: int = 0
        self._lock = asyncio.Lock()
        self._models: List[str] = list(DEFAULT_MODELS)
        self._models_fetch_time: float = 0
        self._models_cache_ttl: float = 300

    @property
    def current_session(self) -> Optional[QwenSession]:
        if not self._sessions or self._current_index >= len(self._sessions):
            return None
        return self._sessions[self._current_index]

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def _clean_expired(self) -> List[str]:
        """清理过期和无效的 session，返回被移除的 username ��表"""
        removed = []
        valid_sessions = []
        for s in self._sessions:
            if s.is_expired():
                logger.debug("Session %s expired (login_time: %s), removing",
                             s.username[:6], s.login_time)
                removed.append(s.username)
            elif not s.is_valid:
                logger.debug("Session %s invalid, removing", s.username[:6])
                removed.append(s.username)
            else:
                valid_sessions.append(s)
        if len(valid_sessions) != len(self._sessions):
            self._sessions = valid_sessions
        return removed

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
                    user_id = await self._fetch_user_id(session, token)
                    qs = QwenSession(account=account, token=token, user_id=user_id or account.username[:12])
                    logger.info("Logged in: %s", account.username[:6])
                    return qs
        except asyncio.TimeoutError:
            logger.warning("Login %s timed out", account.username[:6])
            return None
        except Exception as e:
            logger.warning("Login exception for %s: %s", account.username[:6], e)
            return None

    async def _fetch_user_id(self, session: aiohttp.ClientSession, token: str) -> str:
        try:
            async with session.get(
                f"{AUTH_BASE_URL}/api/v2/user",
                headers={"Authorization": f"Bearer {token}", "User-Agent": DEFAULT_USER_AGENT},
                ssl=False,
            ) as ur:
                if ur.status == 200:
                    return str((await ur.json()).get("data", {}).get("id", ""))
        except Exception:
            pass
        return ""

    async def prelogin_accounts(self, count: int) -> None:
        if not ACCOUNTS:
            logger.warning("No accounts available")
            return
        import random
        shuffled = list(ACCOUNTS)
        random.shuffle(shuffled)
        sessions: List[QwenSession] = []
        for i, account in enumerate(shuffled):
            if len(sessions) >= count:
                break
            logger.info("Trying account %d/%d (%s)...", i + 1, count, account.username[:6])
            qs = await self.login_account(account)
            if qs:
                sessions.append(qs)
                logger.info("Account %s OK", account.username[:6])
            else:
                logger.warning("Account %s failed, skipping", account.username[:6])
        if sessions:
            self._sessions = sessions
            self._current_index = 0
            logger.info("Prelogin done: %d/%d ready", len(sessions), count)
        else:
            logger.error("All %d login attempts failed", count)

    async def switch_to_next(self) -> Optional[QwenSession]:
        async with self._lock:
            if not self._sessions:
                for account in ACCOUNTS:
                    qs = await self.login_account(account)
                    if qs:
                        self._sessions.append(qs)
                        self._current_index = 0
                        return qs
                return None

            if self._current_index < len(self._sessions):
                self._sessions[self._current_index].is_valid = False

            start = (self._current_index + 1) % len(self._sessions)
            for i in range(len(self._sessions)):
                idx = (start + i) % len(self._sessions)
                if self._sessions[idx].is_valid:
                    self._current_index = idx
                    return self._sessions[idx]

            tried_usernames = {s.account.username for s in self._sessions}
            for account in ACCOUNTS:
                if account.username in tried_usernames:
                    continue
                qs = await self.login_account(account)
                if qs:
                    self._sessions[self._current_index] = qs
                    return qs

            current_account = self._sessions[self._current_index].account
            qs = await self.login_account(current_account)
            if qs:
                self._sessions[self._current_index] = qs
                return qs

            return None

    async def get_valid_session(self) -> Optional[QwenSession]:
        session = self.current_session
        if session and session.is_valid:
            return session
        return await self.switch_to_next()

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
                    data_obj = data.get("data") or {}
                    if isinstance(data_obj, dict):
                        code = str(data_obj.get("code", "")).lower()
                        details = str(data_obj.get("details", "")).lower()
                        if code == "unauthorized" or "token" in details or "expired" in details or "log in" in details:
                            session.is_valid = False
                            raise TokenExpiredError(f"Token expired: {data_obj.get('details', '')}")
                    raise RuntimeError(f"Create chat failed: {data}")
                chat_id = str((data.get("data") or {}).get("id", ""))
                if not chat_id:
                    raise RuntimeError(f"Create chat failed: no chat_id in {data}")
                return chat_id

    async def _get_sts_credentials(self, session: QwenSession, filename: str, filesize: int) -> Dict[str, Any]:
        headers = build_headers(session.token)
        headers.update({"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"})
        payload = {"filename": filename, "filesize": filesize, "filetype": "file"}
        for path in ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        f"{BASE_URL}{path}", json=payload, headers=headers, ssl=False,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            creds = data.get("data", data)
                            if all(k in creds for k in ("access_key_id", "access_key_secret", "security_token")):
                                return creds
            except Exception:
                continue
        raise RuntimeError("All STS endpoints failed")

    async def upload_file(self, session: QwenSession, file_data: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        creds = await self._get_sts_credentials(session, filename, len(file_data))
        file_url = await upload_to_oss(file_data, "text/plain", creds)
        file_obj = {
            "id": str(creds.get("file_id", uuid.uuid4())), "name": filename,
            "type": "file", "size": len(file_data), "url": file_url,
            "file_type": "text/plain", "showType": "file", "file_class": "document",
            "user_id": session.user_id, "isQuote": False,
        }
        return file_url, file_obj

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


async def _iter_sse_events(resp, session):
    async for raw in resp.content:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            if line.startswith("{") and "success" in line:
                err = json.loads(line)
                if not err.get("success", True):
                    msg = json.dumps(err, ensure_ascii=False)
                    if "RateLimited" in msg or "daily usage" in msg:
                        session.is_valid = False
                        raise TokenExpiredError(f"Rate limited: {msg}")
                    raise RuntimeError(f"Qwen API error: {msg}")
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        event = parse_sse_event(data_str)
        if event:
            yield event


def build_qwen_message(
    user_content: str, model: str,
    files: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "fid": str(uuid.uuid4()), "parentId": None, "childrenIds": [str(uuid.uuid4())],
        "role": "user", "content": user_content, "user_action": "chat",
        "files": files or [], "timestamp": int(time.time() * 1000), "models": [model],
        "chat_type": "t2t", "feature_config": {
            "thinking_enabled": True, "output_schema": "phase", "research_mode": "normal",
            "auto_thinking": False, "thinking_mode": "Thinking", "thinking_format": "raw",
            "auto_search": False,
        }, "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t",
    }


def build_chat_payload(
    chat_id: str, model: str, qwen_message: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stream": True, "version": "2.1", "incremental_output": True,
        "chat_id": chat_id, "chat_mode": "local", "model": model, "parent_id": None,
        "messages": [qwen_message], "timestamp": int(time.time() * 1000),
    }
