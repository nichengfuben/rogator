from __future__ import annotations

"""QwenClient 登录、预登、session 切换与模型列表拉取。"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from accounts import ACCOUNTS, Account
from core.crypto.crypto import build_headers, build_login_headers, hash_password
from core.transport.routes import AUTH_BASE_URL, BASE_URL, MODELS_PATH
from server.client.session_store import (
    QwenSession,
    fetch_user_id,
    mask_username,
    replace_or_append,
    valid_session_count,
)
from server.formats import DEFAULT_MODELS, LOGIN_TIMEOUT, MODELS_CACHE_FILE, MODELS_FETCH_TIMEOUT

logger = logging.getLogger("rogator")


class SessionLoginMixin:
    _sessions: List[QwenSession]
    _current_index: int
    _account_index: int
    _blocked_accounts: Dict[str, float]
    _lock: asyncio.Lock
    _prelogin_target: int

    def _save_meta(self) -> List[str]: ...

    def _persist_sessions(self) -> List[str]: ...

    async def _ensure_cleanup(self) -> None: ...

    def _is_account_blocked(self, username: str) -> bool: ...

    def _index_of_username(self, username: str) -> Optional[int]: ...

    @property
    def current_session(self) -> Optional[QwenSession]: ...

    async def login_account(self, account: Account) -> Optional[QwenSession]:
        await self._ensure_cleanup()
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
                    replace_or_append(self._sessions, qs)
                    self._persist_sessions()
                    logger.info("Logged in: %s (total: %d)", account.username[:6], len(self._sessions))
                    return qs
        except asyncio.TimeoutError:
            logger.warning("Login %s timed out", account.username[:6])
            return None
        except Exception as e:
            logger.warning("Login exception for %s: %s", account.username[:6], e)
            return None

    async def prelogin_accounts(self, count: Optional[int] = None) -> None:
        target = self._prelogin_target if count is None else count
        await self._ensure_cleanup()
        if not ACCOUNTS:
            logger.warning("No accounts available")
            return
        need = max(0, target - valid_session_count(self._sessions))
        if need <= 0:
            return
        logged = 0
        for _ in range(need):
            account = self._pick_account_for_login()
            if account is None:
                break
            logger.info("Prelogin trying %s...", mask_username(account.username))
            qs = await self.login_account(account)
            if qs:
                logged += 1
                logger.info("Prelogin account %s OK", mask_username(account.username))
            else:
                logger.warning("Prelogin account %s failed", mask_username(account.username))
        if logged:
            logger.info(
                "Prelogin done: %d new, %d total ready (target=%d)",
                logged, valid_session_count(self._sessions), target,
            )
        elif valid_session_count(self._sessions) == 0:
            logger.error("All prelogin attempts failed (target=%d)", target)

    async def ensure_prelogin(self) -> None:
        await self.prelogin_accounts(self._prelogin_target)

    def _pick_account_for_login(self, *, skip: Optional[set[str]] = None) -> Optional[Account]:
        if not ACCOUNTS:
            return None
        skip = skip or set()
        active = {s.username for s in self._sessions if s.is_valid and not s.is_expired()}
        n = len(ACCOUNTS)
        for offset in range(n):
            idx = (self._account_index + offset) % n
            account = ACCOUNTS[idx]
            if account.username in skip:
                continue
            if account.username in active:
                continue
            if self._is_account_blocked(account.username):
                continue
            self._account_index = (idx + 1) % n
            return account
        return None

    async def switch_to_next(self, exclude_username: Optional[str] = None) -> Optional[QwenSession]:
        async with self._lock:
            self.prune_expired_sessions()
            await self._ensure_cleanup()

            valid_indices = [
                i for i, s in enumerate(self._sessions)
                if s.is_valid and not s.is_expired()
                and (exclude_username is None or s.username != exclude_username)
            ]
            if valid_indices:
                idx = random.choice(valid_indices)
                self._current_index = idx
                self._save_meta()
                return self._sessions[idx]

            skip = {exclude_username} if exclude_username else set()
            for _ in range(len(ACCOUNTS)):
                account = self._pick_account_for_login(skip=skip)
                if account is None:
                    break
                qs = await self.login_account(account)
                if qs and qs.username != exclude_username:
                    idx = self._index_of_username(qs.username)
                    self._current_index = idx if idx is not None else 0
                    return qs
                skip.add(account.username)

            await self.ensure_prelogin()
            valid_indices = [
                i for i, s in enumerate(self._sessions)
                if s.is_valid and not s.is_expired()
                and (exclude_username is None or s.username != exclude_username)
            ]
            if valid_indices:
                idx = random.choice(valid_indices)
                self._current_index = idx
                self._save_meta()
                return self._sessions[idx]
            return None

    async def get_valid_session(self) -> Optional[QwenSession]:
        self.prune_expired_sessions()
        await self._ensure_cleanup()
        if valid_session_count(self._sessions) < self._prelogin_target:
            await self.ensure_prelogin()
        session = self.current_session
        if session:
            return session
        valid = [s for s in self._sessions if s.is_valid and not s.is_expired()]
        if valid:
            selected = random.choice(valid)
            idx = self._index_of_username(selected.username)
            if idx is not None:
                self._current_index = idx
                self._save_meta()
            return selected
        return await self.switch_to_next()

    def prune_expired_sessions(self) -> List[str]: ...


class ModelsFetchMixin:
    _models: List[str]
    _models_fetch_time: float
    _models_cache_ttl: float

    async def fetch_models(self, use_cache: bool = True) -> List[str]:
        await self._ensure_cleanup()
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
