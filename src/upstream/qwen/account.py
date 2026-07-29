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

from upstream.qwen.accounts import ACCOUNTS, Account
from upstream.qwen.auth.crypto import build_headers, build_login_headers, hash_password
from upstream.qwen.chat.routes import AUTH_BASE_URL, BASE_URL, MODELS_PATH
from server.records.login_history import LoginHistoryStore
from upstream.qwen.chat.store import (
    QwenSession,
    fetch_user_id,
    mask_username,
    replace_or_append,
    valid_session_count,
)
from server.formats import (
    DEFAULT_MODELS,
    LOGIN_TIMEOUT,
    MODELS_CACHE_FILE,
    MODELS_FETCH_TIMEOUT,
)
from server.model.model_meta import (
    ModelMeta,
    merge_model_meta,
    parse_upstream_models_payload,
    read_models_cache_payload,
)

logger = logging.getLogger("rogator")


def merge_model_lists(*parts: List[str]) -> List[str]:
    """DEFAULT 打底，后续列表只增不减（保序去重）。"""
    seen: set[str] = set()
    merged: List[str] = []
    for part in parts:
        for model_id in part:
            if model_id and model_id not in seen:
                seen.add(model_id)
                merged.append(model_id)
    return merged


class SessionLoginMixin:
    _sessions: List[QwenSession]
    _current_index: int
    _blocked_accounts: Dict[str, float]
    _lock: asyncio.Lock
    _prelogin_target: int
    _login_interval: float
    _login_history: LoginHistoryStore

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
                    async with self._lock:
                        replace_or_append(self._sessions, qs)
                    self._login_history.record(account.username)
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
        interval = max(0.0, self._login_interval)
        for attempt in range(need):
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
            if interval > 0 and attempt < need - 1:
                logger.debug("Prelogin waiting %.1fs before next login", interval)
                await asyncio.sleep(interval)
        if logged:
            logger.info(
                "Prelogin done: %d new, %d total ready (target=%d)",
                logged, valid_session_count(self._sessions), target,
            )
        elif valid_session_count(self._sessions) == 0:
            logger.error("All prelogin attempts failed (target=%d)", target)

    async def ensure_prelogin(self) -> None:
        await self.prelogin_accounts(self._prelogin_target)

    def _active_usernames(self) -> set[str]:
        return {s.username for s in self._sessions if s.is_valid and not s.is_expired()}

    def _pick_account_for_login(self, *, skip: Optional[set[str]] = None) -> Optional[Account]:
        if not ACCOUNTS:
            return None
        skip = skip or set()

        def eligible(account: Account) -> bool:
            return (
                account.username not in skip
                and account.username not in self._active_usernames()
                and not self._is_account_blocked(account.username)
            )

        return self._login_history.pick_account(ACCOUNTS, eligible=eligible)

    def _valid_sessions(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> List[QwenSession]:
        return [
            s for s in self._sessions
            if s.is_valid and not s.is_expired()
            and not self._is_account_blocked(s.username)
            and (exclude_username is None or s.username != exclude_username)
        ]

    def _select_valid_session(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> Optional[QwenSession]:
        valid = self._valid_sessions(exclude_username=exclude_username)
        if not valid:
            return None
        selected = random.choice(valid)
        idx = self._index_of_username(selected.username)
        if idx is not None:
            self._current_index = idx
        return selected

    async def switch_to_next(self, exclude_username: Optional[str] = None) -> Optional[QwenSession]:
        skip: set[str] = {exclude_username} if exclude_username else set()

        async with self._lock:
            self.prune_expired_sessions()

        await self._ensure_cleanup()

        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                self._save_meta()
                return session
            account = self._pick_account_for_login(skip=skip)

        if account is None:
            async with self._lock:
                session = self._select_valid_session(exclude_username=exclude_username)
                if session is not None:
                    self._save_meta()
                    return session
            return None

        skip.add(account.username)
        for _ in range(len(ACCOUNTS)):
            qs = await self.login_account(account)
            if qs and qs.username != exclude_username:
                async with self._lock:
                    idx = self._index_of_username(qs.username)
                    self._current_index = idx if idx is not None else 0
                return qs
            async with self._lock:
                account = self._pick_account_for_login(skip=skip)
            if account is None:
                break
            skip.add(account.username)

        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                self._save_meta()
                return session
        return None

    async def get_valid_session(self, *, exclude_username: Optional[str] = None) -> Optional[QwenSession]:
        self.prune_expired_sessions()
        await self._ensure_cleanup()
        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                return session
        return await self.switch_to_next(exclude_username=exclude_username)

    def prune_expired_sessions(self) -> List[str]: ...


class ModelsFetchMixin:
    _models: List[str]
    _model_meta: Dict[str, ModelMeta]
    _models_fetch_time: float

    def models_refresh_due(self, interval: float) -> bool:
        """距上次成功刷新是否已超过 interval（秒）；无 timestamp 视为需要刷新。"""
        if interval <= 0:
            return True
        if self._models_fetch_time <= 0:
            return True
        return (time.time() - self._models_fetch_time) >= interval

    async def fetch_models(self, use_cache: bool = True) -> List[str]:
        await self._ensure_cleanup()
        now = time.time()
        from server.config import CONFIG

        if use_cache and self._models and not self.models_refresh_due(CONFIG.models_refresh_interval):
            return list(self._models)

        def _keep_cached() -> List[str]:
            if self._models:
                return list(self._models)
            return self.load_models_cache()

        session = await self.get_valid_session()
        if not session:
            return _keep_cached()
        try:
            async with aiohttp.ClientSession() as s:
                headers = build_headers(session.token)
                headers["Accept"] = "application/json"
                async with s.get(
                    f"{BASE_URL}{MODELS_PATH}", headers=headers, ssl=False,
                    timeout=aiohttp.ClientTimeout(total=MODELS_FETCH_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        return _keep_cached()
                    data = await resp.json()
                    remote_meta = parse_upstream_models_payload(data)
                    remote_ids = list(remote_meta.keys())
                    if remote_ids:
                        merged = merge_model_lists(
                            list(DEFAULT_MODELS),
                            self._models,
                            remote_ids,
                        )
                        self._model_meta = merge_model_meta(
                            merged,
                            self._model_meta,
                            remote_meta,
                        )
                        self._models = merged
                        self._models_fetch_time = now
                        self._save_models_cache(merged, self._model_meta)
                        return merged
            return _keep_cached()
        except Exception:
            return _keep_cached()

    def load_models_cache(self) -> List[str]:
        disk_models: List[str] = []
        disk_meta: Dict[str, ModelMeta] = {}
        updated_at = 0.0
        try:
            p = Path(MODELS_CACHE_FILE)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                disk_models, disk_meta, updated_at = read_models_cache_payload(data)
        except Exception:
            pass
        merged = merge_model_lists(list(DEFAULT_MODELS), disk_models)
        self._models = merged
        self._model_meta = merge_model_meta(merged, disk_meta)
        self._models_fetch_time = updated_at
        return list(merged)

    def _save_models_cache(self, models: List[str], meta: Dict[str, ModelMeta]) -> None:
        try:
            p = Path(MODELS_CACHE_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "models": models,
                    "meta": {mid: meta[mid].to_dict() for mid in models if mid in meta},
                    "updated_at": int(time.time()),
                },
                ensure_ascii=False,
                indent=2,
            )
            p.write_text(payload, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to save models cache: %s", exc)
