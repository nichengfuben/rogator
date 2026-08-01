from __future__ import annotations

"""Qwen account pool and model fetch."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from core.session.accounts import Account
from core.session.pool import SessionLoginMixin
from core.session.store import PlatformSession
from upstream.qwen.auth.crypto import build_headers, build_login_headers, hash_password
from upstream.qwen.chat.routes import AUTH_BASE_URL, BASE_URL, MODELS_PATH
from upstream.qwen.chat.store import fetch_user_id
from upstream.qwen.auth.http import run_with_connection_retry
from core.transport.http import upstream_timeout
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
    seen: set[str] = set()
    merged: List[str] = []
    for part in parts:
        for model_id in part:
            if model_id and model_id not in seen:
                seen.add(model_id)
                merged.append(model_id)
    return merged


class QwenLoginMixin(SessionLoginMixin):
    UPSTREAM_NAME = "qwen"

    async def _perform_login(self, account: Account) -> Optional[PlatformSession]:
        async def _run() -> Optional[PlatformSession]:
            # 每次尝试取新 session，避免 reset 后仍持有已关闭引用
            session = await self._ensure_http_session()
            payload = {
                "email": account.username,
                "password": hash_password(account.password),
                "remember_me": True,
            }
            async with session.post(
                f"{AUTH_BASE_URL}/api/v2/auths/signin",
                json=payload,
                headers=build_login_headers(),
                timeout=upstream_timeout(LOGIN_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Login %s HTTP %d", account.username[:6], resp.status)
                    return None
                data = await resp.json()
                if not data.get("success"):
                    logger.warning(
                        "Login %s failed: %s",
                        account.username[:6],
                        data.get("message", ""),
                    )
                    return None
                token = str((data.get("data") or {}).get("access_token", ""))
                if not token:
                    logger.warning("Login %s no token", account.username[:6])
                    return None
                user_id = await fetch_user_id(session, token, AUTH_BASE_URL)
                return PlatformSession(
                    account=account,
                    token=token,
                    user_id=user_id or account.username[:12],
                    upstream="qwen",
                )
        try:
            return await run_with_connection_retry("login", _run, transport_owner=self)
        except asyncio.TimeoutError:
            reset = getattr(self, "reset_http_transport", None)
            if callable(reset):
                await reset()
            logger.warning(
                "Login %s timed out after %.0fs (transport reset)",
                account.username[:6],
                LOGIN_TIMEOUT,
            )
            return None


class ModelsFetchMixin:
    _models: List[str]
    _model_meta: Dict[str, ModelMeta]
    _models_fetch_time: float

    def models_refresh_due(self, interval: float) -> bool:
        if interval <= 0:
            return True
        if self._models_fetch_time <= 0:
            return True
        return (time.time() - self._models_fetch_time) >= interval

    async def _fetch_models_remote(self, s, session, now: float, keep_cached) -> List[str]:
        headers = build_headers(session.token)
        headers["Accept"] = "application/json"
        async with s.get(
            f"{BASE_URL}{MODELS_PATH}",
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=MODELS_FETCH_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return keep_cached()
            data = await resp.json()
            remote_meta = parse_upstream_models_payload(data)
            remote_ids = list(remote_meta.keys())
            if not remote_ids:
                return keep_cached()
            merged = merge_model_lists(list(DEFAULT_MODELS), self._models, remote_ids)
            self._model_meta = merge_model_meta(merged, self._model_meta, remote_meta)
            self._models = merged
            self._models_fetch_time = now
            self._save_models_cache(merged, self._model_meta)
            return merged

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
            async def _run() -> List[str]:
                s = await self._ensure_http_session()
                return await self._fetch_models_remote(s, session, now, _keep_cached)

            return await run_with_connection_retry(
                "fetch_models", _run, transport_owner=self,
            )
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
                if updated_at <= 0:
                    updated_at = float(p.stat().st_mtime)
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
