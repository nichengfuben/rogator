from __future__ import annotations

"""Cursor 上游客户端：Star Cursor 拉号 + Agent 流，无账号池。"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.auth_store import get_access_token, get_token_bundle
from upstream.cursor.config import load_cursor_upstream_config, starcursor_config
from upstream.cursor.models_api import fetch_usable_models
from upstream.cursor.models_store import load_merged, merge_model_lists, write_cache
from upstream.cursor.token_service import CursorTokenService

logger = logging.getLogger("rogator")


def _meta_for(model_id: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "cursor",
        "capabilities": {"chat": True, "thinking": True, "tools": True},
    }


def _model_ids_from_config() -> List[str]:
    cfg = load_cursor_upstream_config()
    models_cfg = cfg.get("models") or {}
    cursor_cfg = cfg.get("cursor") or {}
    ids: set[str] = set()
    default = str(models_cfg.get("default") or cursor_cfg.get("default_model") or "composer-2.5-fast")
    ids.add(default)
    mapping = models_cfg.get("mapping") or {}
    if isinstance(mapping, dict):
        for key, val in mapping.items():
            ids.add(str(key))
            if val:
                ids.add(str(val))
    for mid in models_cfg.get("fallback") or []:
        if mid:
            ids.add(str(mid))
    return sorted(ids)


def _merge_model_inventory(api_models: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    cfg = load_cursor_upstream_config()
    models_cfg = cfg.get("models") or {}
    cursor_cfg = cfg.get("cursor") or {}
    ids: set[str] = set()
    meta: Dict[str, Any] = {}

    for item in api_models:
        mid = str(item.get("modelId") or item.get("model_id") or "")
        if not mid:
            continue
        ids.add(mid)
        meta[mid] = _meta_for(mid)

    default = str(models_cfg.get("default") or cursor_cfg.get("default_model") or "")
    if default:
        ids.add(default)

    mapping = models_cfg.get("mapping") or {}
    if isinstance(mapping, dict):
        for key, val in mapping.items():
            ids.add(str(key))
            if val:
                ids.add(str(val))

    for mid in models_cfg.get("fallback") or []:
        if mid:
            ids.add(str(mid))

    sorted_ids = sorted(ids)
    for mid in sorted_ids:
        meta.setdefault(mid, _meta_for(mid))
    return sorted_ids, meta


class CursorClient:
    UPSTREAM_NAME = "cursor"

    def __init__(self, splitter: Any = None) -> None:
        self._splitter = splitter
        self._tokens = CursorTokenService()
        config_ids = _model_ids_from_config()
        models, meta, updated_at = load_merged(config_ids)
        self._models: List[str] = models
        self._model_meta: Dict[str, Any] = meta
        self._models_fetch_time: float = updated_at
        self._startup_done: bool = False
        self._startup_lock = asyncio.Lock()
        self._conversation_id: Optional[str] = None
        self._workspace: str = os.getcwd()
        cfg = starcursor_config()
        self._poll_interval: float = float(cfg.get("poll_interval", 30))

    def load_models_cache(self) -> List[str]:
        config_ids = _model_ids_from_config()
        models, meta, updated_at = load_merged(config_ids)
        self._models = models
        self._model_meta = meta
        self._models_fetch_time = updated_at
        return list(models)

    def models_refresh_due(self, interval: float) -> bool:
        if interval <= 0:
            return True
        if self._models_fetch_time <= 0:
            return True
        return (time.time() - self._models_fetch_time) >= interval

    def _apply_models(self, ids: List[str], meta: Dict[str, Any], *, persist: bool) -> List[str]:
        config_ids = _model_ids_from_config()
        merged = merge_model_lists(config_ids, ids)
        full_meta = {mid: _meta_for(mid) for mid in merged}
        for mid, item in meta.items():
            if mid in full_meta and isinstance(item, dict):
                full_meta[mid] = {**full_meta[mid], **item}
        self._models = merged
        self._model_meta = full_meta
        self._models_fetch_time = time.time()
        if persist:
            write_cache(merged, full_meta)
        return list(merged)

    def _keep_cached(self) -> List[str]:
        if self._models:
            return list(self._models)
        return self.load_models_cache()

    async def fetch_models(self, *, use_cache: bool = True) -> List[str]:
        from server.config import CONFIG

        if (
            use_cache
            and self._models
            and self._models_fetch_time > 0
            and not self.models_refresh_due(CONFIG.models_refresh_interval)
        ):
            return list(self._models)

        try:
            await self.ensure_token()
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, fetch_usable_models, get_token_bundle())
            if raw:
                ids, meta = _merge_model_inventory(raw)
                logger.info("Cursor GetUsableModels: %d model(s)", len(ids))
                return self._apply_models(ids, meta, persist=True)
        except Exception as exc:
            logger.warning("Cursor GetUsableModels 失败，使用磁盘/配置缓存: %s", exc)

        cached = self._keep_cached()
        if cached:
            return cached

        fallback = _model_ids_from_config()
        return self._apply_models(fallback, {mid: _meta_for(mid) for mid in fallback}, persist=False)

    async def ensure_token(self) -> str:
        token = get_access_token()
        if token:
            return token
        ok = await self._tokens.pull_until_acceptable()
        if not ok:
            raise RuntimeError(
                "Cursor: 无法拉取 Token，请检查 configs/cursor.toml 中 [starcursor].api_keys"
            )
        token = get_access_token()
        if not token:
            raise RuntimeError("Cursor: auth.toml 无 access_token")
        return token

    async def startup(self) -> None:
        async with self._startup_lock:
            if self._startup_done:
                return
            cfg = starcursor_config()
            keys = cfg.get("api_keys") or []
            if not keys:
                logger.warning(
                    "Cursor: configs/cursor.toml 中 [starcursor].api_keys 为空，"
                    "请配置 Star Cursor API Key 后重启"
                )
            elif not get_access_token():
                logger.info("Cursor: 无本地 Token，尝试首次拉号...")
                await self._tokens.pull_until_acceptable()
            else:
                logger.info("Cursor startup: 已有 auth.toml Token")
            self._poll_interval = float(cfg.get("poll_interval", 30))
            try:
                await self.fetch_models(use_cache=False)
            except Exception as exc:
                logger.warning("Cursor startup 拉取模型列表失败: %s", exc)
            self._startup_done = True
            logger.info("Cursor startup: %d model(s), poll_interval=%.0fs", len(self._models), self._poll_interval)

    async def token_maintenance_loop(self, shutdown_event: asyncio.Event) -> None:
        """后台用量监测与自动换号。"""
        wait = max(5.0, self._poll_interval)
        while not shutdown_event.is_set():
            try:
                await self._tokens.auto_check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Cursor token maintenance: %s", exc)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait)
            except asyncio.TimeoutError:
                continue

    async def shutdown(self) -> None:
        await self._tokens.close()
        self._startup_done = False
        logger.debug("Cursor client shut down")
