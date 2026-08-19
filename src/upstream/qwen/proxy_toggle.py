from __future__ import annotations

"""Qwen 代理自动开关管理器。

根据 SM block（DataInspectionFailed）自动切换代理状态，
通过持久化文件记住上次状态，启动时测活验证代理可用性。
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger("rogator")

_PERSIST_PATH = Path("persist/qwen/proxy_toggle.json")
_PROBE_URL = "https://ip.sb/ip"
_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=8)


def _has_proxy_env() -> bool:
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(key, "").strip():
            return True
    return False


async def _probe_proxy_alive() -> bool:
    try:
        async with aiohttp.ClientSession(trust_env=True) as s:
            async with s.get(
                _PROBE_URL, ssl=False, timeout=_PROBE_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    return False
                text = (await resp.text()).strip()
                return len(text) > 0
    except Exception as exc:
        logger.debug("proxy probe failed: %s", exc)
        return False


class ProxyToggleManager:
    """Qwen 代理自动开关：持久化 + 测活 + SM block 防抖切换。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._enabled: bool = False
        self._initialized: bool = False
        self._seen_tasks: set[str] = set()

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not _has_proxy_env():
            self._enabled = False
            self._initialized = True
            logger.info("proxy toggle: no proxy env, disabled")
            return
        alive = await _probe_proxy_alive()
        if not alive:
            self._enabled = False
            self._initialized = True
            logger.info("proxy toggle: proxy env set but probe failed, disabled")
            return
        self._enabled = self._read_persist()
        self._initialized = True
        logger.info("proxy toggle: initialized enabled=%s", self._enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_sm_block(self, req_id: str) -> bool:
        async with self._lock:
            if req_id in self._seen_tasks:
                return self._enabled
            self._seen_tasks.add(req_id)
            self._enabled = not self._enabled
            self._write_persist()
            logger.info(
                "proxy toggle: sm block req=%s switched to enabled=%s",
                req_id[:12], self._enabled,
            )
            return self._enabled

    def release_task(self, req_id: str) -> None:
        self._seen_tasks.discard(req_id)

    def _read_persist(self) -> bool:
        try:
            if _PERSIST_PATH.exists():
                data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
                return bool(data.get("enabled", 0))
        except Exception as exc:
            logger.warning("proxy toggle: read persist failed: %s", exc)
        self._write_persist()
        return False

    def _write_persist(self) -> None:
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PERSIST_PATH.write_text(
                json.dumps({"enabled": int(self._enabled)}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("proxy toggle: write persist failed: %s", exc)


_manager: Optional[ProxyToggleManager] = None

import contextvars

_current_req_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "qwen_current_req_id", default="",
)


def get_proxy_toggle() -> ProxyToggleManager:
    global _manager
    if _manager is None:
        _manager = ProxyToggleManager()
    return _manager


def set_current_req_id(req_id: str) -> contextvars.Token:
    return _current_req_id.set(req_id)


def get_current_req_id() -> str:
    return _current_req_id.get()


def schedule_sm_block_toggle(req_id: str) -> None:
    """同步入口：在 raise_sse_inline_error 等同步上下文中调度异步切换。"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(get_proxy_toggle().on_sm_block(req_id))
    except RuntimeError:
        pass
