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
        async with self._lock:
            if self._initialized:
                logger.debug("proxy toggle: already initialized, skipping")
                return
            logger.debug("proxy toggle: starting initialize...")
            if not _has_proxy_env():
                self._enabled = False
                self._initialized = True
                logger.info("proxy toggle: no proxy env, disabled")
                return
            logger.debug("proxy toggle: proxy env detected, starting proxy probe...")
            alive = await _probe_proxy_alive()
            logger.debug("proxy toggle: probe result: alive=%s", alive)
            # 先读取用户持久化保存的状态，探针仅用于判断代理是否实际可用
            # 避免因为启动瞬间网络波动/代理刚启动导致探针失败，覆盖用户之前保存的 enabled=1
            persisted_enabled = self._read_persist()
            logger.debug("proxy toggle: read persist: persisted_enabled=%s (None=file not exists)", persisted_enabled)
            if not alive:
                logger.debug("proxy toggle: probe failed, handling branch...")
                # 探针失败: 持久化文件不存在 → 安全默认禁用
                if persisted_enabled is None:
                    self._enabled = False
                    self._write_persist()
                    self._initialized = True
                    logger.info("proxy toggle: proxy env set but probe failed, disabled")
                    return
                # 持久化文件存在但里面enabled是0 → 保持禁用
                if not persisted_enabled:
                    self._enabled = False
                    self._initialized = True
                    logger.info("proxy toggle: proxy env set but probe failed, disabled")
                    return
                # 持久化文件存在且用户显式设了 enabled=1 → 尊重用户选择忽略探针失败
                self._enabled = True
                self._initialized = True
                logger.warning("proxy toggle: probe failed but respect persisted enabled=1, proxy will still be used")
                return
            logger.debug("proxy toggle: probe succeeded")
            # 探针成功，正常读取持久化状态
            if persisted_enabled is None:
                self._enabled = False
                # 持久化文件不存在，把默认值写回
                self._write_persist()
            else:
                self._enabled = persisted_enabled
            self._initialized = True
            logger.info("proxy toggle: initialized enabled=%s", self._enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_sm_block(self, req_id: str, used_enabled: bool) -> bool:
        """SM block 触发代理切换：基于请求时的状态计算新值，避免并发翻转回原点。"""
        async with self._lock:
            if req_id in self._seen_tasks:
                return self._enabled
            self._seen_tasks.add(req_id)
            new_value = not used_enabled
            if new_value == self._enabled:
                logger.info(
                    "proxy toggle: sm block req=%s target enabled=%s already active, skip",
                    req_id[:12], new_value,
                )
                return self._enabled
            self._enabled = new_value
            self._write_persist()
            logger.info(
                "proxy toggle: sm block req=%s used_enabled=%s → enabled=%s",
                req_id[:12], used_enabled, self._enabled,
            )
            return self._enabled

    def release_task(self, req_id: str) -> None:
        self._seen_tasks.discard(req_id)

    def _read_persist(self) -> bool:
        if not _PERSIST_PATH.exists():
            # 没有持久化文件，返回 None 标记不存在，让调用方处理默认值
            return None  # type: ignore
        try:
            data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
            return bool(data.get("enabled", 0))
        except Exception as exc:
            logger.warning("proxy toggle: read persist failed: %s", exc)
        return None  # type: ignore

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


def get_proxy_toggle() -> ProxyToggleManager:
    global _manager
    if _manager is None:
        _manager = ProxyToggleManager()
    return _manager
