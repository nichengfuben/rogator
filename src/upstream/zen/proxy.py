from __future__ import annotations

"""Zen 代理池：静态（config.toml）+ 动态（proxy_pool.json）。"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rogator")


class ZenProxyError(RuntimeError):
    """代理连通失败，应切换节点重试。"""


def normalize_proxy_url(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("none", "direct", "null"):
        return None
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return "http://{}".format(text)


def load_dynamic_proxy_pool(path: str) -> List[str]:
    """读取动态代理池，按 latency 升序；文件缺失/损坏时返回空列表。"""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("zen dynamic proxy load failed (%s): %s", path, exc)
        return []
    if isinstance(data, dict):
        entries = data.get("proxies", [])
    elif isinstance(data, list):
        entries = data
    else:
        return []
    if not isinstance(entries, list):
        return []
    parsed: List[Dict[str, Any]] = []
    for entry in entries:
        proxy_raw: Any = ""
        latency: Any = None
        if isinstance(entry, dict):
            proxy_raw = entry.get("proxy") or entry.get("url") or ""
            latency = entry.get("latency")
        elif isinstance(entry, str):
            proxy_raw = entry
        else:
            continue
        proxy_url = normalize_proxy_url(proxy_raw)
        if not proxy_url:
            continue
        try:
            latency_val = float(latency) if latency is not None else float("inf")
        except (TypeError, ValueError):
            latency_val = float("inf")
        parsed.append({"proxy": proxy_url, "latency": latency_val})
    parsed.sort(key=lambda item: item["latency"])
    ordered: List[str] = []
    seen = set()
    for item in parsed:
        proxy = item["proxy"]
        if proxy in seen:
            continue
        seen.add(proxy)
        ordered.append(proxy)
    return ordered


def merge_proxy_pools(
    static_pool: List[Optional[str]],
    dynamic_pool: List[str],
) -> List[Optional[str]]:
    merged: List[Optional[str]] = list(static_pool) if static_pool else [None]
    existing = {p for p in merged if p is not None}
    for proxy in dynamic_pool:
        if proxy in existing:
            continue
        merged.append(proxy)
        existing.add(proxy)
    return merged or [None]


def load_static_pool_from_config(raw: Any) -> List[Optional[str]]:
    if not isinstance(raw, list) or not raw:
        return [None]
    out: List[Optional[str]] = []
    for item in raw:
        out.append(normalize_proxy_url(item))
    return out or [None]


def build_proxy_pool_from_toml(raw: Dict[str, Any]) -> tuple[List[Optional[str]], str, str]:
    """从 upstream zen config 构建合并池。返回 (pool, pool_file, state_file)。"""
    section = raw.get("proxy") if isinstance(raw.get("proxy"), dict) else {}
    static = load_static_pool_from_config(section.get("static"))
    pool_file = str(section.get("pool_file") or "proxy_pool.json").strip()
    state_file = str(
        section.get("state_file") or "persist/zen/proxy_state.json"
    ).strip()
    dynamic = load_dynamic_proxy_pool(pool_file)
    merged = merge_proxy_pools(static, dynamic)
    logger.debug(
        "zen proxy pool: static=%d dynamic=%d merged=%d file=%s",
        len(static), len(dynamic), len(merged), pool_file,
    )
    return merged, pool_file, state_file


def is_proxy_error(exc: BaseException) -> bool:
    if isinstance(exc, ZenProxyError):
        return True
    text = str(exc).lower()
    markers = (
        "cannot connect to host", "connection refused", "connection reset",
        "connection aborted", "proxy", "timed out", "timeout", "ssl",
        "certificate",
    )
    return any(kw in text for kw in markers)


class NodeManager:
    """按索引轮换代理节点，状态落盘；支持节点级 mute。"""

    # 429 节点静音时长（秒）
    MUTE_DURATION: float = 3600.0

    def __init__(self, pool: List[Optional[str]], state_file: str) -> None:
        self._pool: List[Optional[str]] = pool if pool else [None]
        self._state_file = state_file
        self._current_index = 0
        self._lock = asyncio.Lock()
        # 节点级 mute：{节点描述: 解除静音的 Unix 时间戳}
        self._muted: Dict[str, float] = {}
        self._load()

    def _describe(self, index: int) -> str:
        node = self._pool[index]
        return "direct" if node is None else node

    @property
    def current_proxy(self) -> Optional[str]:
        return self._pool[self._current_index]

    @property
    def current_description(self) -> str:
        return self._describe(self._current_index)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def _is_muted(self, desc: str) -> bool:
        """检查节点是否处于静音期（调用方须持有 _lock）。"""
        until = self._muted.get(desc)
        if until is None:
            return False
        if time.time() >= until:
            del self._muted[desc]
            return False
        return True

    async def mute_current(self, duration: float = MUTE_DURATION) -> None:
        """将当前节点静音指定秒数；后续 switch_next 会跳过该节点。"""
        async with self._lock:
            desc = self._describe(self._current_index)
            self._muted[desc] = time.time() + duration
            logger.debug(
                "zen node muted: %s for %.0fs (until %s)",
                desc, duration,
                time.strftime("%H:%M:%S", time.localtime(self._muted[desc])),
            )

    def _load(self) -> None:
        path = Path(self._state_file)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            idx = int(data.get("current_node_index", 0))
            if 0 <= idx < len(self._pool):
                self._current_index = idx
                logger.debug(
                    "zen NodeManager restored index=%d (%s)",
                    idx, self._describe(idx),
                )
                return
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("zen NodeManager load failed: %s", exc)
        self._current_index = 0

    def _save_sync(self) -> None:
        data = {
            "current_node_index": self._current_index,
            "current_node": self._describe(self._current_index),
            "updated_at": int(time.time()),
        }
        path = Path(self._state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(path.parent),
                delete=False,
                suffix=".tmp",
                encoding="utf-8",
            ) as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                tmp_path = fh.name
            os.replace(tmp_path, str(path))
        except Exception as exc:
            logger.error("zen NodeManager save failed: %s", exc)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def switch_next(self) -> str:
        async with self._lock:
            pool_len = len(self._pool)
            # 尝试找到下一个非 mute 节点，最多遍历整个池
            for _ in range(pool_len):
                self._current_index = (self._current_index + 1) % pool_len
                desc = self._describe(self._current_index)
                if not self._is_muted(desc):
                    break
            else:
                # 所有节点都被 mute，保持当前位置
                desc = self._describe(self._current_index)
                logger.debug(
                    "zen all %d nodes muted, staying at %s",
                    pool_len, desc,
                )
                return desc
            logger.debug("zen NodeManager switched -> %d (%s)", self._current_index, desc)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._save_sync)
        except Exception as exc:
            logger.error("zen NodeManager persist failed: %s", exc)
        return desc

    async def reload_pool(self, new_pool: List[Optional[str]]) -> None:
        """原子替换代理池并重置索引为 0（新池顺序已变，旧索引无意义）。"""
        if not new_pool:
            new_pool = [None]
        async with self._lock:
            old_size = len(self._pool)
            self._pool = new_pool
            self._current_index = 0
            # 清理不再存在于新池中的 mute 记录
            new_descs = set()
            for i in range(len(new_pool)):
                new_descs.add(self._describe(i))
            stale = [k for k in self._muted if k not in new_descs]
            for k in stale:
                del self._muted[k]
            logger.debug(
                "zen NodeManager pool reloaded: %d -> %d nodes, index=%d (%s), stale_mutes=%d",
                old_size, len(new_pool),
                self._current_index, self._describe(self._current_index),
                len(stale),
            )
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._save_sync)
        except Exception as exc:
            logger.error("zen NodeManager persist after reload failed: %s", exc)
