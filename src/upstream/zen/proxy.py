from __future__ import annotations

"""Zen 代理池：静态（config.toml）+ 动态（proxy_pool.json）。"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from upstream.zen.proxy_state import load_state_payload, save_state_payload

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
    top_n: int = 0,
) -> List[Optional[str]]:
    """合并静态与动态池。static_pool 已包含 env 注入的代理 + toml ``[proxy].static``。

    ``top_n`` 仅作用于动态池（按 latency 升序的入参由调用方保证），不影响静态候选。
    """
    merged: List[Optional[str]] = list(static_pool) if static_pool else [None]
    existing = {p for p in merged if p is not None}
    candidates = dynamic_pool[:top_n] if top_n > 0 else dynamic_pool
    for proxy in candidates:
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


def load_static_pool_from_env() -> List[Optional[str]]:
    """从 HTTP(S)_PROXY / ALL_PROXY 环境变量读取静态代理。

    多个变量同时存在时按 HTTPS_PROXY → HTTP_PROXY → ALL_PROXY 顺序取首个非空，
    归一化为 URL 字符串；与 toml 中 ``[proxy].static`` 合并时 env 项排在前面。
    """
    candidates = (
        os.environ.get("HTTPS_PROXY"),
        os.environ.get("https_proxy"),
        os.environ.get("HTTP_PROXY"),
        os.environ.get("http_proxy"),
        os.environ.get("ALL_PROXY"),
        os.environ.get("all_proxy"),
    )
    for raw in candidates:
        url = normalize_proxy_url(raw)
        if url:
            return [url]
    return []


def _coalesce_static_pool(
    env_pool: List[Optional[str]],
    static_toml: List[Optional[str]],
) -> List[Optional[str]]:
    """env 代理排在 toml static 之前；保留首次出现的项，去重。"""
    static: List[Optional[str]] = []
    seen: set = set()
    for url in env_pool + static_toml:
        if url in seen:
            continue
        seen.add(url)
        static.append(url)
    return static or [None]


def _resolve_pool_paths(
    section: Dict[str, Any],
) -> tuple[str, str, int]:
    """从 toml 解析动态池路径、状态文件路径、动态 top_n。"""
    from upstream.zen import DEFAULT_DYNAMIC_TOP_N
    from server.config.files import PROJECT_ROOT, USER_UPSTREAM_DIR

    # 动态代理池路径固定：避免 toml 误配或脚本 cwd 漂移。
    pool_file = str(USER_UPSTREAM_DIR / "zen" / "proxy_pool.json")
    state_file = str(
        section.get("state_file") or "persist/zen/proxy_state.json"
    ).strip()
    top_n_raw = section.get("dynamic_top_n")
    try:
        top_n = int(top_n_raw) if top_n_raw is not None else DEFAULT_DYNAMIC_TOP_N
    except (TypeError, ValueError):
        top_n = DEFAULT_DYNAMIC_TOP_N
    # pool_file / state_file 转绝对路径：便于跨 cwd 调用
    if not Path(pool_file).is_absolute():
        pool_file = str(PROJECT_ROOT / pool_file)
    if not Path(state_file).is_absolute():
        state_file = str(PROJECT_ROOT / state_file)
    return pool_file, state_file, top_n


def build_proxy_pool_from_toml(
    raw: Dict[str, Any],
) -> tuple[List[Optional[str]], str, str, int, List[Optional[str]]]:
    """从 upstream zen config 构建合并池。

    返回 ``(pool, pool_file, state_file, top_n, static_pool)``：
    - ``static_pool`` 是 env + toml ``[proxy].static`` 合并后的静态节点，供
      后台刷新时区分动态池节点（不入 proxy_pool.json）。
    - 合并顺序：env 代理（首个非空）→ toml ``[proxy].static`` → 动态池。
    - 动态池路径固定为 ``config/upstream/zen/proxy_pool.json``（相对项目根），
      不再通过 toml 配置——避免多份漂移或路径写错导致主服务读不到代理。
    """
    section = raw.get("proxy") if isinstance(raw.get("proxy"), dict) else {}
    env_pool = load_static_pool_from_env()
    static_toml = load_static_pool_from_config(section.get("static"))
    static = _coalesce_static_pool(env_pool, static_toml)
    pool_file, state_file, top_n = _resolve_pool_paths(section)
    dynamic = load_dynamic_proxy_pool(pool_file)
    merged = merge_proxy_pools(static, dynamic, top_n=top_n)
    logger.debug(
        "zen proxy pool: env=%d toml_static=%d dynamic=%d merged=%d top_n=%d file=%s",
        len(env_pool), len(static_toml), len(dynamic), len(merged), top_n, pool_file,
    )
    return merged, pool_file, state_file, top_n, static


def save_proxy_pool_file(
    path: str,
    pool: List[Optional[str]],
    static_pool: Optional[List[Optional[str]]] = None,
) -> None:
    """将动态代理池写回 proxy_pool.json，静态节点不入此文件。

    静态池由 ``[proxy].static`` / ``HTTP(S)_PROXY`` 环境变量管理；调用方
    通过 ``static_pool`` 告知"哪些代理属于静态"以便落盘前剔除，缺省则按
    ``pool`` 全部落盘（仅在纯动态场景下使用）。
    """
    static_known: set = {p for p in (static_pool or []) if p is not None}
    proxies = []
    seen: set = set()
    for url in pool:
        if url is None or url in static_known or url in seen:
            continue
        seen.add(url)
        proxies.append({"proxy": url, "status": "ok", "latency": 0.0, "model_count": 0})
    data = {
        "timestamp": int(time.time()),
        "total_tested": 0,
        "working": len(proxies),
        "elapsed_seconds": 0.0,
        "proxies": proxies,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as exc:
        logger.warning("zen save proxy pool failed: %s", exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


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
        self.__lock: Optional[asyncio.Lock] = None
        # 节点级 mute：{节点描述: 解除静音的 Unix 时间戳}
        self._muted: Dict[str, float] = {}
        self._load()

    @property
    def _lock(self) -> asyncio.Lock:
        """延迟创建锁，避免 Python 3.8 下无 event loop 时 RuntimeError。"""
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

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

    def all_nodes_muted(self) -> bool:
        """检查是否所有节点都处于静音期。"""
        for i in range(len(self._pool)):
            if not self._is_muted(self._describe(i)):
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
        # mute 后立即落盘，否则下次进程退出 mute 就丢失；
        # run_in_executor 不阻塞 event loop，写失败仅记日志
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._save_sync)
        except Exception as exc:
            logger.error("zen NodeManager persist after mute failed: %s", exc)

    def _load(self) -> None:
        idx, muted, _restored, _expired = load_state_payload(
            self._state_file, len(self._pool), self._describe,
        )
        self._current_index = idx
        self._muted = muted

    def _save_sync(self) -> None:
        save_state_payload(
            self._state_file,
            current_index=self._current_index,
            describe_at=self._describe,
            muted=self._muted,
        )

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
                # 所有节点都被 mute，保持当前位置；调用方应检测并返回 429
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
            new_descs = {self._describe(i) for i in range(len(new_pool))}
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
