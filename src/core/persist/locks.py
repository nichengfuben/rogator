"""全局持久化互斥与迁移去重状态：进程内单点定义。

``_save_lock``（threading）与 ``_save_lock_async``（asyncio）必须成对出现，
否则同步路径与异步路径会交错写同一个 sessions.json。``_migrated_upstreams``
按 upstream 名去重一次迁移。早期版本在 ``store.py`` 与 ``login_history.py``
各自维护一份，引入历史漂移；现在统一收敛到本模块，调用方改为
``from core.persist.locks import save_lock, save_lock_async, get_async_lock``。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

_save_lock = threading.Lock()
_save_lock_async: Optional["asyncio.Lock"] = None
_migrated_upstreams: set[str] = set()


def get_async_lock() -> "asyncio.Lock":
    """返回事件循环内的 ``asyncio.Lock``；首次调用必须在 running loop 里。"""
    global _save_lock_async
    if _save_lock_async is None:
        _save_lock_async = asyncio.Lock()
    return _save_lock_async


def reset_async_lock() -> None:
    """测试用：丢弃已创建的 asyncio.Lock，让下一次 ``get_async_lock()`` 重新创建。"""
    global _save_lock_async
    _save_lock_async = None


def mark_migrated(upstream: str) -> bool:
    """记录已迁移的 upstream 名；首次返 True，重复返 False。"""
    key = upstream.strip().lower()
    if key in _migrated_upstreams:
        return False
    _migrated_upstreams.add(key)
    return True


__all__ = [
    "_save_lock",
    "_save_lock_async",
    "_migrated_upstreams",
    "save_lock",
    "save_lock_async",
    "get_async_lock",
    "mark_migrated",
]

# 兼容旧命名
save_lock = _save_lock
save_lock_async = _save_lock_async