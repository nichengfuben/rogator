"""请求调度、活跃任务跟踪与弹性重试。"""

from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import anyio

from echotools.base.logger import get_logger

from server.config import CONFIG

if TYPE_CHECKING:
    from state import AppState, QueueFullError

logger = get_logger("rogator")


def _max_queue_size(fixed: Optional[int] = None) -> int:
    if fixed is not None:
        return int(fixed)
    import state as state_mod

    return int(state_mod.MAX_QUEUE_SIZE)


class RequestScheduler:
    """并发/队列上限可热更：未传固定值时读 CONFIG / state.MAX_QUEUE_SIZE。"""

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        max_queue: Optional[int] = None,
    ) -> None:
        self._fixed_concurrent = max_concurrent
        self._fixed_queue = max_queue
        self._active: int = 0
        self._pending: int = 0
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._shutting_down: bool = False

    def _concurrent_limit(self) -> int:
        if self._fixed_concurrent is not None:
            return int(self._fixed_concurrent)
        return int(CONFIG.max_concurrent)

    def _queue_limit(self) -> int:
        return _max_queue_size(self._fixed_queue)

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> int:
        return self._pending

    def mark_shutting_down(self) -> None:
        self._shutting_down = True

    async def wake_for_config(self) -> None:
        """配置热更后唤醒等待并发槽的协程以重新读上限。"""
        async with self._cond:
            self._cond.notify_all()

    async def wait_idle(self, timeout: float = 30.0) -> bool:
        start = time.time()
        while self._pending > 0 and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)
        return self._pending == 0

    async def _reserve_pending(self) -> None:
        from state import QueueFullError

        async with self._lock:
            if self._shutting_down:
                raise QueueFullError("Shutting down")
            if self._pending >= self._queue_limit():
                raise QueueFullError("Queue full")
            self._pending += 1

    async def _release_pending(self) -> None:
        async with self._lock:
            self._pending = max(0, self._pending - 1)

    async def _acquire_concurrent(self) -> None:
        async with self._cond:
            while True:
                limit = self._concurrent_limit()
                if limit == -1 or self._active < limit:
                    self._active += 1
                    return
                await self._cond.wait()

    async def _release_concurrent(self) -> None:
        async with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    async def acquire_slot(self) -> None:
        await self._reserve_pending()
        try:
            await self._acquire_concurrent()
        except BaseException:
            await self._release_pending()
            raise

    async def release_slot(self) -> None:
        await self._release_concurrent()
        await self._release_pending()

    async def submit(self, coro_factory) -> Any:
        await self.acquire_slot()
        try:
            return await coro_factory()
        finally:
            await self.release_slot()


class ActiveRequestTracker:
    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register(self, req_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            self._tasks[req_id] = task

    async def unregister(self, req_id: str) -> None:
        async with self._lock:
            self._tasks.pop(req_id, None)

    async def cancel_all(self) -> int:
        current = asyncio.current_task()
        async with self._lock:
            targets = [
                t for t in self._tasks.values() if t is not current and not t.done()
            ]
            for t in targets:
                t.cancel()
            self._tasks = {r: t for r, t in self._tasks.items() if t is current}
            return len(targets)

    @property
    def count(self) -> int:
        return len(self._tasks)


@asynccontextmanager
async def tracked_request(state: "AppState", req_id: str) -> AsyncIterator[None]:
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("tracked_request requires a running task")
    await state.tracker.register(req_id, task)
    slot_acquired = False
    try:
        await state.scheduler.acquire_slot()
        slot_acquired = True
        yield
    finally:
        if slot_acquired:
            await state.scheduler.release_slot()
        await state.tracker.unregister(req_id)


async def models_refresh_loop(state: "AppState") -> None:
    while not state.is_shutting_down:
        interval = max(0.0, CONFIG.models_refresh_interval)
        if interval <= 0:
            return
        await state.refresh_models(require_session=True)
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def upstream_init_background(state: "AppState") -> None:
    """轻量上游初始化（不登录）；登录由 session maintenance 负责。"""
    try:
        await state.startup_upstreams()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Upstream init failed: %s", exc)


async def session_maintenance_loop(
    state: "AppState",
    client: Any,
    upstream: str,
    interval: float = 0,
) -> None:
    from core.session.store import CLEANUP_INTERVAL

    wait = interval if interval > 0 else CLEANUP_INTERVAL
    while not state.is_shutting_down:
        try:
            cleanup = getattr(client, "cleanup_expired_sessions", None)
            if callable(cleanup):
                cleanup()
            replenish = getattr(client, "replenish_sessions", None)
            if callable(replenish):
                scheduler = getattr(state, "scheduler", None)
                await replenish(scheduler=scheduler)
            wait_replenish = getattr(client, "wait_for_replenish_or_timeout", None)
            if inspect.iscoroutinefunction(wait_replenish):
                await wait_replenish(wait)
            else:
                await asyncio.wait_for(state.shutdown_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


async def session_cleanup_loop(state: "AppState", interval: float = 0) -> None:
    """兼容旧测试：仅维护默认 qwen client。"""
    await session_maintenance_loop(state, state.client, "qwen", interval=interval)


def start_background_tasks(state: "AppState") -> List[asyncio.Task]:
    return [asyncio.create_task(_background_tasks_main(state), name="rogator-bg")]


async def _background_tasks_main(state: "AppState") -> None:
    async with anyio.create_task_group() as tg:
        tg.start_soon(upstream_init_background, state)
        for name, client in state._clients.items():
            if not hasattr(client, "replenish_sessions"):
                continue
            pool_ok = getattr(client, "_login_pool_available", None)
            if callable(pool_ok):
                if not pool_ok():
                    continue
            tg.start_soon(session_maintenance_loop, state, client, name)
        for _name, client in state._clients.items():
            token_maint = getattr(client, "token_maintenance_loop", None)
            if callable(token_maint):
                tg.start_soon(token_maint, state.shutdown_event)
        tg.start_soon(models_refresh_loop, state)
