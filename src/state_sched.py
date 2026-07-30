from __future__ import annotations

"""请求调度、活跃任务跟踪与弹性重试。"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from echotools.logger import get_logger

from server.config import CONFIG
from server.formats import (
    MAX_REQUEST_RESTARTS,
    RESTART_DELAY,
    TokenExpiredError,
)

if TYPE_CHECKING:
    from state import AppState, QueueFullError

logger = get_logger("rogator")


def _max_queue_size() -> int:
    import state as state_mod
    return int(state_mod.MAX_QUEUE_SIZE)


class RequestScheduler:
    def __init__(self, max_concurrent: int, max_queue: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent != -1 else None
        self._pending: int = 0
        self._lock = asyncio.Lock()
        self._shutting_down: bool = False

    @property
    def pending(self) -> int:
        return self._pending

    def mark_shutting_down(self) -> None:
        self._shutting_down = True

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
            if self._pending >= _max_queue_size():
                raise QueueFullError("Queue full")
            self._pending += 1

    async def _release_pending(self) -> None:
        async with self._lock:
            self._pending = max(0, self._pending - 1)

    async def acquire_slot(self) -> None:
        await self._reserve_pending()
        try:
            if self._semaphore is not None:
                await self._semaphore.acquire()
        except BaseException:
            await self._release_pending()
            raise

    async def release_slot(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()
        await self._release_pending()

    async def submit(self, coro_factory) -> Any:
        await self._reserve_pending()
        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    return await coro_factory()
            return await coro_factory()
        finally:
            await self._release_pending()


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
            targets = [t for t in self._tasks.values() if t is not current and not t.done()]
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


async def run_resilient(req_id: str, state: "AppState", func) -> Any:
    attempts = 0
    last_error: Optional[Exception] = None
    qwen = state._clients.get("qwen")
    while True:
        if state.is_shutting_down:
            raise asyncio.CancelledError("Shutting down")
        task = asyncio.current_task()
        await state.tracker.register(req_id, task)
        try:
            return await func()
        except asyncio.CancelledError:
            if state.is_shutting_down:
                raise
            attempts += 1
            logger.debug("Resilient: %s cancelled (restart #%d)", req_id, attempts)
        except TokenExpiredError as e:
            logger.warning("Token expired for %s: %s", req_id, e)
            if qwen is None:
                raise RuntimeError("No Qwen client for token retry") from e
            new_session = await qwen.switch_to_next()
            if new_session is None:
                raise RuntimeError("All sessions expired, no valid session available") from e
            logger.info("Switched to session %s, retrying %s", new_session.username[:6], req_id)
            attempts += 1
        except Exception as e:
            last_error = e
            attempts += 1
            error_str = str(e)
            logger.debug("Resilient retry %s #%d: %s", req_id, attempts, error_str[:200])
            if qwen and ("401" in error_str or "unauthorized" in error_str.lower()):
                await qwen.switch_to_next()
        finally:
            await state.tracker.unregister(req_id)
        if MAX_REQUEST_RESTARTS != -1 and attempts >= MAX_REQUEST_RESTARTS:
            raise RuntimeError(f"Max restarts ({MAX_REQUEST_RESTARTS}) exceeded for {req_id}") from last_error
        delay = RESTART_DELAY * (2 ** (attempts - 1))
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
        else:
            raise asyncio.CancelledError("Shutting down")


async def models_refresh_loop(state: "AppState") -> None:
    interval = max(0.0, CONFIG.models_refresh_interval)
    if interval <= 0:
        return
    while not state.is_shutting_down:
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
                await replenish()
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


async def session_cleanup_loop(state: "AppState", interval: float = 0) -> None:
    """兼容旧测试：仅维护默认 qwen client。"""
    await session_maintenance_loop(state, state.client, "qwen", interval=interval)


def start_background_tasks(state: "AppState") -> List[asyncio.Task]:
    tasks: List[asyncio.Task] = [
        asyncio.create_task(upstream_init_background(state)),
    ]
    for name, client in state._clients.items():
        token_loop = getattr(client, "token_maintenance_loop", None)
        if callable(token_loop):
            tasks.append(
                asyncio.create_task(token_loop(state.shutdown_event))
            )
            continue
        if not hasattr(client, "replenish_sessions"):
            continue
        tasks.append(asyncio.create_task(session_maintenance_loop(state, client, name)))
    tasks.append(asyncio.create_task(models_refresh_loop(state)))
    return tasks
