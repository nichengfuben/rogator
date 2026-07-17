from __future__ import annotations

"""Core state, scheduler, and resilient execution for the Qwen adapter."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from echotools.logger import get_logger
from echotools.protocol.base import ToolProtocol

from server.formats import (
    MAX_CHARS,
    MAX_CONCURRENT,
    MAX_QUEUE_SIZE,
    MAX_REQUEST_RESTARTS,
    RESTART_DELAY,
    SHUTDOWN_CANCEL_GRACE,
    SHUTDOWN_WAIT_IDLE_TIMEOUT,
    DEFAULT_MODEL,
    TokenExpiredError,
)
from echotools import get_protocol
from server.qwen_client import QwenClient, QwenSession

logger = get_logger("rogator")


# ============================================================
# 自定义异常
# ============================================================


class QueueFullError(Exception):
    pass


# ============================================================
# 长文本分割器
# ============================================================

class LongTextSplitter:
    def __init__(self, max_chars: int = MAX_CHARS):
        self.max_chars = max_chars

    def split(self, text: str):
        if len(text) <= self.max_chars:
            return text, None, None
        send_text = text[-self.max_chars:]
        remaining_text = text[:-self.max_chars]
        filename = f"remaining_{int(time.time())}_{__import__('uuid').uuid4().hex[:8]}.txt"
        return send_text, filename, remaining_text.encode("utf-8")


# ============================================================
# 调度器
# ============================================================

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

    async def submit(self, coro_factory) -> Any:
        async with self._lock:
            if self._shutting_down:
                raise QueueFullError("Shutting down")
            if self._pending >= MAX_QUEUE_SIZE:
                raise QueueFullError("Queue full")
            self._pending += 1
        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    return await coro_factory()
            return await coro_factory()
        finally:
            async with self._lock:
                self._pending = max(0, self._pending - 1)


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


# ============================================================
# 弹性执行
# ============================================================

async def _run_resilient(req_id: str, state: "AppState", func) -> Any:
    attempts = 0
    last_error: Optional[Exception] = None
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
            new_session = await state.client.switch_to_next()
            if new_session is None:
                raise RuntimeError("All sessions expired, no valid session available") from e
            logger.info("Switched to session %s, retrying %s", new_session.username[:6], req_id)
            attempts += 1
        except Exception as e:
            last_error = e
            attempts += 1
            error_str = str(e)
            logger.debug("Resilient retry %s #%d: %s", req_id, attempts, error_str[:200])
            if "401" in error_str or "unauthorized" in error_str.lower():
                await state.client.switch_to_next()
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


# ============================================================
# 应用状态
# ============================================================

class AppState:
    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()
        self._shutdown_requested = False
        self.splitter = LongTextSplitter()
        self.client = QwenClient(self.splitter)
        self.scheduler = RequestScheduler(MAX_CONCURRENT, MAX_QUEUE_SIZE)
        self.tracker = ActiveRequestTracker()
        self.protocol: ToolProtocol = get_protocol("entml")
        self._models: List[str] = self.client.load_models_cache()
        self.model: str = DEFAULT_MODEL
        self._bg_tasks: List[asyncio.Task] = []

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_requested or self.shutdown_event.is_set()

    async def refresh_models(self) -> None:
        try:
            self._models = await self.client.fetch_models(use_cache=False)
            logger.info("Refreshed models: %d", len(self._models))
        except Exception as e:
            logger.warning("Refresh models failed: %s", e)

    async def _models_refresh_loop(self, interval: float = 86400.0) -> None:
        while not self.is_shutting_down:
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                await self.refresh_models()

    def start_background_tasks(self) -> None:
        self._bg_tasks.append(asyncio.create_task(self._models_refresh_loop()))

    async def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.shutdown_event.set()
        for task in self._bg_tasks:
            if not task.done():
                task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self.scheduler.mark_shutting_down()
        cancelled = await self.tracker.cancel_all()
        if cancelled:
            await asyncio.sleep(SHUTDOWN_CANCEL_GRACE)
        await self.scheduler.wait_idle(timeout=SHUTDOWN_WAIT_IDLE_TIMEOUT)
