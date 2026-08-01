from __future__ import annotations

"""Core state and application lifecycle."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from echotools.fncall import ToolProtocol, get_protocol
from echotools.logger import get_logger

from server.config import CONFIG
from server.formats import (
    SHUTDOWN_CANCEL_GRACE,
    DEFAULT_MODEL,
)
from core.session.store import valid_session_count
from core.dispatch import select_upstream
from core.registry import load_upstreams
from state_sched import (
    ActiveRequestTracker,
    RequestScheduler,
    models_refresh_loop,
    run_resilient,
    start_background_tasks,
    tracked_request,
)
from core.transport.http import close_shared_connector

logger = get_logger("rogator")

_run_resilient = run_resilient
MAX_QUEUE_SIZE = CONFIG.max_queue_size


class QueueFullError(Exception):
    pass


QWEN_SEND_MAX_CHARS = CONFIG.qwen_send_max_chars
MODEL_CONTEXT_LENGTH = CONFIG.model_context_length


class LongTextSplitter:
    def __init__(
        self,
        max_chars: int = QWEN_SEND_MAX_CHARS,
        *,
        send_full_prompt: bool = False,
    ) -> None:
        self.max_chars = max_chars
        self.send_full_prompt = send_full_prompt

    def split(self, text: str):
        """Inject 后整段 prompt 超限：尾部 max_chars → send，剩余前缀 → 附件。"""
        if self.send_full_prompt or len(text) <= self.max_chars:
            return text, None, None
        send_text = text[-self.max_chars:]
        remaining_text = text[:-self.max_chars]
        filename = f"remaining_{int(time.time())}_{__import__('uuid').uuid4().hex[:8]}.txt"
        return send_text, filename, remaining_text.encode("utf-8")


class AppState:
    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()
        self._shutdown_requested = False
        self.splitter = LongTextSplitter(
            max_chars=CONFIG.qwen_send_max_chars,
            send_full_prompt=CONFIG.send_full_prompt,
        )
        self._registry = load_upstreams()
        self._clients: Dict[str, Any] = {}
        self._models_inventory: Dict[str, set[str]] = {}
        self._models: List[str] = []
        for name, mod in self._registry.modules.items():
            client = mod.create_client(self.splitter)
            self._clients[name] = client
            self._models_inventory[name] = set(client.load_models_cache())
        self._rebuild_unified_models()
        self.client = self._clients.get("qwen") or self.client_for(DEFAULT_MODEL, ("chat",))
        self.scheduler = RequestScheduler()
        self.tracker = ActiveRequestTracker()
        self.protocol: ToolProtocol = get_protocol("entml")
        self.model: str = DEFAULT_MODEL
        self._bg_tasks: List[asyncio.Task] = []

    def _rebuild_unified_models(self) -> None:
        ordered: List[str] = []
        seen: set[str] = set()
        for name in self._registry.names():
            for model_id in self._models_inventory.get(name, ()):
                if model_id not in seen:
                    ordered.append(model_id)
                    seen.add(model_id)
        self._models = ordered

    def owner_of_model(self, internal_id: str) -> str:
        for name, owned in self._models_inventory.items():
            if internal_id in owned:
                return name
        return "unknown"

    def merged_model_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        for client in self._clients.values():
            client_meta = getattr(client, "_model_meta", None)
            if isinstance(client_meta, dict):
                meta.update(client_meta)
        return meta

    def models_fetch_timestamp(self) -> float:
        times = [
            float(getattr(c, "_models_fetch_time", 0.0) or 0.0)
            for c in self._clients.values()
        ]
        return max(times) if times else 0.0

    def _models_by_upstream(self) -> Dict[str, set[str]]:
        return dict(self._models_inventory)

    def client_for(
        self,
        model_id: str,
        required_capabilities: tuple[str, ...] = ("chat",),
        *,
        upstream_name: Optional[str] = None,
    ) -> Any:
        if upstream_name:
            mod = self._registry.get(upstream_name)
        else:
            mod = select_upstream(
                model_id=model_id,
                required_capabilities=required_capabilities,
                models_by_upstream=self._models_by_upstream(),
                registry=self._registry,
            )
        cached = self._clients.get(mod.name)
        if cached is not None:
            return cached
        client = mod.create_client(self.splitter)
        self._clients[mod.name] = client
        self._models_inventory.setdefault(mod.name, set(client.load_models_cache()))
        self._rebuild_unified_models()
        return client

    async def startup_upstreams(self) -> None:
        """轻量上游初始化（不登录）；登录由后台 ``upstream_bootstrap`` 负责。"""
        for name, client in self._clients.items():
            startup = getattr(client, "startup", None)
            if callable(startup):
                await startup()
                self._models_inventory[name] = set(client.load_models_cache())
        self._rebuild_unified_models()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_requested or self.shutdown_event.is_set()

    async def refresh_models(self, *, require_session: bool = False, force: bool = False) -> None:
        try:
            updated = False
            qwen = self._clients.get("qwen")
            if qwen is not None:
                if not force and not qwen.models_refresh_due(CONFIG.models_refresh_interval):
                    logger.debug(
                        "Refresh models skipped: cache fresh (%.0fs ago, interval=%.0fs)",
                        time.time() - qwen._models_fetch_time,
                        CONFIG.models_refresh_interval,
                    )
                elif require_session and valid_session_count(qwen._sessions) == 0:
                    logger.debug("Refresh models skipped: no valid session")
                else:
                    models = await qwen.fetch_models(use_cache=not force)
                    if models:
                        self._models_inventory["qwen"] = set(models)
                        updated = True
                        logger.info("Refreshed qwen models: %d", len(models))
            for name, client in self._clients.items():
                if name == "qwen":
                    continue
                fetch = getattr(client, "fetch_models", None)
                if not callable(fetch):
                    continue
                if not force and hasattr(client, "models_refresh_due"):
                    if not client.models_refresh_due(CONFIG.models_refresh_interval):
                        continue
                models = await fetch(use_cache=not force)
                if models:
                    self._models_inventory[name] = set(models)
                    updated = True
                    logger.info("Refreshed %s models: %d", name, len(models))
            if updated:
                self._rebuild_unified_models()
        except Exception as e:
            logger.warning("Refresh models failed: %s", e)

    def start_background_tasks(self) -> None:
        qwen = self._clients.get("qwen")
        if qwen is not None:
            self.client = qwen
        self._bg_tasks.extend(start_background_tasks(self))

    async def _stop_background_tasks(self) -> None:
        for task in self._bg_tasks:
            if not task.done():
                task.cancel()
        if not self._bg_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._bg_tasks, return_exceptions=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown: background tasks did not exit within 2.0s")

    async def _drain_active_requests(self) -> None:
        self.scheduler.mark_shutting_down()
        cancelled = await self.tracker.cancel_all()
        if cancelled:
            logger.info("Shutdown: cancelled %d active stream/request task(s)", cancelled)
            await asyncio.sleep(SHUTDOWN_CANCEL_GRACE)
        idle = await self.scheduler.wait_idle(timeout=CONFIG.shutdown_wait_active_requests)
        if idle:
            return
        remaining = self.tracker.count
        if remaining:
            logger.warning(
                "Shutdown: %d request(s) still active after %.1fs, forcing cancel",
                remaining,
                CONFIG.shutdown_wait_active_requests,
            )
        await self.tracker.cancel_all()
        await asyncio.sleep(SHUTDOWN_CANCEL_GRACE)

    async def _shutdown_upstreams(self) -> None:
        qwen = self._clients.get("qwen")
        if qwen is not None and hasattr(qwen, "_persist_sessions"):
            qwen._persist_sessions()
        for client in self._clients.values():
            persist = getattr(client, "_persist_sessions", None)
            if callable(persist) and client is not qwen:
                persist()
            shutdown = getattr(client, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                await shutdown()
            except Exception as exc:
                logger.debug("Upstream shutdown failed: %s", exc)
        await close_shared_connector()

    async def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        logger.info(
            "Shutdown: stopping background tasks and active requests (target=%d)...",
            self.tracker.count,
        )
        self._shutdown_requested = True
        self.shutdown_event.set()
        await self._stop_background_tasks()
        await self._drain_active_requests()
        await self._shutdown_upstreams()
        self._shutdown_complete = True
