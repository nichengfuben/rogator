from __future__ import annotations

"""Platform adapter, model helpers, log buffers, and proxy state.

Merged from: adaptercore.py, models.py, logs.py, proxy.py
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import aiohttp

logger = logging.getLogger(__name__)

try:
    from src.core.dispatch.candidate import Candidate
except ModuleNotFoundError:
    from .runtime import Candidate

try:
    from src.platforms.base import PlatformAdapter
except ModuleNotFoundError:
    from .runtime import PlatformAdapter

from ..client import QwenClient


# ---------------------------------------------------------------------------
# ProxyState (from proxy.py)
# ---------------------------------------------------------------------------


class ProxyState:
    """Track whether proxy use is forced on, forced off, or inherited."""

    def __init__(self) -> None:
        self.override: Optional[bool] = None

    def set_enabled(self, enabled: bool) -> None:
        """Force proxy on or off."""
        self.override = bool(enabled)

    def load(self, override: Optional[bool]) -> None:
        """Restore the persisted override state."""
        self.override = override

    def is_enabled(self) -> bool:
        """Return whether proxy is currently forced on."""
        return bool(self.override)

    def to_dict(self) -> dict:
        """Serialize the state for persistence."""
        return {"enabled": self.override}


# ---------------------------------------------------------------------------
# Model helpers (from models.py)
# ---------------------------------------------------------------------------

_KEYS: Tuple[str, ...] = ("id", "modelId", "model_id", "name")


def _id_from_dict(item: dict) -> Optional[str]:
    for key in _KEYS:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def _iter_ids(items: Iterable[Any]) -> Iterable[str]:
    for item in items:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            model_id = _id_from_dict(item)
            if model_id:
                yield model_id


def extract_model_ids(raw: Any) -> List[str]:
    """Extract de-duplicated model identifiers from heterogeneous payloads."""
    result: List[str] = []
    seen: Set[str] = set()

    def push(value: str) -> None:
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    if isinstance(raw, list):
        for item in _iter_ids(raw):
            push(item)
        return result
    if not isinstance(raw, dict):
        return result

    candidates = []
    data = raw.get("data")
    if isinstance(data, list):
        candidates.append(data)
    elif isinstance(data, dict):
        nested = data.get("models")
        if isinstance(nested, list):
            candidates.append(nested)
    simple = raw.get("models")
    if isinstance(simple, list):
        candidates.append(simple)

    for block in candidates:
        for item in _iter_ids(block):
            push(item)
    return result


# ---------------------------------------------------------------------------
# Log buffers (from logs.py)
# ---------------------------------------------------------------------------


class LogsMixin:
    """Collect short-lived log buffers and expose flush helpers."""

    async def _flush_relogin_buffer(self) -> None:
        await asyncio.sleep(1)
        self._flush_relogin_buffer_now()
        self._relogin_flush_task = None

    async def _flush_retry_log_buffer(self) -> None:
        await asyncio.sleep(1)
        self._flush_retry_log_buffer_now()
        self._retry_log_flush_task = None

    async def _flush_login_fail_buffer(self) -> None:
        await asyncio.sleep(1)
        self._flush_login_fail_buffer_now()
        self._login_fail_flush_task = None

    def _log_queued_relogin(self, username_prefix: str) -> None:
        self._relogin_log_buffer.append(username_prefix)
        task: Optional[asyncio.Task] = getattr(self, '_relogin_flush_task', None)
        if task is None or task.done():
            self._relogin_flush_task = asyncio.create_task(self._flush_relogin_buffer())

    def _log_retry(self, message: str) -> None:
        self._retry_log_buffer.append(message)
        task: Optional[asyncio.Task] = getattr(self, '_retry_log_flush_task', None)
        if task is None or task.done():
            self._retry_log_flush_task = asyncio.create_task(self._flush_retry_log_buffer())

    def _log_login_failure(self, username_prefix: str, error_message: str) -> None:
        self._login_fail_buffer.append((username_prefix, error_message))
        task: Optional[asyncio.Task] = getattr(self, '_login_fail_flush_task', None)
        if task is None or task.done():
            self._login_fail_flush_task = asyncio.create_task(self._flush_login_fail_buffer())

    def _flush_relogin_buffer_now(self) -> None:
        self._relogin_log_buffer.clear()

    def _flush_retry_log_buffer_now(self) -> None:
        self._retry_log_buffer.clear()

    def _flush_login_fail_buffer_now(self) -> None:
        self._login_fail_buffer.clear()


# ---------------------------------------------------------------------------
# QwenAdapter (from adaptercore.py)
# ---------------------------------------------------------------------------


class QwenAdapter(PlatformAdapter):
    """Expose the Qwen client through the platform adapter interface."""

    def __init__(self) -> None:
        super().__init__(platform="qwen")
        self._client = QwenClient()
        self._session: Optional[aiohttp.ClientSession] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def ensure_initialized(self) -> None:
        """Initialize the underlying HTTP client once."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
            connector = aiohttp.TCPConnector(ssl=False, limit=100)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            await self._client.init_immediate(self._session)
            asyncio.create_task(self._client.background_setup())
            self._initialized = True

    async def shutdown(self) -> None:
        """Shut down the client and the shared HTTP session."""
        if not self._initialized:
            return
        await self._client.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._initialized = False

    async def candidates(self) -> List[Candidate]:
        """Return available account-backed candidates."""
        await self.ensure_initialized()
        return await self._client.candidates()

    async def ensure_candidates(self, count: int) -> int:
        """Return the current candidate count."""
        await self.ensure_initialized()
        return await self._client.ensure_candidates(count)

    async def complete(
        self,
        candidate: Candidate,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Proxy chat completion calls to the underlying Qwen client."""
        await self.ensure_initialized()
        async for chunk in self._client.complete(candidate, messages, model, stream, **kwargs):
            yield chunk

    async def stop(self, candidate: Candidate) -> bool:
        """Stop the active generation for the given candidate."""
        await self.ensure_initialized()
        return await self._client.stop_candidate_generation(candidate)

    async def get_models(self) -> List[str]:
        """Return the current model list."""
        await self.ensure_initialized()
        return self._client.get_models()

    async def set_proxy_enabled(self, enabled: bool) -> None:
        """Force-enable or disable proxy usage."""
        await self.ensure_initialized()
        self._client.set_proxy_enabled(enabled)

    async def is_proxy_enabled(self) -> bool:
        """Return whether proxy use is currently forced on."""
        await self.ensure_initialized()
        return self._client.is_proxy_enabled()

    async def refresh_models(self) -> None:
        """Trigger a remote model refresh."""
        await self.ensure_initialized()
        await self._client.refresh_models()

    async def generate_video(
        self,
        prompt: str,
        image_url: str,
        token: str,
        user_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Expose the client image-to-video helper."""
        await self.ensure_initialized()
        return await self._client.generate_video(prompt, image_url, token, user_id, **kwargs)

    async def synthesize_tts(
        self,
        text: str,
        token: str,
        **kwargs: Any,
    ) -> Optional[str]:
        """Expose the client TTS helper."""
        await self.ensure_initialized()
        return await self._client.synthesize_tts(text, token, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Return a lightweight adapter config view."""
        return {
            "platform": "qwen",
            "models": self._client.get_models(),
        }
