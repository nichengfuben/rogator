from __future__ import annotations

"""Runtime compatibility shims and backward-compatible re-exports.

Merged from: runtime.py, shared.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..transport.endpoints import (
    BASE_URL,
    CAPS,
    MODELS,
    MODELS_PERSIST_PATH,
    SMART_PROXY_ENABLED,
    USER_AGENT,
    USER_AGENT_MOBILE,
    SEC_CH_UA,
    FRONTEND_VERSION,
    BAXIA_SDK_VERSION,
    BXUA_VERSION,
    CUSTOM_BASE64_CHARS,
    PERSIST_PATH,
    TASK_TIMERS_PATH,
    PROXY_SELECTOR_PERSIST_PATH,
    GENERATED_IMAGE_DIR,
    GENERATED_VIDEO_DIR,
    TTS_DIR,
    UPLOAD_TEMP_DIR,
    LOGIN_BATCH,
    LOGIN_BATCH_SIZE,
    LOGIN_CONCURRENCY,
    LOGIN_POOL_SIZE,
    LOGIN_SELECT_MIN,
    LOGIN_SELECT_MAX,
    INITIAL_LOGIN_MAX,
    LOGIN_POLL_INTERVAL,
    TOKEN_EXPIRY_MARGIN,
    TOKEN_REFRESH_INTERVAL,
    COOKIE_REFRESH_INTERVAL,
    PERSIST_INTERVAL,
    SSE_TIMEOUT,
    TTS_TIMEOUT,
    VIDEO_CDN_BASE,
    VIDEO_TASK_MAX_POLL_TIME,
    VIDEO_TASK_POLL_INTERVAL,
    DEFAULT_FULL_SETTINGS,
)


# ---------------------------------------------------------------------------
# Runtime shims (from runtime.py)
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """Minimal candidate object compatible with the adapter runtime."""

    id: str
    platform: str
    resource_id: str
    models: List[str]
    context_length: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    chat: bool = False
    vision: bool = False
    thinking: bool = False
    search: bool = False
    image_gen: bool = False
    image_edit: bool = False
    audio_gen: bool = False
    video_gen: bool = False
    continuation: bool = False
    artifacts: bool = False


def make_id(platform: str, resource_id: str) -> str:
    """Build a stable candidate identifier."""
    return f"{platform}:{resource_id}"


class PlatformAdapter:
    """Minimal base adapter interface used when the host project is absent."""

    @property
    def name(self) -> str:
        raise NotImplementedError


class ModelsCache:
    """Small local model cache fallback."""

    def __init__(self, namespace: str, models: List[str], fetch_enabled: bool = False) -> None:
        self.namespace = namespace
        self.models = list(models)
        self.fetch_enabled = fetch_enabled

    async def load(self) -> None:
        """No-op fallback load."""
        return None

    async def _do_refresh(
        self,
        fetcher: Callable[[], Awaitable[List[str]]],
        on_update: Optional[Callable[[List[str]], Awaitable[None]]] = None,
    ) -> None:
        """Refresh models through the provided fetcher."""
        models = await fetcher()
        if models:
            self.models = list(models)
            if on_update is not None:
                await on_update(self.models)

    async def start_refresh_loop(
        self,
        fetcher: Callable[[], Awaitable[List[str]]],
        interval: int,
        on_update: Optional[Callable[[List[str]], Awaitable[None]]] = None,
    ) -> None:
        """Run a simple periodic refresh loop."""
        while True:
            await self._do_refresh(fetcher, on_update=on_update)
            await asyncio.sleep(interval)


class ProxySelector:
    """Small persistence-backed proxy selector fallback."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.prefer_proxy = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            self.prefer_proxy = bool(data.get('prefer_proxy', False))
        except Exception:
            self.prefer_proxy = False

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({'prefer_proxy': self.prefer_proxy}, indent=2), encoding='utf-8')

    def select(self) -> bool:
        """Return the current proxy preference."""
        return self.prefer_proxy

    def record(self, used_proxy: bool, success: bool, latency_ms: Optional[float] = None) -> None:
        """Update a simple preference heuristic."""
        if success:
            if latency_ms is not None and latency_ms < 2000:
                self.prefer_proxy = used_proxy
        else:
            self.prefer_proxy = False if used_proxy else self.prefer_proxy
        self._save()


class _ProxyConfig:
    def __init__(self) -> None:
        self.proxy_enabled = False


class _PlatformsProxyConfig:
    def is_platform_enabled(self, platform: str) -> bool:
        return True


class _Config:
    def __init__(self) -> None:
        self.proxy = _ProxyConfig()
        self.platforms_proxy = _PlatformsProxyConfig()


def get_config() -> _Config:
    """Return a minimal configuration object."""
    return _Config()


def get_proxy_server() -> str:
    """Return an empty proxy URL in standalone mode."""
    return ''


# ---------------------------------------------------------------------------
# Backward-compatible re-exports (from shared.py)
# ---------------------------------------------------------------------------

from ..crypto.crypto import (
    collect_fingerprint_data,
    custom_encode,
    generate_bxua,
    generate_device_id,
    generate_fingerprint,
    get_baxia_tokens,
    get_bxumidtoken,
    hash_password,
    lzw_compress,
)
from ..crypto.crypto import build_cookie_string, build_headers, build_login_headers, build_stop_headers
from ..storage.storage import (
    DATA_URI_EXT_MAP,
    EXTENSION_TO_MIME,
    build_file_object,
    build_url_file_object,
    build_wav_from_pcm,
    get_file_category,
    get_mime_type,
    save_image_file,
    save_video_file,
    save_wav_file,
)
from ..storage.storage import build_oss_authorization
from ..transport.sse import parse_sse_event, parse_sse_line
from ..media.video import build_cdn_video_url
from ..compat.payloads import (
    DEFAULT_FEATURE_CONFIG,
    build_i2v_payload,
    build_new_chat_payload,
    build_payload,
    build_replace_content_payload,
    build_stop_payload,
    build_tts_payload,
)
from ..transport.endpoints import HASH_FIELDS


__all__ = [
    "Any",
    "Dict",
    "BASE_URL",
    "CAPS",
    "MODELS",
    "MODELS_PERSIST_PATH",
    "SMART_PROXY_ENABLED",
    "USER_AGENT",
    "USER_AGENT_MOBILE",
    "SEC_CH_UA",
    "FRONTEND_VERSION",
    "BAXIA_SDK_VERSION",
    "BXUA_VERSION",
    "CUSTOM_BASE64_CHARS",
    "DEFAULT_FULL_SETTINGS",
    "DEFAULT_FEATURE_CONFIG",
    "PERSIST_PATH",
    "TASK_TIMERS_PATH",
    "PROXY_SELECTOR_PERSIST_PATH",
    "GENERATED_IMAGE_DIR",
    "GENERATED_VIDEO_DIR",
    "TTS_DIR",
    "UPLOAD_TEMP_DIR",
    "LOGIN_BATCH",
    "LOGIN_BATCH_SIZE",
    "LOGIN_CONCURRENCY",
    "LOGIN_POOL_SIZE",
    "LOGIN_SELECT_MIN",
    "LOGIN_SELECT_MAX",
    "INITIAL_LOGIN_MAX",
    "LOGIN_POLL_INTERVAL",
    "TOKEN_EXPIRY_MARGIN",
    "TOKEN_REFRESH_INTERVAL",
    "COOKIE_REFRESH_INTERVAL",
    "PERSIST_INTERVAL",
    "SSE_TIMEOUT",
    "TTS_TIMEOUT",
    "VIDEO_CDN_BASE",
    "VIDEO_TASK_MAX_POLL_TIME",
    "VIDEO_TASK_POLL_INTERVAL",
    "build_cookie_string",
    "build_headers",
    "build_login_headers",
    "build_stop_headers",
    "build_payload",
    "build_new_chat_payload",
    "build_stop_payload",
    "build_i2v_payload",
    "build_tts_payload",
    "build_replace_content_payload",
    "build_file_object",
    "build_url_file_object",
    "build_cdn_video_url",
    "build_oss_authorization",
    "build_wav_from_pcm",
    "get_file_category",
    "get_mime_type",
    "save_image_file",
    "save_video_file",
    "save_wav_file",
    "extract_model_ids",
    "parse_sse_event",
    "parse_sse_line",
    "EXTENSION_TO_MIME",
    "DATA_URI_EXT_MAP",
    "HASH_FIELDS",
    "generate_bxua",
    "get_bxumidtoken",
    "get_baxia_tokens",
    "generate_device_id",
    "generate_fingerprint",
    "collect_fingerprint_data",
    "custom_encode",
    "lzw_compress",
    "hash_password",
    "Candidate",
    "make_id",
    "PlatformAdapter",
    "ModelsCache",
    "ProxySelector",
    "ProxyState",
    "LogsMixin",
    "QwenAdapter",
]
