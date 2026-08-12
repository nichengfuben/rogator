from __future__ import annotations

"""Zen 上游客户端：无账号、OpenAI 兼容 API + 代理池。"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.session.models_cache import ModelsCacheMixin
from core.transport.conn_retry import run_with_connection_retry
from core.transport.http import upstream_timeout
from core.transport.owned import HttpTransportMixin
from server.formats import UpstreamUnavailableError
from upstream.zen.chat_stream import extract_error_info, post_chat_stream
from upstream.zen.payload import build_headers
from upstream.zen.proxy import (
    NodeManager,
    ZenProxyError,
    build_proxy_pool_from_toml,
    is_proxy_error,
)
from upstream.zen.proxy import (
    load_dynamic_proxy_pool,
    load_static_pool_from_config,
    merge_proxy_pools,
)
from upstream.zen.routes import (
    AUTO_REFRESH_MODELS,
    BASE_URL,
    DEFAULT_MODELS,
    FALLBACK_MODEL,
    FALLBACK_MODEL_ENABLED,
    MODELS_CACHE_TTL,
    MODELS_FETCH_TIMEOUT,
    MODELS_PATH,
    PROXY_REFRESH_INTERVAL,
    RETRY_COUNT,
)

logger = logging.getLogger("rogator")


class ZenModelNotSupportedError(RuntimeError):
    """上游不支持该模型（含 401 需鉴权模型）。"""


class ZenValidationError(RuntimeError):
    """上游校验请求参数失败，不可重试。"""


def _load_zen_toml() -> Dict[str, Any]:
    try:
        from server.config.app_config import _load_upstream_toml
        return _load_upstream_toml("zen") or {}
    except Exception:
        return {}


class ZenClient(HttpTransportMixin, ModelsCacheMixin):
    UPSTREAM_NAME = "zen"

    def __init__(self, splitter: Any = None) -> None:
        self._splitter = splitter
        self._init_http_transport()
        self._init_models_cache(list(DEFAULT_MODELS))
        raw = _load_zen_toml()
        pool, pool_file, state_file = build_proxy_pool_from_toml(raw)
        self.node_manager = NodeManager(pool, state_file)
        # 后台刷新所需状态
        section = raw.get("proxy") if isinstance(raw.get("proxy"), dict) else {}
        self._pool_file: str = pool_file
        self._static_pool = load_static_pool_from_config(section.get("static"))
        interval_raw = section.get("refresh_interval_seconds")
        try:
            self._refresh_interval: float = float(interval_raw) if interval_raw is not None else PROXY_REFRESH_INTERVAL
        except (TypeError, ValueError):
            self._refresh_interval = PROXY_REFRESH_INTERVAL
        self._refresh_task: Optional[asyncio.Task] = None

    def load_models_cache(self) -> List[str]:
        return list(self._models)

    async def startup(self) -> None:
        if self._refresh_interval > 0:
            self._refresh_task = asyncio.create_task(
                self._proxy_refresh_loop(), name="zen_proxy_refresh",
            )

    async def shutdown(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.close_http_transport()

    async def _proxy_refresh_loop(self) -> None:
        """后台定时重载动态代理池；异常仅记录日志，不影响服务。"""
        logger.debug(
            "zen proxy refresh loop started: interval=%.0fs file=%s",
            self._refresh_interval, self._pool_file,
        )
        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
            except asyncio.CancelledError:
                return
            try:
                dynamic = load_dynamic_proxy_pool(self._pool_file)
                merged = merge_proxy_pools(self._static_pool, dynamic)
                await self.node_manager.reload_pool(merged)
                logger.debug(
                    "zen proxy pool refreshed: dynamic=%d merged=%d",
                    len(dynamic), len(merged),
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("zen proxy refresh failed: %s", exc)

    async def fetch_models(self, *, use_cache: bool = True) -> List[str]:
        now = time.time()
        if (
            use_cache
            and self._models
            and (now - self._models_fetch_time) < MODELS_CACHE_TTL
        ):
            return list(self._models)

        async def _run() -> List[str]:
            http = await self._ensure_http_session()
            url = f"{BASE_URL}{MODELS_PATH}"
            timeout = upstream_timeout(MODELS_FETCH_TIMEOUT)
            kw: Dict[str, Any] = {
                "headers": build_headers(stream=False),
                "timeout": timeout,
            }
            proxy = self.node_manager.current_proxy
            if proxy:
                kw["proxy"] = proxy
            async with http.get(url, **kw) as resp:
                if resp.status != 200:
                    logger.warning("zen fetch_models HTTP %d", resp.status)
                    return list(DEFAULT_MODELS)
                data = await resp.json(content_type=None)
            return self._parse_models_payload(data)

        try:
            models = await run_with_connection_retry(
                "zen_fetch_models", _run, upstream="zen", transport_owner=self,
            )
        except Exception as exc:
            logger.warning("zen fetch_models failed: %s", exc)
            return list(DEFAULT_MODELS)
        self._models = list(models)
        self._models_fetch_time = time.time()
        # 同步新模型到注册表和 kimi-code config.toml
        try:
            from upstream.zen.models.registry_sync import sync_zen_registry
            sync_zen_registry(self._models)
        except Exception as exc:
            logger.debug("zen registry sync skipped: %s", exc)
        return list(self._models)

    def _parse_models_payload(self, data: Any) -> List[str]:
        if not isinstance(data, dict):
            return list(DEFAULT_MODELS)
        err = extract_error_info(data)
        if err:
            logger.warning("zen fetch_models error: %s", err["message"])
            return list(DEFAULT_MODELS)
        rows = data.get("data") or []
        if not isinstance(rows, list):
            return list(DEFAULT_MODELS)
        models = [
            str(m.get("id", ""))
            for m in rows
            if isinstance(m, dict) and m.get("id")
        ]
        free = [m for m in models if m.endswith("-free")]
        return free or models or list(DEFAULT_MODELS)

    async def _maybe_fallback_model(self, model: str) -> Optional[str]:
        if not FALLBACK_MODEL_ENABLED or model == FALLBACK_MODEL:
            return None
        if AUTO_REFRESH_MODELS:
            available = await self.fetch_models(use_cache=False)
        else:
            available = list(DEFAULT_MODELS)
        base = model.replace("-local", "")
        if base in available:
            return None
        logger.debug("zen model %s not in list, fallback -> %s", model, FALLBACK_MODEL)
        return FALLBACK_MODEL

    async def stream_chat(
        self,
        payload: Dict[str, Any],
        *,
        _fallback_applied: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = str(payload.get("model") or "")
        if not _fallback_applied:
            fb = await self._maybe_fallback_model(model)
            if fb is not None:
                alt = dict(payload)
                alt["model"] = fb
                async for event in self.stream_chat(alt, _fallback_applied=True):
                    yield event
                return
        async for event in self._stream_with_retries(payload, _fallback_applied):
            yield event

    async def _stream_with_retries(
        self,
        payload: Dict[str, Any],
        fallback_applied: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        last_error: Optional[Exception] = None
        for attempt in range(1 + RETRY_COUNT):
            proxy = self.node_manager.current_proxy
            desc = self.node_manager.current_description
            try:
                if attempt > 0:
                    logger.debug("zen retry %d/%d via %s", attempt, RETRY_COUNT, desc)
                async for event in post_chat_stream(self, payload, proxy=proxy):
                    yield event
                return
            except ZenModelNotSupportedError as exc:
                async for event in self._on_model_unsupported(
                    payload, fallback_applied, exc,
                ):
                    yield event
                return
            except ZenValidationError:
                raise
            except UpstreamUnavailableError as exc:
                # 上游不可用（429/502/503等）：静音当前节点 1h，切换到下一个
                await self.node_manager.mute_current()
                new_node = await self.node_manager.switch_next()
                if "429" in str(exc):
                    logger.debug("zen 429 rate limited via %s, switch -> %s", desc, new_node)
                else:
                    logger.debug("zen upstream unavailable via %s, switch -> %s", desc, new_node)
                await self.reset_http_transport()
                last_error = exc
                continue
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as exc:
                last_error = exc
                if isinstance(exc, ZenProxyError) or is_proxy_error(exc):
                    await self.node_manager.mute_current()
                    new_node = await self.node_manager.switch_next()
                    logger.debug("zen proxy error via %s, switch -> %s", desc, new_node)
                    await self.reset_http_transport()
                    continue
                logger.debug(
                    "zen attempt %d/%d via %s failed: %s",
                    attempt + 1, 1 + RETRY_COUNT, desc, exc,
                )
                await self.reset_http_transport()
        new_node = await self.node_manager.switch_next()
        if last_error is not None and "429" in str(last_error):
            raise UpstreamUnavailableError(
                "HTTP 429 - Rate limit exceeded",
                upstream="zen",
            )
        raise UpstreamUnavailableError(
            "zen request failed: all retries exhausted",
            upstream="zen",
        )

    async def _on_model_unsupported(
        self,
        payload: Dict[str, Any],
        fallback_applied: bool,
        exc: ZenModelNotSupportedError,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if (
            FALLBACK_MODEL_ENABLED
            and not fallback_applied
            and payload.get("model") != FALLBACK_MODEL
        ):
            logger.debug("zen model unsupported, fallback: %s", exc)
            alt = dict(payload)
            alt["model"] = FALLBACK_MODEL
            async for event in self.stream_chat(alt, _fallback_applied=True):
                yield event
            return
        raise exc
