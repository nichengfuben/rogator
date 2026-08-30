from __future__ import annotations

"""Zen 上游客户端：无账号、OpenAI 兼容 API + 代理池。"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from core.session.models_cache import ModelsCacheMixin
from core.transport.http import HttpTransportMixin, run_with_connection_retry, upstream_timeout
from server.formats import UpstreamUnavailableError
from upstream.zen.chat_stream import extract_error_info, post_chat_stream
from upstream.zen.models_sync import sync_zen_registry as _sync_zen_registry
from upstream.zen.openai_chat import build_headers
from upstream.zen.proxy import (
    NodeManager,
    ZenProxyError,
    build_proxy_pool_from_toml,
    is_proxy_error,
)
from upstream.zen.proxy import (
    load_dynamic_proxy_pool,
    merge_proxy_pools,
    save_proxy_pool_file,
)
from upstream.zen import (
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
        pool, pool_file, state_file, top_n, static_pool = build_proxy_pool_from_toml(raw)
        self.node_manager = NodeManager(pool, state_file)
        # 后台刷新所需状态
        section = raw.get("proxy") if isinstance(raw.get("proxy"), dict) else {}
        self._pool_file: str = pool_file
        # _static_pool 已包含 env + toml static；保存动态池前用它剔除静态项
        self._static_pool: List[Optional[str]] = static_pool
        self._dynamic_top_n: int = top_n
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
        # 启动时判断池是否为空：文件不存在或加载为空 → empty=True
        # 文件不存在要明确判定为空，避免日志误导
        pool_empty = (
            not os.path.exists(self._pool_file)
            or not load_dynamic_proxy_pool(self._pool_file)
        )
        logger.debug(
            "zen proxy refresh loop started: interval=%.0fs file=%s empty=%s",
            self._refresh_interval, self._pool_file, pool_empty,
        )
        # 启动时池为空 → 跳过首次 sleep 立即刷新一次；非空 → 先等一个间隔
        first_run = pool_empty
        while True:
            if not first_run:
                try:
                    await asyncio.sleep(self._refresh_interval)
                except asyncio.CancelledError:
                    return
            first_run = False
            try:
                dynamic = load_dynamic_proxy_pool(self._pool_file)
                merged = merge_proxy_pools(self._static_pool, dynamic, top_n=self._dynamic_top_n)
                await self.node_manager.reload_pool(merged)
                # 仅持久化动态池节点：剔除静态代理（env + toml）与 None，
                # 避免 proxy_pool.json 被静态配置污染
                static_set = {p for p in self._static_pool if p is not None}
                dynamic_only = [
                    p for p in merged if p is not None and p not in static_set
                ]
                if dynamic_only:
                    save_proxy_pool_file(self._pool_file, dynamic_only)
                elif os.path.exists(self._pool_file):
                    # 本轮 dynamic 为空：保留历史落盘，避免下一轮 startup 误判为
                    # "池为空需立即刷新"（参见 _proxy_refresh_loop 的 first_run 判定）。
                    logger.debug(
                        "zen proxy pool empty this round, keep existing %s",
                        self._pool_file,
                    )
                logger.debug(
                    "zen proxy pool refreshed: dynamic=%d merged=%d persisted=%d",
                    len(dynamic), len(merged), len(dynamic_only),
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
            _sync_zen_registry(self._models)
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

    async def _mute_and_switch(self, desc: str, *, reason: str) -> str:
        """静音当前节点并切换到下一个，返回新节点描述。"""
        await self.node_manager.mute_current()
        new_node = await self.node_manager.switch_next()
        logger.debug("zen %s via %s, switch -> %s", reason, desc, new_node)
        await self.reset_http_transport()
        return new_node

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
                if await self._on_upstream_unavailable(
                    desc, exc, is_rate_limit=("429" in str(exc)),
                ):
                    raise
                last_error = exc
                continue
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as exc:
                if not await self._on_general_stream_error(desc, exc):
                    raise
                last_error = exc
                continue
        raise self._build_final_stream_error(last_error)

    async def _on_upstream_unavailable(
        self,
        desc: str,
        exc: UpstreamUnavailableError,
        *,
        is_rate_limit: bool,
    ) -> bool:
        """UpstreamUnavailable 路径：mute 当前节点 + 切下一个；若全 mute 抛出 429。

        返回 ``True`` 表示应终止循环（已抛 429）；``False`` 表示继续重试。
        """
        reason = "429 rate limited" if is_rate_limit else "upstream unavailable"
        await self._mute_and_switch(desc, reason=reason)
        if self.node_manager.all_nodes_muted():
            raise UpstreamUnavailableError(
                "HTTP 429 - Rate limit exceeded", upstream="zen",
            ) from exc
        return False

    async def _on_general_stream_error(
        self,
        desc: str,
        exc: BaseException,
    ) -> bool:
        """普通异常：代理错误走 mute+switch；其余仅 reset transport。返回 ``True`` 继续重试。"""
        if isinstance(exc, ZenProxyError) or is_proxy_error(exc):
            await self._mute_and_switch(desc, reason="proxy error")
            if self.node_manager.all_nodes_muted():
                raise UpstreamUnavailableError(
                    "HTTP 429 - All upstream nodes rate limited", upstream="zen",
                ) from exc
            return True
        logger.debug(
            "zen attempt via %s failed: %s", desc, exc,
        )
        await self.reset_http_transport()
        return True

    def _build_final_stream_error(self, last_error: Optional[Exception]) -> Exception:
        if last_error is not None and "429" in str(last_error):
            return UpstreamUnavailableError(
                "HTTP 429 - Rate limit exceeded", upstream="zen",
            )
        return UpstreamUnavailableError(
            "zen request failed: all retries exhausted", upstream="zen",
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
