from __future__ import annotations

"""自建 Token 服务：拉号 / 用量监测 / API Key 轮询（无 Rogator 账号池）。"""

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp

from server.retry.http_client import client_session

from echotools.logger import get_logger

from upstream.cursor.auth.store import get_access_token, write_auth, write_token_backup
from upstream.cursor.auth.token_pool import (
    ApiError,
    KeyPool,
    TokenServiceAPI,
    extract_tokens_from_pull,
    extract_user_id,
    fetch_usage_summary,
    is_limit_reached,
    parse_usage,
)
from upstream.cursor.setup.config import token_service_config

logger = get_logger("rogator")


class CursorTokenService:
    """自建 Token 服务拉号 + auto 轮询；无 Rogator 账号池。"""

    def __init__(self) -> None:
        self._cfg = token_service_config()
        self._pool = KeyPool(
            keys=list(self._cfg.get("api_keys") or []),
            threshold=int(self._cfg.get("switch_threshold", 80)),
            refresh_interval=int(self._cfg.get("status_refresh_interval", 30)),
        )
        self._api = TokenServiceAPI(
            str(self._cfg.get("base_url") or ""),
            timeout=int(self._cfg.get("request_timeout", 15)),
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self.last_usage_pct: float = 0.0

    def reload_config(self) -> None:
        self._cfg = token_service_config()
        self._pool = KeyPool(
            keys=list(self._cfg.get("api_keys") or []),
            threshold=int(self._cfg.get("switch_threshold", 80)),
            refresh_interval=int(self._cfg.get("status_refresh_interval", 30)),
        )
        self._api = TokenServiceAPI(
            str(self._cfg.get("base_url") or ""),
            timeout=int(self._cfg.get("request_timeout", 15)),
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = client_session()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def current_token(self) -> Optional[str]:
        return get_access_token()

    async def _refresh_key_state(self, s) -> bool:
        session = await self._ensure_session()
        try:
            data = await self._api.key_status(session, s.key)
            s.name = data.get("name", s.name)
            s.is_active = data.get("is_active", s.is_active)
            s.daily_used = data.get("daily_used")
            s.daily_limit = data.get("daily_limit")
            s.rpm = data.get("rate_limit_per_minute")
            s.total_used = data.get("total_used")
            s.last_checked = time.time()
            s.errors = 0
            return True
        except (ApiError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Cursor Key[%s] 刷新失败: %s", s.masked(), exc)
            s.errors += 1
            return False

    async def _ensure_usable_key(self):
        total = len(self._pool.all())
        if total == 0:
            return None
        for _ in range(total):
            s = self._pool.current
            if s is None:
                return None
            if self._pool.is_stale(s):
                await self._refresh_key_state(s)
            if self._pool.should_switch(s):
                logger.warning(
                    "Cursor Key[%s] daily_used=%s >= 阈值 %s，切换",
                    s.masked(),
                    s.daily_used,
                    self._pool.threshold,
                )
                self._pool.switch_next()
                continue
            return s
        return None

    async def _handle_acquire_error(self, e: ApiError, s) -> None:
        if e.status == 403:
            payload = e.payload
            if payload.get("error") == "Daily limit reached":
                s.daily_used = payload.get("daily_used", s.daily_used)
                s.daily_limit = payload.get("daily_limit", s.daily_limit)
                s.last_checked = time.time()
                logger.warning(
                    "Cursor Key[%s] 每日上限 (%s/%s)，切换",
                    s.masked(),
                    s.daily_used,
                    s.daily_limit,
                )
            else:
                s.is_active = False
                logger.error("Cursor Key[%s] 已被禁用", s.masked())
            self._pool.switch_next()
        elif e.status == 429:
            wait = e.payload.get("retry_after", 5)
            await asyncio.sleep(wait)
        elif e.status == 503:
            await asyncio.sleep(3)
        elif e.status == 500:
            logger.error("Cursor 卡密尝试全部失败，切换 Key")
            self._pool.switch_next()
        elif e.status == 401:
            s.is_active = False
            logger.error("Cursor Key[%s] 无效，切换", s.masked())
            self._pool.switch_next()
        else:
            await asyncio.sleep(2)

    async def _acquire_token(self) -> Optional[Dict[str, Any]]:
        max_retry = int(self._cfg.get("max_retry_per_pull", 3))
        session = await self._ensure_session()
        for _attempt in range(1, max_retry + 1):
            s = await self._ensure_usable_key()
            if s is None:
                logger.error("Cursor: 所有 API Key 均不可用")
                return None
            try:
                result = await self._api.pull_token(session, s.key)
                if s.daily_used is not None:
                    s.daily_used += 1
                logger.info(
                    "Cursor Token 拉取成功 Key[%s] card=%s",
                    s.masked(),
                    result.get("card_number", "?"),
                )
                return result
            except ApiError as e:
                await self._handle_acquire_error(e, s)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("Cursor 拉号网络异常: %s", e)
                await asyncio.sleep(2)
        return None

    async def pull_and_write(self) -> bool:
        async with self._lock:
            result = await self._acquire_token()
            if not result:
                return False
            access, refresh, email, card = extract_tokens_from_pull(result)
            if not access:
                logger.error("Cursor pull-token 无 access_token")
                return False
            if not write_auth(access, refresh):
                logger.error("Cursor 写入 auth.json 失败")
                return False
            write_token_backup(result)
            logger.info(
                "Cursor auth 已更新 uid=%s email=%s card=%s",
                extract_user_id(access)[:8] or "-",
                email,
                card,
            )
            return True

    async def _verify_usage(self, threshold: float, attempt: int, attempts: int) -> bool:
        token = get_access_token()
        if not token:
            return False
        session = await self._ensure_session()
        try:
            raw = await fetch_usage_summary(session, token)
            u = parse_usage(raw)
            if not is_limit_reached(u, threshold=threshold):
                logger.info(
                    "Cursor 换号成功 用量 %.1f%% < %.1f%%",
                    u["total_pct"],
                    threshold,
                )
                return True
            if attempt < attempts:
                logger.warning(
                    "Cursor 新号仍超阈值 %.1f%% >= %.1f%%，继续换号",
                    u["total_pct"],
                    threshold,
                )
        except (ApiError, aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.warning("Cursor 校验用量失败: %s", e)
            return False
        return False

    async def pull_until_acceptable(self, threshold: Optional[float] = None) -> bool:
        threshold = float(
            threshold if threshold is not None else self._cfg.get("usage_threshold", 90.0)
        )
        attempts = int(self._cfg.get("max_retry_per_pull", 3))
        for attempt in range(1, attempts + 1):
            if not await self.pull_and_write():
                return False
            if await self._verify_usage(threshold, attempt, attempts):
                return True
        return False

    async def auto_check_once(self) -> bool:
        """单次 auto 监测：无 token 则拉号；超阈值则换号。"""
        self.reload_config()
        threshold = float(self._cfg.get("usage_threshold", 90.0))
        token = get_access_token()
        if not token:
            logger.debug("Cursor 本地无 Token，自动拉号...")
            ok = await self.pull_until_acceptable(threshold)
            return ok
        session = await self._ensure_session()
        try:
            raw = await fetch_usage_summary(session, token)
            u = parse_usage(raw)
            self.last_usage_pct = float(u.get("total_pct", 0.0))
            if is_limit_reached(u, threshold=threshold):
                logger.warning(
                    "Cursor 用量 %.1f%% >= %.1f%%，自动换号",
                    u["total_pct"],
                    threshold,
                )
                return await self.pull_until_acceptable(threshold)
            return True
        except (ApiError, aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.warning("Cursor 查询用量失败: %s", e)
            return False
