from __future__ import annotations

"""Star Cursor 拉号 / 用量监测 / API Key 轮询（无 Rogator 账号池）。"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from echotools.logger import get_logger

from upstream.cursor.auth_store import get_access_token, write_auth, write_token_backup
from upstream.cursor.config import starcursor_config

logger = get_logger("rogator")

USAGE_SUMMARY_URL = "https://cursor.com/api/usage-summary"


@dataclass
class KeyState:
    key: str
    name: str = ""
    is_active: bool = True
    daily_used: Optional[int] = None
    daily_limit: Optional[int] = None
    rpm: Optional[int] = None
    total_used: Optional[int] = None
    last_checked: float = 0.0
    errors: int = 0

    def masked(self) -> str:
        k = self.key
        return f"{k[:6]}...{k[-4:]}" if len(k) > 10 else k


class ApiError(Exception):
    def __init__(self, status: int, payload: Dict[str, Any]):
        self.status = status
        self.payload = payload or {}
        super().__init__(f"HTTP {status}: {payload}")


class KeyPool:
    def __init__(self, keys: List[str], threshold: int, refresh_interval: int):
        self._states: List[KeyState] = [KeyState(key=k) for k in keys]
        self._idx = 0
        self.threshold = threshold
        self.refresh_interval = refresh_interval

    @property
    def current(self) -> Optional[KeyState]:
        return self._states[self._idx] if self._states else None

    def all(self) -> List[KeyState]:
        return list(self._states)

    def is_empty(self) -> bool:
        return not self._states

    def switch_next(self) -> Optional[KeyState]:
        if not self._states:
            return None
        old = self.current
        self._idx = (self._idx + 1) % len(self._states)
        new = self.current
        if old and new and old.key != new.key:
            logger.info("Cursor Key 切换: %s -> %s", old.masked(), new.masked())
        return new

    def is_stale(self, s: KeyState) -> bool:
        return (time.time() - s.last_checked) >= self.refresh_interval

    def should_switch(self, s: KeyState) -> bool:
        if s.daily_used is None:
            return False
        if not s.is_active:
            return True
        if s.daily_limit is not None and s.daily_used >= s.daily_limit:
            return True
        return s.daily_used >= self.threshold


class StarCursorAPI:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def _get(
        self,
        session: aiohttp.ClientSession,
        path: str,
        key: Optional[str] = None,
    ) -> Dict[str, Any]:
        headers = {"X-API-Key": key} if key else {}
        async with session.get(
            f"{self.base_url}{path}", headers=headers, timeout=self._timeout,
        ) as r:
            try:
                data = await r.json(content_type=None)
            except Exception:
                data = {"error": await r.text()}
            if r.status != 200:
                raise ApiError(r.status, data)
            return data

    async def pull_token(self, session: aiohttp.ClientSession, key: str) -> Dict[str, Any]:
        return await self._get(session, "/api/v1/pull-token", key)

    async def key_status(self, session: aiohttp.ClientSession, key: str) -> Dict[str, Any]:
        return await self._get(session, "/api/v1/key-status", key)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    try:
        part = token.split(".")[1]
        pad = 4 - len(part) % 4
        if pad != 4:
            part += "=" * pad
        return json.loads(base64.urlsafe_b64decode(part))
    except Exception:
        return {}


def extract_user_id(access_token: str) -> str:
    sub = _decode_jwt_payload(access_token).get("sub", "")
    if not sub:
        return ""
    return sub.split("|", 1)[1] if "|" in sub else sub


def build_session_cookie(access_token: str) -> str:
    user_id = extract_user_id(access_token)
    if not user_id:
        return ""
    return f"{user_id}%3A%3A{access_token}"


def extract_tokens_from_pull(data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    ct = data.get("cursor_token") or {}
    if isinstance(ct, str):
        return ct, ct, "-", data.get("card_number") or "-"
    access = ct.get("access_token") or ct.get("accessToken") or data.get("access_token") or ""
    refresh = ct.get("refresh_token") or ct.get("refreshToken") or access
    email = ct.get("email") or data.get("email") or "-"
    card = data.get("card_number") or data.get("card") or "-"
    return access, refresh, email, card


def parse_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = data.get("individualUsage", {}).get("plan", {}) if isinstance(data, dict) else {}
    breakdown = plan.get("breakdown") or {}
    auto_pct = float(plan.get("autoPercentUsed") or 0)
    api_pct = float(plan.get("apiPercentUsed") or 0)
    total = float(breakdown.get("total") or 0)
    total_pct = (auto_pct + api_pct) / 2.0 if (auto_pct > 0 or api_pct > 0) else 0.0
    used = round(total * total_pct / 100.0, 2) if total > 0 else 0.0
    return {
        "total_pct": total_pct,
        "used": used,
        "remaining": max(total - used, 0),
        "is_unlimited": bool(data.get("isUnlimited", False)),
    }


def is_limit_reached(u: Dict[str, Any], threshold: float = 95.0) -> bool:
    if u.get("is_unlimited"):
        return False
    return float(u.get("total_pct", 0)) >= threshold


async def fetch_usage_summary(
    session: aiohttp.ClientSession,
    access_token: str,
    timeout: int = 20,
) -> Dict[str, Any]:
    cookie_val = build_session_cookie(access_token)
    if not cookie_val:
        raise ValueError("无法从 access_token 解析 user_id")
    headers = {
        "accept": "*/*",
        "referer": "https://cursor.com/agents",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "cookie": f"WorkosCursorSessionToken={cookie_val}",
    }
    tm = aiohttp.ClientTimeout(total=timeout)
    async with session.get(USAGE_SUMMARY_URL, headers=headers, timeout=tm) as r:
        try:
            data = await r.json(content_type=None)
        except Exception:
            data = {"error": await r.text()}
        if r.status != 200:
            raise ApiError(r.status, data)
        return data


class CursorTokenService:
    """Star Cursor 拉号 + auto 轮询；无 Rogator 账号池。"""

    def __init__(self) -> None:
        self._cfg = starcursor_config()
        self._pool = KeyPool(
            keys=list(self._cfg.get("api_keys") or []),
            threshold=int(self._cfg.get("switch_threshold", 80)),
            refresh_interval=int(self._cfg.get("status_refresh_interval", 30)),
        )
        self._api = StarCursorAPI(
            str(self._cfg.get("base_url") or ""),
            timeout=int(self._cfg.get("request_timeout", 15)),
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    def reload_config(self) -> None:
        self._cfg = starcursor_config()
        self._pool = KeyPool(
            keys=list(self._cfg.get("api_keys") or []),
            threshold=int(self._cfg.get("switch_threshold", 80)),
            refresh_interval=int(self._cfg.get("status_refresh_interval", 30)),
        )
        self._api = StarCursorAPI(
            str(self._cfg.get("base_url") or ""),
            timeout=int(self._cfg.get("request_timeout", 15)),
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def current_token(self) -> Optional[str]:
        return get_access_token()

    async def _refresh_key_state(self, s: KeyState) -> bool:
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

    async def _ensure_usable_key(self) -> Optional[KeyState]:
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

    async def _handle_acquire_error(self, e: ApiError, s: KeyState) -> None:
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
        for attempt in range(1, max_retry + 1):
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
                logger.error("Cursor 写入 auth.toml 失败")
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
            return await self.pull_until_acceptable(threshold)
        session = await self._ensure_session()
        try:
            raw = await fetch_usage_summary(session, token)
            u = parse_usage(raw)
            if is_limit_reached(u, threshold=threshold):
                logger.warning(
                    "Cursor 用量 %.1f%% >= %.1f%%，自动换号",
                    u["total_pct"],
                    threshold,
                )
                return await self.pull_until_acceptable(threshold)
            logger.debug("Cursor 用量正常 %.1f%%", u["total_pct"])
            return True
        except (ApiError, aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.warning("Cursor 查询用量失败: %s", e)
            return False
