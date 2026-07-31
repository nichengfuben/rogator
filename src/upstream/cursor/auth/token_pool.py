from __future__ import annotations

"""自建 Token 服务：API Key 池、拉号与用量解析。"""

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from echotools.logger import get_logger

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
            logger.info("Cursor Key ??: %s -> %s", old.masked(), new.masked())
        return new

    def is_stale(self, s: KeyState) -> bool:
        return (time.time() - s.last_checked) >= self.refresh_interval

    def should_switch(self, s: KeyState) -> bool:
        """仅在 Key 失效或达到服务端日限额时切换；无日限时不因本地计数误杀。"""
        if not s.is_active:
            return True
        if s.daily_used is None:
            return False
        if s.daily_limit is None or s.daily_limit <= 0:
            return False
        # switch_threshold：日限额百分比（≤100）或绝对次数（>100）
        if self.threshold <= 100:
            line = s.daily_limit * (self.threshold / 100.0)
        else:
            line = float(self.threshold)
        return float(s.daily_used) >= line


class TokenServiceAPI:
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
        "is_unlimited": bool(data.get("is_unlimited", False)),
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
        raise ValueError("??? access_token ?? user_id")
    from upstream.cursor.setup.config import cursor_cli_user_agent

    headers = {
        "accept": "*/*",
        "referer": "https://cursor.com/agents",
        "user-agent": cursor_cli_user_agent(),
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
