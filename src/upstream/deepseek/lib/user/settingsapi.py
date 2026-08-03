from __future__ import annotations

"""DeepSeek client/settings 与登录后 warmup（对齐 HAR 2026-08）。"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST
from upstream.deepseek.lib.protocol.headers import build_basic_headers, build_headers

logger = logging.getLogger(__name__)

SETTINGS_SCOPES: Tuple[str, ...] = ("main", "model", "web_upgrade", "banner")


def collect_setting_ids(settings: Dict[str, Any]) -> List[int]:

    ids: List[int] = []
    for item in settings.values():
        if not isinstance(item, dict):
            continue
        raw = item.get("id")
        if raw is None:
            continue
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return ids


async def fetch_client_settings(
    session: aiohttp.ClientSession,
    token: str,
    device_id: str,
    scope: str,
    *,
    settings_token: str = "",
) -> Optional[Dict[str, Any]]:

    headers = build_basic_headers(token)
    if settings_token:
        headers["x-settings-token"] = settings_token
    url = "https://{}/api/v0/client/settings".format(DEFAULT_HOST)
    try:
        async with session.get(
            url,
            headers=headers,
            params={"did": device_id, "scope": scope},
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                logger.debug("client/settings [%s] HTTP %d", scope, resp.status)
                return None
            data = await resp.json()
            if data.get("code") != 0:
                return None
            inner = data.get("data") or {}
            if inner.get("biz_code", -1) != 0:
                return None
            return inner.get("biz_data")
    except Exception as exc:
        logger.debug("client/settings [%s] failed: %s", scope, exc)
        return None


async def report_client_settings(
    session: aiohttp.ClientSession,
    token: str,
    *,
    device_id: str,
    sso_id: str,
    settings_ids: List[int],
) -> bool:

    if not settings_ids or not sso_id:
        return False
    headers = build_basic_headers(token)
    payload = {
        "settings_ids": settings_ids,
        "did": device_id,
        "sso_id": sso_id,
    }
    url = "https://{}/api/v0/client/settings/report".format(DEFAULT_HOST)
    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            if data.get("code") != 0:
                return False
            inner = data.get("data") or {}
            return inner.get("biz_code", -1) == 0
    except Exception as exc:
        logger.warning("client/settings/report failed: %s", exc)
        return False


async def warmup_account_client(
    session: aiohttp.ClientSession,
    token: str,
    *,
    device_id: str,
    user_id: str,
) -> None:
    """登录后拉取各 scope settings 并 report，对齐 FE 启动序列。"""
    if not token or not device_id or not user_id:
        return
    collected: Set[int] = set()
    for scope in SETTINGS_SCOPES:
        biz = await fetch_client_settings(session, token, device_id, scope)
        if not biz:
            continue
        settings = biz.get("settings")
        if isinstance(settings, dict):
            collected.update(collect_setting_ids(settings))
    if not collected:
        logger.debug("deepseek settings warmup: no ids from scopes")
        return
    ok = await report_client_settings(
        session,
        token,
        device_id=device_id,
        sso_id=user_id,
        settings_ids=sorted(collected),
    )
    if ok:
        logger.info(
            "deepseek settings report ok (ids=%d, did=%s…)",
            len(collected),
            device_id[:8],
        )
    else:
        logger.debug("deepseek settings report skipped or failed")
