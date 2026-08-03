# src/platforms/deepseek/core/hif.py
"""DeepSeek HIF 服务令牌管理"""

import asyncio
import logging
import time
from typing import Any, Optional, Tuple

import aiohttp

from upstream.deepseek.lib.protocol.consts import (
    HIF_DLIQ_URL,
    HIF_LEIM_URL,
    HIF_REFRESH_INTERVAL,
)
from upstream.deepseek.lib.protocol.headers import build_basic_headers

logger = logging.getLogger(__name__)


async def fetch_hif_tokens(
    session: Any,
) -> Tuple[str, str, float]:

    headers = build_basic_headers()

    async def _get(url: str) -> str:
        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=False,
            ) as resp:
                if resp.status != 200:
                    logger.warning("HIF 令牌获取失败 %s: HTTP %d", url, resp.status)
                    return ""
                data = await resp.json()
                code = data.get("code", data.get("data", {}).get("code", -1))
                # DeepSeek HIF 有时把 biz_data 包在 data 下，有时平铺在根级
                biz = (
                    data.get("data", {}).get("biz_data", {}) if "data" in data else data
                )
                val = biz.get("value", "")
                return str(val) if val else ""
        except Exception as exc:
            logger.warning("HIF 令牌获取异常 %s: %s", url, exc)
            return ""

    leim, dliq = await asyncio.gather(_get(HIF_LEIM_URL), _get(HIF_DLIQ_URL))
    expire_at = time.time() + HIF_REFRESH_INTERVAL
    logger.debug("HIF 令牌刷新完成 leim=%s... dliq=%s...", leim[:8], dliq[:8])
    return leim, dliq, expire_at


class HifTokenManager:
    """HIF 令牌自动刷新管理器。"""

    def __init__(self) -> None:
        self._leim: str = ""
        self._dliq: str = ""
        self._expire_at: float = 0.0
        self._session: Optional[Any] = None

    def bind_session(self, session: Any) -> None:
        self._session = session

    def is_expired(self) -> bool:
        return time.time() >= self._expire_at - 60

    async def ensure_valid(self) -> Tuple[str, str]:
        if self.is_expired() and self._session is not None:
            self._leim, self._dliq, self._expire_at = await fetch_hif_tokens(
                self._session
            )
        return self._leim, self._dliq

    @property
    def leim(self) -> str:
        return self._leim

    @property
    def dliq(self) -> str:
        return self._dliq
