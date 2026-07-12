from __future__ import annotations

"""Authentication and account background-management helpers."""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp

from accounts import Account
from .crypto import (
    BASE_URL,
    build_headers,
    build_login_headers,
    generate_cookies,
    generate_fingerprint,
    hash_password,
)

logger = logging.getLogger(__name__)

# Constants from endpoints.py used by auth
AUTH_BASE_URL = "https://auth.qwen.ai"
AUTH_API_PREFIX = "/api/v2"
CHAT_API_PREFIX = "/api/v2"
SIGNIN_PATH = f"{AUTH_API_PREFIX}/auths/signin"
SETTINGS_PATH = f"{CHAT_API_PREFIX}/user/settings"
COOKIE_REFRESH_INTERVAL = 1800
LOGIN_BATCH_SIZE = 3
LOGIN_POLL_INTERVAL = 300
LOGIN_POOL_SIZE = 8
LOGIN_SELECT_MIN = 2
LOGIN_SELECT_MAX = 5
TOKEN_EXPIRY_MARGIN = 600

DEFAULT_FULL_SETTINGS: Dict[str, Any] = {
    "ui": {
        "notificationEnabled": False,
        "theme": "dark",
        "language": "",
        "chatBubble": True,
        "showUsername": False,
        "widescreenMode": False,
        "title": {"auto": False},
        "autoTags": True,
        "largeTextAsFile": False,
        "splitLargeChunks": False,
        "scrollOnBranchChange": True,
        "responseAutoCopy": False,
        "models": [],
        "richTextInput": False,
    },
    "mcp_remind": False,
    "mcp": {
        "code-interpreter": False,
        "fire-crawl": False,
        "amap": False,
        "image-generation": False,
    },
    "memory": {
        "enable_memory": False,
        "enable_history_memory": False,
        "memory_version_reminder": False,
    },
    "reminder": {"project_version_reminder": False},
    "tts_speaker": {
        "speaker": "Cherry",
        "description": "一位阳光、积极、友好且自然的年轻女士",
        "url": "",
        "gender": "female",
    },
    "tts_speaker_v2": {
        "speaker": "Nini",
        "description": "像糯米糍一样软糯黏腻的嗓音",
        "url": "",
        "gender": "female",
        "is_personal": False,
        "speaker_id": "",
        "spk_name": "邻家妹妹",
    },
    "aipodcast": {"host": "", "guest": ""},
    "code_settings": {
        "custom_prompt": "",
        "diff_display": "split",
        "branch_format": "",
        "last_repo_choice": "",
        "last_branch_choice": "",
    },
    "manage_cookies": None,
    "personalization": {
        "name": "",
        "description": "",
        "style": None,
        "instruction": "",
        "enable_for_new_chat": False,
    },
    "tools_enabled": {
        "web_search": False,
        "web_extractor": False,
        "web_search_image": False,
        "image_gen_tool": True,
        "image_edit_tool": True,
        "code_interpreter": False,
        "bio": False,
        "history_retriever": False,
        "image_zoom_in_tool": False,
    },
}


class AuthMixin:
    """Mixin implementing Qwen account authentication workflows."""

    async def _login(self, account: Account) -> bool:
        account.password_hash = account.password_hash or hash_password(account.password)
        payload = {"email": account.username, "password": account.password_hash}
        async with self._session.post(
            f"{AUTH_BASE_URL}{SIGNIN_PATH}",
            json=payload,
            headers=build_login_headers(),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=30),
            proxy=self._get_proxy_kwarg(),
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Qwen 登录失败 HTTP {response.status}: {(await response.text())[:300]}")
            data = await response.json()
            token = ((data.get("data") or {}).get("access_token") or data.get("access_token") or "")
            if not token:
                raise RuntimeError(f"Qwen 登录响应缺少 token: {data}")
            account.token = token
            account.is_login = True
            account.last_login = time.time()
            account.token_expires = time.time() + 24 * 60 * 60
            return True

    async def _fetch_user_settings(self, account: Account) -> Dict[str, Any]:
        async with self._session.get(
            f"{BASE_URL}{SETTINGS_PATH}",
            headers=build_headers(account.token, cookies=self._cookies),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=30),
            proxy=self._get_proxy_kwarg(),
        ) as response:
            if response.status != 200:
                return {}
            return await response.json(content_type=None)

    async def _fetch_user_profile(self, account: Account) -> Dict[str, Any]:
        async with self._session.get(
            f"{AUTH_BASE_URL}/api/v2/user",
            headers=build_headers(account.token, cookies=self._cookies),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=30),
            proxy=self._get_proxy_kwarg(),
        ) as response:
            if response.status != 200:
                return {}
            return await response.json(content_type=None)

    async def _configure_account(self, account: Account) -> None:
        profile = await self._fetch_user_profile(account)
        profile_data = profile.get("data", profile) if isinstance(profile, dict) else {}
        account.user_id = str(profile_data.get("id") or profile_data.get("user_id") or account.user_id or "")

        settings = await self._fetch_user_settings(account)
        settings_data = settings.get("data", settings) if isinstance(settings, dict) else {}
        memory = settings_data.get("memory") if isinstance(settings_data, dict) else {}
        if isinstance(memory, dict):
            account.memory_disabled = not bool(memory.get("enabled", True))
        context_length = settings_data.get("context_length") if isinstance(settings_data, dict) else None
        if isinstance(context_length, int) and context_length > 0:
            account.context_length = context_length

    async def _save_default_settings(self, account: Account) -> None:
        try:
            async with self._session.put(
                f"{BASE_URL}{SETTINGS_PATH}",
                json=DEFAULT_FULL_SETTINGS,
                headers=build_headers(account.token, cookies=self._cookies),
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
                proxy=self._get_proxy_kwarg(),
            ) as response:
                await response.read()
        except Exception as exc:
            logger.debug("Qwen 默认设置下发���败 %s: %s", account.username[:6], exc)

    async def _login_and_configure(self, account: Account) -> None:
        await self._login(account)
        await self._configure_account(account)
        await self._save_default_settings(account)

    def _is_token_expired(self, account: Account) -> bool:
        if not account.token:
            return True
        if not account.token_expires:
            return False
        return account.token_expires <= time.time() + TOKEN_EXPIRY_MARGIN

    def _select_login_batch(self) -> List[Account]:
        pool = [acc for acc in self._account_states.values() if self._is_token_expired(acc) or not acc.is_login]
        if not pool:
            return []
        random.shuffle(pool)
        upper = min(LOGIN_SELECT_MAX, max(LOGIN_SELECT_MIN, LOGIN_BATCH_SIZE), len(pool))
        lower = min(LOGIN_SELECT_MIN, upper)
        size = upper if upper == lower else random.randint(lower, upper)
        return pool[:size]

    async def _initial_login_pass(self) -> None:
        batch = self._select_login_batch()
        if not batch:
            return
        semaphore = asyncio.Semaphore(LOGIN_POOL_SIZE)

        async def worker(acc: Account) -> None:
            async with semaphore:
                try:
                    if acc.token and not self._is_token_expired(acc):
                        acc.is_login = True
                        return
                    if acc.token and self._is_token_expired(acc):
                        self._log_queued_relogin(acc.username[:6])
                        acc.token = ""
                        acc.is_login = False
                    await self._login_and_configure(acc)
                except Exception as exc:
                    acc.is_login = False
                    self._log_login_failure(acc.username[:6], str(exc))

        await asyncio.gather(*(worker(acc) for acc in batch), return_exceptions=True)
        self._rebuild_candidates()
        self._save_persist()

    async def _login_poll_loop(self) -> None:
        timers = self._load_task_timers()
        last_run = timers.get("login_poll", 0)
        remaining = LOGIN_POLL_INTERVAL - (time.time() - last_run)
        while not self._closing:
            sleep_time = remaining if remaining > 0 else LOGIN_POLL_INTERVAL
            remaining = -1
            await asyncio.sleep(sleep_time)
            if self._closing:
                break
            try:
                batch = self._select_login_batch()
                if batch:
                    await self._process_login_batch(batch)
            except Exception as exc:
                logger.warning("Qwen 登录轮询异常: %s", exc)
            timers["login_poll"] = time.time()
            self._save_task_timers(timers)

    async def _process_login_batch(self, batch: List[Account]) -> None:
        """Process a batch of accounts for login."""
        network_breaker_hit = False
        for acc in batch:
            if self._closing or network_breaker_hit:
                break
            try:
                if acc.token and self._is_token_expired(acc):
                    acc.token = ""
                    acc.is_login = False
                await self._login_and_configure(acc)
            except Exception as exc:
                err_str = str(exc)
                self._log_login_failure(acc.username[:6], err_str)
                if not network_breaker_hit and (
                    "Cannot connect" in err_str
                    or "远程计算机拒绝" in err_str
                    or "连接" in err_str
                ):
                    network_breaker_hit = True
        self._rebuild_candidates()
        self._save_persist()

    async def _bg_cookie_refresh(self) -> None:
        while not self._closing:
            await asyncio.sleep(COOKIE_REFRESH_INTERVAL)
            if self._closing:
                break
            try:
                self._fp = generate_fingerprint()
                self._cookies = generate_cookies(self._fp)
                self._save_persist()
            except Exception as exc:
                logger.debug("Qwen Cookie 刷新失败: %s", exc)
