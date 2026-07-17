from __future__ import annotations

"""Primary Qwen HTTP client implementation."""

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp

try:
    from src.core.dispatch.candidate import Candidate, make_id
except ModuleNotFoundError:
    from .compat.runtime import Candidate, make_id

try:
    from src.core.models_cache import ModelsCache
except ModuleNotFoundError:
    from .compat.runtime import ModelsCache

try:
    from src.core.proxy_selector import ProxySelector
except ModuleNotFoundError:
    from .compat.runtime import ProxySelector

from ..accounts import ACCOUNTS, Account
from .crypto.auth import AuthMixin
from .transport.chat_session import ChatSession, UploadMixin
from .transport.endpoints import (
    BASE_URL, CAPS, CHAT_PATH, COOKIE_REFRESH_INTERVAL, MODELS, MODELS_PATH,
    PROXY_SELECTOR_PERSIST_PATH, PERSIST_INTERVAL, SMART_PROXY_ENABLED,
    SSE_TIMEOUT, TASK_TIMERS_PATH, TTS_DIR,
)
from .crypto.crypto import WafBlockedError, build_headers, generate_cookies, generate_fingerprint
from .compat.platform import LogsMixin, ProxyState, extract_model_ids
from .compat.payloads import build_payload
from .storage.storage import load_persist, load_task_timers, save_persist, save_task_timers
from .transport.sse import StreamHandler
from .media.tts import MediaMixin, TtsService
from .media.video import VideoGenMixin, VideoService

class QwenClient(AuthMixin, UploadMixin, MediaMixin, VideoGenMixin, LogsMixin):
    """Current Qwen web client used by the adapter."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._account_states: Dict[str, Account] = {}
        self._candidates: List[Candidate] = []
        self._models: List[str] = list(MODELS)
        self._fp = generate_fingerprint()
        self._cookies: Dict[str, Any] = generate_cookies(self._fp)
        self._bg_tasks: List[asyncio.Task] = []
        self._closing = False
        self._active_chats: Dict[str, str] = {}
        self._models_cache = ModelsCache("qwen", MODELS, fetch_enabled=False)
        self._proxy_state = ProxyState()
        self._proxy_selector = ProxySelector(Path(PROXY_SELECTOR_PERSIST_PATH))
        self._relogin_log_buffer: List[str] = []
        self._relogin_flush_task: Optional[asyncio.Task] = None
        self._retry_log_buffer: List[str] = []
        self._retry_log_flush_task: Optional[asyncio.Task] = None
        self._login_fail_buffer: List[Tuple[str, str]] = []
        self._login_fail_flush_task: Optional[asyncio.Task] = None
        self._chat_session: Optional[ChatSession] = None
        self._tts_service: Optional[TtsService] = None
        self._video_service: Optional[VideoService] = None

    def get_models(self) -> List[str]:
        return list(self._models)

    def set_proxy_enabled(self, enabled: bool) -> None:
        self._proxy_state.set_enabled(enabled)

    def is_proxy_enabled(self) -> bool:
        return self._proxy_state.is_enabled()

    def _get_proxy_kwarg(self) -> Optional[str]:
        try:
            from src.core.config import get_config
        except ModuleNotFoundError:
            from .compat.runtime import get_config
        try:
            from src.core.server import get_proxy_server
        except ModuleNotFoundError:
            from .compat.runtime import get_proxy_server

        config = get_config()
        if not config.proxy.proxy_enabled:
            return None
        if self._proxy_state.override is True:
            return get_proxy_server() if config.platforms_proxy.is_platform_enabled("qwen") else None
        if self._proxy_state.override is False:
            return None
        if SMART_PROXY_ENABLED and self._proxy_selector.select():
            return get_proxy_server()
        return None

    async def init_immediate(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        self._init_accounts()
        await self._models_cache.load()
        self._models = list(self._models_cache.models)
        self._cookies = load_persist(self._account_states, self._cookies, self._proxy_state)
        self._rebuild_candidates()
        self._init_services(session)

    def _init_accounts(self) -> None:
        for account in ACCOUNTS:
            self._account_states[account.username] = Account(username=account.username, password=account.password)

    def _init_services(self, session: aiohttp.ClientSession) -> None:
        self._chat_session = ChatSession(session, self._get_proxy_kwarg, lambda: self._cookies, lambda: self._fp)
        self._tts_service = TtsService(
            session, self._get_proxy_kwarg, lambda: self._cookies, lambda: self._fp,
            self._chat_session.create, self._chat_session.send_placeholder_message, self._chat_session.cleanup,
        )
        self._video_service = VideoService(
            session, self._get_proxy_kwarg, lambda: self._cookies, self._chat_session.create, self._chat_session.cleanup,
        )

    async def background_setup(self) -> None:
        await self._initial_login_pass()
        self._bg_tasks.extend([
            asyncio.create_task(self._login_poll_loop()),
            asyncio.create_task(self._bg_cookie_refresh()),
            asyncio.create_task(self._bg_persist()),
            asyncio.create_task(self._models_cache.start_refresh_loop(
                self.fetch_remote_models, interval=86400, on_update=self._on_models_update,
            )),
        ])

    def update_models(self, models: List[str]) -> None:
        seen = set()
        self._models = [m for m in list(MODELS) + list(models) if m not in seen and not seen.add(m)]
        self._rebuild_candidates()

    async def close(self) -> None:
        self._closing = True
        self._cancel_log_tasks()
        await self._cancel_bg_tasks()
        self._save_persist()

    def _cancel_log_tasks(self) -> None:
        for attr_name, flush_name in [
            ("_relogin_flush_task", "_flush_relogin_buffer_now"),
            ("_retry_log_flush_task", "_flush_retry_log_buffer_now"),
            ("_login_fail_flush_task", "_flush_login_fail_buffer_now"),
        ]:
            task = getattr(self, attr_name, None)
            if task is not None and not task.done():
                task.cancel()
                setattr(self, attr_name, None)
            flush = getattr(self, flush_name, None)
            if callable(flush):
                flush()

    async def _cancel_bg_tasks(self) -> None:
        for task in self._bg_tasks:
            task.cancel()
        for task in self._bg_tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
        self._bg_tasks.clear()

    def _rebuild_candidates(self) -> None:
        self._candidates = [
            Candidate(
                id=make_id("qwen", account.username[:12]),
                platform="qwen",
                resource_id=account.username[:12],
                models=list(self._models),
                context_length=account.context_length,
                meta={"email": account.username, "token": account.token, "user_id": account.user_id},
                **CAPS,
            )
            for account in self._account_states.values()
            if account.is_login and account.token
        ]

    async def candidates(self) -> List[Candidate]:
        return list(self._candidates)

    async def ensure_candidates(self, count: int) -> int:
        return len(self._candidates)

    def _save_persist(self) -> None:
        save_persist(self._account_states, self._cookies, self._proxy_state)

    def _load_task_timers(self) -> Dict[str, float]:
        return load_task_timers(TASK_TIMERS_PATH)

    def _save_task_timers(self, timers: Dict[str, float]) -> None:
        save_task_timers(TASK_TIMERS_PATH, timers)

    async def _bg_persist(self) -> None:
        while not self._closing:
            await asyncio.sleep(PERSIST_INTERVAL)
            self._save_persist()

    async def refresh_models(self) -> None:
        await self._models_cache._do_refresh(self.fetch_remote_models, on_update=self._on_models_update)

    async def _on_models_update(self, models: List[str]) -> None:
        self._models = list(models)
        for cand in self._candidates:
            cand.models = list(models)
        if self._account_states:
            self._rebuild_candidates()
    async def _fetch_models_from_endpoint(self, endpoint: str, headers: Dict[str, str]) -> List[str]:
        try:
            async with self._session.get(
                endpoint, headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
                proxy=self._get_proxy_kwarg(),
            ) as response:
                if response.status != 200:
                    return []
                return extract_model_ids(await response.json(content_type=None))
        except Exception:
            return []

    async def fetch_remote_models(self) -> List[str]:
        token = self._get_any_valid_token()
        if not token:
            return []
        headers = {**build_headers(token), "Accept": "application/json"}
        for endpoint in [f"{BASE_URL}{MODELS_PATH}", f"{BASE_URL}/api/v1/models"]:
            models = await self._fetch_models_from_endpoint(endpoint, headers)
            if models:
                return models
        return []

    def _get_any_valid_token(self) -> Optional[str]:
        for account in self._account_states.values():
            if account.token:
                return account.token
        return None

    async def _create_chat(self, token: str, model: str, chat_type: str = "t2t") -> str:
        return await self._chat_session.create(token, model, chat_type)

    async def _cleanup_chat(self, chat_id: str, token: str) -> None:
        await self._chat_session.cleanup(chat_id, token)

    async def _send_placeholder_message(self, chat_id: str, token: str, model: str):
        return await self._chat_session.send_placeholder_message(chat_id, token, model)

    async def complete(
        self, candidate: Candidate, messages: List[Dict[str, Any]], model: str, stream: bool,
        *, thinking: bool = False, search: bool = False, tts: bool = False,
        upload_files: Optional[List[Tuple[bytes, str]]] = None, **kw: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                async for chunk in self._do_request(
                    candidate, messages, model, stream=stream, thinking=thinking,
                    search=search, tts=tts, upload_files=upload_files,
                ):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                self._log_retry(f"{'WAF ' if isinstance(exc, WafBlockedError) else ''}retry {attempt + 1}/3: {exc}")
        if last_error is not None:
            raise last_error

    async def _prepare_file_objects(
        self, messages: List[Dict[str, Any]], token: str, user_id: str,
        upload_files: Optional[List[Tuple[bytes, str]]] = None,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if upload_files:
            for data, name in upload_files:
                result.append(await self.upload_file(data, name, token, user_id))
        for uri in self._extract_base64_images(messages):
            result.append(await self.upload_file_from_base64(uri, token, user_id))
        return result

    def _build_request_args(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        chat_id: str,
        file_objects: List[Dict[str, Any]],
        thinking: bool,
        search: bool,
        token: str,
        stream: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        payload = build_payload(
            messages=messages,
            model=model,
            chat_id=chat_id,
            files=file_objects,
            thinking_enabled=thinking,
            auto_thinking=False,
            thinking_mode="Thinking" if thinking else "Fast",
            thinking_format="raw",
            auto_search=search,
            stream=stream,
        )
        headers = build_headers(token, chat_id=chat_id, include_sse=True, fingerprint=self._fp, cookies=self._cookies)
        return payload, headers

    async def _do_request(
        self,
        candidate: Candidate,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        stream: bool = True,
        thinking: bool = False,
        search: bool = False,
        tts: bool = False,
        upload_files: Optional[List[Tuple[bytes, str]]] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        token = str(candidate.meta.get("token", ""))
        user_id = str(candidate.meta.get("user_id", ""))
        if not token:
            raise RuntimeError("candidate token is missing")
        file_objects = await self._prepare_file_objects(messages, token, user_id, upload_files)
        chat_id = await self._chat_session.create(token, model, "t2t")
        self._active_chats[candidate.id] = chat_id
        proxy_used = self._get_proxy_kwarg() is not None
        request_start = time.time()
        success = False
        try:
            payload, headers = self._build_request_args(
                messages, model, chat_id, file_objects, thinking, search, token, stream,
            )
            async with self._session.post(
                f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(connect=10, total=SSE_TIMEOUT),
                proxy=self._get_proxy_kwarg(),
            ) as response:
                await self._check_response(response, candidate)
                handler = StreamHandler(self.download_image)
                async for item in handler.stream(response):
                    yield item
                if tts and handler.last_response_id:
                    await self.request_tts(chat_id, handler.last_response_id, token, TTS_DIR)
                success = True
        finally:
            self._active_chats.pop(candidate.id, None)
            asyncio.ensure_future(self._chat_session.cleanup(chat_id, token))
            if SMART_PROXY_ENABLED:
                if success:
                    self._proxy_selector.record(proxy_used, True, (time.time() - request_start) * 1000)
                else:
                    self._proxy_selector.record(proxy_used, False)

    async def _check_response(self, response: aiohttp.ClientResponse, candidate: Candidate) -> None:
        if response.status != 200:
            body = await response.text()
            self._handle_token_expiry(response.status, body, candidate)
            raise RuntimeError(f"chat HTTP {response.status}: {body[:300]}")
        if "text/html" in response.headers.get("Content-Type", ""):
            raise WafBlockedError("upstream returned HTML instead of SSE")

    def _handle_token_expiry(self, status: int, body: str, candidate: Candidate) -> None:
        """Handle token expiry by invalidating the account."""
        if status != 401 and "Token has expired" not in body and "unauthorized" not in body.lower():
            return
        email = str(candidate.meta.get("email", ""))
        if email in self._account_states:
            account = self._account_states[email]
            account.is_login = False
            account.token = ""
            self._rebuild_candidates()
            self._save_persist()
        raise RuntimeError(f"token expired: {body[:200]}")

    async def stop_generation(self, chat_id: str, token: str) -> bool:
        return await self._chat_session.stop(chat_id, token)

    async def delete_chat(self, chat_id: str, token: str) -> bool:
        return await self._chat_session.delete(chat_id, token)

    async def stop_candidate_generation(self, candidate: Candidate) -> bool:
        chat_id = self._active_chats.get(candidate.id)
        if not chat_id:
            return False
        return await self.stop_generation(chat_id, str(candidate.meta.get("token", "")))
