from __future__ import annotations

"""TTS service for the current Qwen web protocol.

Merged from: tts.py, media.py (TTS parts)
"""

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import aiohttp

from ..transport.routes import BASE_URL, TTS_DIR, TTS_PATH, TTS_TIMEOUT
from ..crypto.crypto import build_headers
from ..compat.payload import build_replace_content_payload, build_tts_payload
from ..storage.storage import save_wav_file

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def build_tts_headers(token: str, chat_id: str, fingerprint: str, cookies: dict) -> dict:
    """Build headers for TTS request."""
    headers = build_headers(
        token,
        chat_id=chat_id,
        include_sse=True,
        fingerprint=fingerprint,
        cookies=cookies,
    )
    headers["Accept"] = "*/*"
    return headers


async def process_tts_response(response: aiohttp.ClientResponse) -> List[str]:
    """Process SSE response and extract TTS fragments."""
    chunks: List[str] = []
    buffer = b""
    async for raw in response.content.iter_any():
        if not raw:
            continue
        buffer += raw
        lines = buffer.split(b"\n")
        buffer = lines[-1]
        for line_bytes in lines[:-1]:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].lstrip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            tts_fragment = delta.get("tts")
            if tts_fragment:
                chunks.append(tts_fragment)
            if delta.get("status") == "finished":
                break
    return chunks


def decode_and_save_wav(chunks: List[str], save_dir: str) -> Optional[str]:
    """Combine chunks, decode base64, and save as WAV file."""
    if not chunks:
        return None
    combined = "".join(chunks)
    padding = (-len(combined)) % 4
    if padding:
        combined += "=" * padding
    return save_wav_file(base64.b64decode(combined), save_dir)


class TtsService:
    """Encapsulate the end-to-end TTS flow."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        proxy_resolver: Callable[[], Optional[str]],
        cookies_provider: Callable[[], dict],
        fingerprint_provider: Callable[[], str],
        create_chat: Callable[[str, str, str], Awaitable[str]],
        get_response_id: Callable[[str, str, str], Awaitable[Tuple[Optional[str], str]]],
        cleanup_chat: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self._session = session
        self._resolve_proxy = proxy_resolver
        self._cookies = cookies_provider
        self._fingerprint = fingerprint_provider
        self._create_chat = create_chat
        self._get_response_id = get_response_id
        self._cleanup_chat = cleanup_chat

    async def synthesize(
        self,
        text: str,
        token: str,
        model: str = "qwen3-max",
        save_dir: str = TTS_DIR,
    ) -> Optional[str]:
        """Run the full placeholder-replace-synthesize TTS flow."""
        try:
            chat_id = await self._create_chat(token, model, "t2t")
            response_id, origin_text = await self._get_response_id(chat_id, token, model)
            if not response_id:
                return None
            if not await self.replace_message_content(chat_id, response_id, text, origin_text.strip(), token):
                return None
            return await self.request_tts(chat_id, response_id, token, save_dir)
        finally:
            if "chat_id" in locals():
                asyncio.ensure_future(self._cleanup_chat(chat_id, token))

    async def replace_message_content(
        self,
        chat_id: str,
        response_id: str,
        new_content: str,
        origin_content: str,
        token: str,
    ) -> bool:
        """Replace an assistant message before TTS synthesis."""
        url = f"{BASE_URL}/api/v2/chats/{chat_id}/messages/{response_id}"
        headers = build_headers(token, chat_id=chat_id, cookies=self._cookies())
        payload = build_replace_content_payload(new_content, origin_content)
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
            try:
                async with self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=30),
                    proxy=self._resolve_proxy(),
                ) as resp:
                    if resp.status == 200:
                        return True
                    logger.warning("内容替换失败 HTTP %d: %s", resp.status, (await resp.text())[:200])
            except Exception as exc:
                logger.warning("内容替换异常: %s", exc)
        return False

    async def request_tts(
        self,
        chat_id: str,
        response_id: str,
        token: str,
        save_dir: str = TTS_DIR,
    ) -> Optional[str]:
        """Request TTS audio and persist the decoded WAV file."""
        headers = build_tts_headers(token, chat_id, self._fingerprint(), self._cookies())
        async with self._session.post(
            f"{BASE_URL}{TTS_PATH}?chat_id={chat_id}",
            json=build_tts_payload(chat_id, response_id),
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=TTS_TIMEOUT),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                return None
            chunks = await process_tts_response(response)
        return decode_and_save_wav(chunks, save_dir)


class MediaMixin:
    """Mixin providing TTS synthesis helpers."""

    async def _replace_message_content(
        self,
        chat_id: str,
        response_id: str,
        new_content: str,
        origin_content: str,
        token: str,
    ) -> bool:
        """Replace an assistant message content before TTS."""
        url = f"{BASE_URL}/api/v2/chats/{chat_id}/messages/{response_id}"
        headers = build_headers(token, chat_id=chat_id, cookies=self._cookies)
        payload = build_replace_content_payload(new_content, origin_content)
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
            try:
                async with self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=30),
                    proxy=self._get_proxy_kwarg(),
                ) as resp:
                    if resp.status == 200:
                        return True
                    logger.warning("内容替换失败 HTTP %d: %s", resp.status, (await resp.text())[:200])
            except Exception as exc:
                logger.warning("内容替换异常: %s", exc)
        return False

    async def request_tts(
        self,
        chat_id: str,
        response_id: str,
        token: str,
        save_dir: str = TTS_DIR,
    ) -> Optional[str]:
        """Request TTS audio and persist the decoded WAV file."""
        headers = build_tts_headers(token, chat_id, self._fp, self._cookies)
        async with self._session.post(
            f"{BASE_URL}{TTS_PATH}?chat_id={chat_id}",
            json=build_tts_payload(chat_id, response_id),
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=TTS_TIMEOUT),
            proxy=self._get_proxy_kwarg(),
        ) as resp:
            if resp.status != 200:
                logger.warning("TTS 请求失败 HTTP %d", resp.status)
                return None
            chunks = await process_tts_response(resp)
        return decode_and_save_wav(chunks, save_dir)

    async def synthesize_tts(
        self,
        text: str,
        token: str,
        model: str = "qwen3-max",
        save_dir: str = TTS_DIR,
    ) -> Optional[str]:
        """Run the full placeholder-replace-synthesize TTS flow."""
        chat_id: Optional[str] = None
        try:
            chat_id = await self._create_chat(token, model, "t2t")
            response_id, origin_text = await self._send_placeholder_message(chat_id, token, model)
            if not response_id:
                return None
            ok = await self._replace_message_content(chat_id, response_id, text, origin_text.strip(), token)
            if not ok:
                return None
            return await self.request_tts(chat_id, response_id, token, save_dir)
        finally:
            if chat_id:
                asyncio.ensure_future(self._cleanup_chat(chat_id, token))
