from __future__ import annotations

"""Chat session helpers."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from upstream.qwen.auth.crypto import build_headers_async, build_stop_headers_async
from upstream.qwen.chat.routes import (
    BASE_URL,
    CHAT_PATH,
    DELETE_CHAT_PATH,
    GENERATED_IMAGE_DIR,
    NEW_CHAT_PATH,
    STOP_CHAT_PATH,
)
from upstream.qwen.chat.upload.parse import parse_sse_event
from upstream.qwen.chat.upload.payload import (
    build_new_chat_payload,
    build_payload,
    build_stop_payload,
)
from upstream.qwen.chat.upload.storage import save_image_file


class ChatSession:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        proxy_resolver: Callable[[], Optional[str]],
        cookies_provider: Callable[[], Dict[str, Any]],
        fingerprint_provider: Callable[[], str],
    ) -> None:
        self._session = session
        self._resolve_proxy = proxy_resolver
        self._cookies = cookies_provider
        self._fingerprint = fingerprint_provider

    async def create(self, token: str, model: str, chat_type: str = "t2t") -> str:
        url = f"{BASE_URL}{NEW_CHAT_PATH}"
        headers = await build_headers_async(token, include_version=True, api_path=NEW_CHAT_PATH)
        async with self._session.post(
            url,
            json=build_new_chat_payload(model, chat_type),
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Qwen create-chat failed: HTTP {response.status}: {(await response.text())[:300]}"
                )
            data = await response.json()
            chat_id = (data.get("data") or {}).get("id", "")
            if not data.get("success") or not chat_id:
                raise RuntimeError(
                    f"Qwen create-chat returned an invalid payload: {data}"
                )
            return chat_id

    async def stop(self, chat_id: str, token: str, response_id: str = "") -> bool:
        if not chat_id or not token:
            return False
        async with self._session.post(
            f"{BASE_URL}{STOP_CHAT_PATH}",
            json=build_stop_payload(chat_id, response_id),
            headers=await build_stop_headers_async(token),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            return response.status in {200, 204}

    async def delete(self, chat_id: str, token: str) -> bool:
        if not chat_id or not token:
            return False
        async with self._session.delete(
            f"{BASE_URL}{DELETE_CHAT_PATH.format(chat_id=chat_id)}",
            headers=await build_headers_async(token, cookies=self._cookies()),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            return response.status in {200, 204}

    async def cleanup(self, chat_id: str, token: str) -> None:
        try:
            await self.delete(chat_id, token)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return

    async def download_image(
        self, image_url: str, save_dir: str = GENERATED_IMAGE_DIR
    ) -> Optional[str]:
        async with self._session.get(
            image_url,
            headers={
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
            },
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=60),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                return None
            return save_image_file(
                await response.read(),
                response.headers.get("Content-Type", "image/png"),
                save_dir,
            )

    async def send_placeholder_message(
        self, chat_id: str, token: str, model: str
    ) -> Tuple[Optional[str], str]:
        payload = build_payload(
            messages=[{"role": "user", "content": "注意：啥都不要说，直接输出\\即可"}],
            model=model,
            chat_id=chat_id,
            thinking_enabled=False,
            auto_thinking=False,
            thinking_mode="Fast",
            thinking_format="raw",
            stream=True,
        )
        headers = await build_headers_async(
            token,
            chat_id=chat_id,
            include_sse=True,
            fingerprint=self._fingerprint(),
            cookies=self._cookies(),
        )
        async with self._session.post(
            f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}",
            json=payload,
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=60),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                return None, ""
            return await self._consume_placeholder(response)

    @staticmethod
    async def _consume_placeholder(
        resp: aiohttp.ClientResponse,
    ) -> Tuple[Optional[str], str]:
        response_id: Optional[str] = None
        origin_text = ""
        buffer = b""
        async for raw in resp.content.iter_any():
            if not raw:
                continue
            buffer += raw
            lines = buffer.split(b"\n")
            buffer = lines[-1]
            for line_bytes in lines[:-1]:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                event = parse_sse_event(line[5:].lstrip())
                if event is None:
                    continue
                if event.get("type") == "response_created":
                    response_id = event.get("response_id")
                elif event.get("type") == "answer":
                    origin_text += event.get("content", "")
        return response_id, origin_text
