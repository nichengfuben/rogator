from __future__ import annotations

"""Chat session and upload helpers.

Merged from: chat_session.py, upload.py
"""

import asyncio
import base64
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from upstream.qwen.chat.routes import (
    BASE_URL,
    CHAT_PATH,
    DELETE_CHAT_PATH,
    GENERATED_IMAGE_DIR,
    NEW_CHAT_PATH,
    STS_TOKEN_PATHS,
    STOP_CHAT_PATH,
    USER_AGENT,
)
from upstream.qwen.auth.crypto import build_headers, build_stop_headers
from upstream.qwen.chat.upload.payload import build_new_chat_payload, build_payload, build_stop_payload
from upstream.qwen.chat.upload.storage import save_image_file
from upstream.qwen.chat.sse import parse_sse_event

_MAX_FILE_SIZES: Dict[str, int] = {
    "video": 500 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "image": 20 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}


class ChatSession:
    """Low-level chat lifecycle operations."""

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
        """Create a new chat and return its identifier."""
        url = f"{BASE_URL}{NEW_CHAT_PATH}"
        headers = build_headers(token, include_version=False)
        async with self._session.post(
            url,
            json=build_new_chat_payload(model, chat_type),
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Qwen create-chat failed: HTTP {response.status}: {(await response.text())[:300]}")
            data = await response.json()
            chat_id = (data.get("data") or {}).get("id", "")
            if not data.get("success") or not chat_id:
                raise RuntimeError(f"Qwen create-chat returned an invalid payload: {data}")
            return chat_id

    async def stop(self, chat_id: str, token: str) -> bool:
        """Stop generation for an active chat."""
        if not chat_id or not token:
            return False
        async with self._session.post(
            f"{BASE_URL}{STOP_CHAT_PATH}",
            json=build_stop_payload(chat_id),
            headers=build_stop_headers(token),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            return response.status in {200, 204}

    async def delete(self, chat_id: str, token: str) -> bool:
        """Delete a chat."""
        if not chat_id or not token:
            return False
        async with self._session.delete(
            f"{BASE_URL}{DELETE_CHAT_PATH.format(chat_id=chat_id)}",
            headers=build_headers(token, cookies=self._cookies()),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
            proxy=self._resolve_proxy(),
        ) as response:
            return response.status in {200, 204}

    async def cleanup(self, chat_id: str, token: str) -> None:
        """Delete a chat, suppressing transport failures."""
        try:
            await self.delete(chat_id, token)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return

    async def download_image(self, image_url: str, save_dir: str = GENERATED_IMAGE_DIR) -> Optional[str]:
        """Download an image asset to local storage."""
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
            return save_image_file(await response.read(), response.headers.get("Content-Type", "image/png"), save_dir)

    async def send_placeholder_message(self, chat_id: str, token: str, model: str) -> Tuple[Optional[str], str]:
        """Send a placeholder prompt and return ``(response_id, origin_text)``."""
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
        headers = build_headers(
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
    async def _consume_placeholder(resp: aiohttp.ClientResponse) -> Tuple[Optional[str], str]:
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


class UploadMixin:
    """Provide file upload and image-download capabilities."""

    async def _get_sts_credentials(
        self,
        token: str,
        filename: str,
        filesize: int,
        filetype: str,
    ) -> Dict[str, Any]:
        """Request temporary STS credentials for OSS upload."""
        from upstream.qwen.chat.upload.storage import build_oss_authorization

        headers = {
            "authorization": f"Bearer {token}",
            "content-type": "application/json;charset=UTF-8",
            "source": "web",
            "user-agent": USER_AGENT,
            "origin": BASE_URL,
            "referer": f"{BASE_URL}/",
            "accept": "application/json",
        }
        payload = {"filename": filename, "filesize": filesize, "filetype": filetype}
        last_error: Optional[Exception] = None
        for path in STS_TOKEN_PATHS:
            try:
                async with self._session.post(
                    f"{BASE_URL}{path}",
                    json=payload,
                    headers=headers,
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=15),
                    proxy=self._get_proxy_kwarg(),
                ) as response:
                    if response.status != 200:
                        last_error = RuntimeError(f"STS HTTP {response.status}: {(await response.text())[:200]}")
                        continue
                    data = await response.json()
                    creds = data.get("data", data)
                    if all(key in creds for key in {"access_key_id", "access_key_secret", "security_token"}):
                        return creds
                    last_error = RuntimeError(f"invalid STS payload: {data}")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"all STS endpoints failed: {last_error}")

    async def _upload_to_oss(
        self,
        file_data: bytes,
        content_type: str,
        creds: Dict[str, Any],
    ) -> str:
        """Upload bytes to OSS and return the final file URL."""
        from upstream.qwen.chat.upload.storage import build_oss_authorization

        file_url = str(creds.get("file_url", ""))
        object_key = str(creds.get("file_path", ""))
        parsed = urlparse(file_url)
        bucket_host = parsed.netloc
        bucket_name = bucket_host.split(".")[0]
        resource = f"/{bucket_name}/{object_key}"
        date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        oss_headers = {"x-oss-security-token": str(creds.get("security_token", ""))}
        authorization = build_oss_authorization(
            "PUT",
            content_type,
            date,
            oss_headers,
            resource,
            str(creds.get("access_key_id", "")),
            str(creds.get("access_key_secret", "")),
        )
        headers = {
            "Host": bucket_host,
            "Date": date,
            "Content-Type": content_type,
            "Content-Length": str(len(file_data)),
            "Authorization": authorization,
            "x-oss-security-token": str(creds.get("security_token", "")),
            "User-Agent": USER_AGENT,
        }
        async with self._session.put(
            f"https://{bucket_host}/{object_key}",
            data=file_data,
            headers=headers,
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            if response.status not in {200, 201}:
                raise RuntimeError(f"OSS PUT failed: HTTP {response.status}: {(await response.text())[:300]}")
        return file_url

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        token: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Upload one file and return the corresponding Qwen file object."""
        from upstream.qwen.chat.upload.storage import get_file_category, get_mime_type
        from upstream.qwen.chat.upload.storage import build_file_object

        content_type = get_mime_type(filename)
        file_type, _ = get_file_category(content_type)
        file_size = len(file_data)
        if file_size <= 0:
            raise RuntimeError(f"empty file: {filename}")
        limit = _MAX_FILE_SIZES.get(file_type, 20 * 1024 * 1024)
        if file_size > limit:
            raise RuntimeError(f"file too large: {filename} ({file_size} > {limit})")
        creds = await self._get_sts_credentials(token, filename, file_size, file_type)
        file_url = await self._upload_to_oss(file_data, content_type, creds)
        return build_file_object(
            file_id=str(creds.get("file_id", uuid.uuid4())),
            file_url=file_url,
            filename=filename,
            size=file_size,
            content_type=content_type,
            user_id=user_id,
        )

    async def upload_file_from_path(self, file_path: str, token: str, user_id: str) -> Dict[str, Any]:
        """Upload a local file by path."""
        if not os.path.exists(file_path):
            raise RuntimeError(f"file not found: {file_path}")
        return await self.upload_file(Path(file_path).read_bytes(), os.path.basename(file_path), token, user_id)

    async def upload_file_from_base64(self, data_uri: str, token: str, user_id: str) -> Dict[str, Any]:
        """Upload a base64 data URI as a file object."""
        from upstream.qwen.chat.upload.storage import DATA_URI_EXT_MAP

        if not data_uri.startswith("data:") or ";base64," not in data_uri:
            raise RuntimeError("invalid base64 data URI")
        header, encoded = data_uri.split(";base64,", 1)
        mime_type = header.split("data:", 1)[1]
        padding = (-len(encoded)) % 4
        if padding:
            encoded += "=" * padding
        filename = f"upload_{uuid.uuid4().hex[:8]}{DATA_URI_EXT_MAP.get(mime_type, '.bin')}"
        return await self.upload_file(base64.b64decode(encoded), filename, token, user_id)

    def _extract_base64_images(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract inline base64 images from OpenAI-style messages."""
        results: List[str] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    candidate = str(image_url.get("url", ""))
                else:
                    candidate = str(image_url)
                if candidate.startswith("data:"):
                    results.append(candidate)
        return results

    async def download_image(self, image_url: str, save_dir: str = GENERATED_IMAGE_DIR) -> Optional[str]:
        """Download an image and return the saved local path."""
        async with self._session.get(
            image_url,
            headers={
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
                "User-Agent": USER_AGENT,
            },
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=60),
            proxy=self._get_proxy_kwarg(),
        ) as response:
            if response.status != 200:
                return None
            return save_image_file(await response.read(), response.headers.get("Content-Type", "image/png"), save_dir)
