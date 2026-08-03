from __future__ import annotations

"""QwenClient 的文件与多模态媒体上传能力。"""

import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from upstream.qwen.auth.crypto import build_headers
from upstream.qwen.chat.routes import (
    BASE_URL,
    GENERATED_IMAGE_DIR,
    GENERATED_VIDEO_DIR,
    USER_AGENT,
)
from upstream.qwen.chat.store import QwenSession
from upstream.qwen.chat.upload.oss import upload_to_oss
from upstream.qwen.chat.upload.parse import wait_file_parsed
from upstream.qwen.chat.upload.storage import (
    DATA_URI_EXT_MAP,
    build_file_object,
    get_file_category,
    get_mime_type,
    save_image_file,
    save_video_file,
)

logger = logging.getLogger("rogator")

_MAX_FILE_SIZES: Dict[str, int] = {
    "video": 500 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "image": 20 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}


class UploadMixin:
    async def _maybe_parse_document(
        self,
        session: QwenSession,
        file_obj: Dict[str, Any],
    ) -> None:
        if file_obj.get("file_class") != "document":
            return
        file_id = str(file_obj.get("id") or "")
        if not file_id:
            return
        ok = await wait_file_parsed(self, session, file_id)
        if not ok:
            logger.warning(
                "Document parse failed or timed out: %s",
                file_obj.get("name", file_id[:8]),
            )

    async def _request_sts_token(
        self, path: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BASE_URL}{path}",
                json=payload,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                creds = data.get("data", data)
                if all(
                    k in creds
                    for k in ("access_key_id", "access_key_secret", "security_token")
                ):
                    return creds
                return None

    async def _get_sts_credentials(
        self, session: QwenSession, filename: str, filesize: int, filetype: str
    ) -> Dict[str, Any]:
        headers = build_headers(session.token)
        headers.update(
            {
                "Content-Type": "application/json;charset=UTF-8",
                "Accept": "application/json",
            }
        )
        payload = {"filename": filename, "filesize": filesize, "filetype": filetype}
        for path in ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]:
            try:
                creds = await self._request_sts_token(path, payload, headers)
                if creds:
                    return creds
            except Exception:
                continue
        raise RuntimeError("All STS endpoints failed")

    async def upload_file(
        self, session: QwenSession, file_data: bytes, filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        content_type = get_mime_type(filename)
        file_type, _ = get_file_category(content_type)
        file_size = len(file_data)
        limit = _MAX_FILE_SIZES.get(file_type, 20 * 1024 * 1024)
        if file_size > limit:
            raise RuntimeError(f"file too large: {filename} ({file_size} > {limit})")
        creds = await self._get_sts_credentials(session, filename, file_size, file_type)
        file_url = await upload_to_oss(file_data, content_type, creds)
        file_obj = build_file_object(
            file_id=str(creds.get("file_id", uuid.uuid4())),
            file_url=file_url,
            filename=filename,
            size=file_size,
            content_type=content_type,
            user_id=session.user_id,
        )
        await self._maybe_parse_document(session, file_obj)
        return file_url, file_obj

    async def upload_file_from_base64(
        self, session: QwenSession, data_uri: str
    ) -> Tuple[str, Dict[str, Any]]:

        if not data_uri.startswith("data:") or ";base64," not in data_uri:
            raise RuntimeError("invalid base64 data URI")
        header, encoded = data_uri.split(";base64,", 1)
        mime_type = header.split("data:", 1)[1]
        padding = (-len(encoded)) % 4
        if padding:
            encoded += "=" * padding
        filename = (
            f"upload_{uuid.uuid4().hex[:8]}{DATA_URI_EXT_MAP.get(mime_type, '.bin')}"
        )
        return await self.upload_file(session, base64.b64decode(encoded), filename)

    @staticmethod
    def extract_base64_images(messages: List[Dict[str, Any]]) -> List[str]:

        results: List[str] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                candidate = (
                    str(image_url.get("url", ""))
                    if isinstance(image_url, dict)
                    else str(image_url)
                )
                if candidate.startswith("data:"):
                    results.append(candidate)
        return results

    async def download_image(
        self, image_url: str, save_dir: str = GENERATED_IMAGE_DIR
    ) -> Optional[str]:
        """下载图片并保存到本地，返回保存路径。"""
        async with aiohttp.ClientSession() as s:
            async with s.get(
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
            ) as resp:
                if resp.status != 200:
                    return None
                return save_image_file(
                    await resp.read(),
                    resp.headers.get("Content-Type", "image/png"),
                    save_dir,
                )

    async def upload_file_from_url(
        self, session: QwenSession, media_url: str
    ) -> Tuple[str, Dict[str, Any]]:

        async with aiohttp.ClientSession() as s:
            async with s.get(
                media_url,
                headers={
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "User-Agent": USER_AGENT,
                },
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"download file failed: HTTP {resp.status}")
                data = await resp.read()
                content_type = resp.headers.get(
                    "Content-Type", "application/octet-stream"
                ).split(";", 1)[0]
        ext = DATA_URI_EXT_MAP.get(content_type, ".bin")
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        return await self.upload_file(session, data, filename)

    async def upload_file_from_path(
        self,
        session: QwenSession,
        file_path: str,
    ) -> Tuple[str, Dict[str, Any]]:

        if not os.path.exists(file_path):
            raise RuntimeError(f"file not found: {file_path}")
        data = Path(file_path).read_bytes()
        return await self.upload_file(session, data, os.path.basename(file_path))

    @staticmethod
    def extract_remote_media_urls(messages: List[Dict[str, Any]]) -> List[str]:

        results: List[str] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                url_obj: Any = None
                if part_type == "image_url":
                    url_obj = part.get("image_url")
                elif part_type == "video_url":
                    url_obj = part.get("video_url")
                elif part_type == "input_audio":
                    audio_obj = part.get("input_audio") or {}
                    url_obj = (
                        audio_obj.get("url")
                        if isinstance(audio_obj, dict)
                        else audio_obj
                    )
                if isinstance(url_obj, dict):
                    candidate = str(url_obj.get("url", "") or "")
                else:
                    candidate = str(url_obj or "")
                if candidate and not candidate.startswith("data:"):
                    results.append(candidate)
        return results

    async def download_video(
        self, video_url: str, save_dir: str = GENERATED_VIDEO_DIR
    ) -> Optional[str]:

        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    video_url,
                    headers={
                        "Accept": "*/*",
                        "Connection": "keep-alive",
                        "Origin": BASE_URL,
                        "Referer": f"{BASE_URL}/",
                        "User-Agent": USER_AGENT,
                    },
                    ssl=False,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Download video failed: HTTP %d", resp.status)
                        return None
                    return save_video_file(await resp.read(), save_dir)
        except Exception as exc:
            logger.warning("Download video exception: %s", exc)
            return None

    async def prepare_message_files(
        self,
        session: QwenSession,
        messages: List[Dict[str, Any]],
        extra_files: Optional[List[Tuple[bytes, str]]] = None,
    ) -> List[Dict[str, Any]]:
        # 单个文件上传失败会记录日志但不中断其余文件，最大程度保留可用多模态输入。
        file_objects: List[Dict[str, Any]] = []
        for data, name in extra_files or []:
            try:
                _, file_obj = await self.upload_file(session, data, name)
                file_objects.append(file_obj)
            except Exception as exc:
                logger.warning("Upload extra file %s failed: %s", name, exc)
        for data_uri in self.extract_base64_images(messages):
            try:
                _, file_obj = await self.upload_file_from_base64(session, data_uri)
                file_objects.append(file_obj)
            except Exception as exc:
                logger.warning("Upload inline image failed: %s", exc)
        for media_url in self.extract_remote_media_urls(messages):
            try:
                _, file_obj = await self.upload_file_from_url(session, media_url)
                file_objects.append(file_obj)
            except Exception as exc:
                logger.warning("Upload remote media %s failed: %s", media_url, exc)
        return file_objects
