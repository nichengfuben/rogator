from __future__ import annotations

"""简化版 QwenClient 的文件/图片上传能力。"""

import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from core.transport.routes import BASE_URL, GENERATED_IMAGE_DIR, GENERATED_VIDEO_DIR, USER_AGENT
from core.crypto.crypto import build_headers
from core.storage.storage import (
    DATA_URI_EXT_MAP,
    build_file_object,
    get_file_category,
    get_mime_type,
    save_image_file,
    save_video_file,
)
from server.client.oss import upload_to_oss
from server.client.session_store import QwenSession

logger = logging.getLogger("rogator")

_MAX_FILE_SIZES: Dict[str, int] = {
    "video": 500 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "image": 20 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}


class UploadMixin:
    """为简化版 QwenClient 提供文件上传与 base64 图片上传能力。"""

    async def _request_sts_token(self, path: str, payload: Dict[str, Any],
                                  headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BASE_URL}{path}", json=payload, headers=headers, ssl=False,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                creds = data.get("data", data)
                if all(k in creds for k in ("access_key_id", "access_key_secret", "security_token")):
                    return creds
                return None

    async def _get_sts_credentials(self, session: QwenSession, filename: str, filesize: int, filetype: str) -> Dict[str, Any]:
        headers = build_headers(session.token)
        headers.update({"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"})
        payload = {"filename": filename, "filesize": filesize, "filetype": filetype}
        for path in ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]:
            try:
                creds = await self._request_sts_token(path, payload, headers)
                if creds:
                    return creds
            except Exception:
                continue
        raise RuntimeError("All STS endpoints failed")

    async def upload_file(self, session: QwenSession, file_data: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
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
        return file_url, file_obj

    async def upload_file_from_base64(self, session: QwenSession, data_uri: str) -> Tuple[str, Dict[str, Any]]:
        """将消息中的 base64 图片上传为 Qwen 文件对象。"""
        if not data_uri.startswith("data:") or ";base64," not in data_uri:
            raise RuntimeError("invalid base64 data URI")
        header, encoded = data_uri.split(";base64,", 1)
        mime_type = header.split("data:", 1)[1]
        padding = (-len(encoded)) % 4
        if padding:
            encoded += "=" * padding
        filename = f"upload_{uuid.uuid4().hex[:8]}{DATA_URI_EXT_MAP.get(mime_type, '.bin')}"
        return await self.upload_file(session, base64.b64decode(encoded), filename)

    @staticmethod
    def extract_base64_images(messages: List[Dict[str, Any]]) -> List[str]:
        """从 OpenAI 风格消息中提取内联 base64 图片。"""
        results: List[str] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                candidate = str(image_url.get("url", "")) if isinstance(image_url, dict) else str(image_url)
                if candidate.startswith("data:"):
                    results.append(candidate)
        return results

    async def download_image(self, image_url: str, save_dir: str = GENERATED_IMAGE_DIR) -> Optional[str]:
        """下载图片并保存到本地，返回保存路径。用于 Qwen 生成图的回存。"""
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
                return save_image_file(await resp.read(), resp.headers.get("Content-Type", "image/png"), save_dir)

    async def upload_file_from_url(self, session: QwenSession, image_url: str) -> Tuple[str, Dict[str, Any]]:
        """下载远程图片 URL 并上传为 Qwen 文件对象。"""
        async with aiohttp.ClientSession() as s:
            async with s.get(
                image_url,
                headers={
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "User-Agent": USER_AGENT,
                },
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"download image failed: HTTP {resp.status}")
                data = await resp.read()
                content_type = resp.headers.get("Content-Type", "image/png").split(";", 1)[0]
        ext = DATA_URI_EXT_MAP.get(content_type, ".jpg")
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        return await self.upload_file(session, data, filename)

    async def upload_file_from_path(
        self, session: QwenSession, file_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """上传本地磁盘上的文件（按路径读取），返回 (file_url, file_obj)。"""
        if not os.path.exists(file_path):
            raise RuntimeError(f"file not found: {file_path}")
        data = Path(file_path).read_bytes()
        return await self.upload_file(session, data, os.path.basename(file_path))

    def extract_image_urls(self, messages: List[Dict[str, Any]]) -> List[str]:
        """从消息中提取非 base64 的远程图片 URL（用于以 URL 形式直传给 Qwen 的场景）。"""
        results: List[str] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                candidate = str(image_url.get("url", "")) if isinstance(image_url, dict) else str(image_url)
                if candidate and not candidate.startswith("data:"):
                    results.append(candidate)
        return results

    async def download_video(self, video_url: str, save_dir: str = GENERATED_VIDEO_DIR) -> Optional[str]:
        """下载生成的视频并保存到本地，返回保存路径。失败时记录日志并返回 None。"""
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
        """汇总消息中所有需要上传的内容（额外文件 + 内联 base64 图片），

        返回按顺序上传后的 Qwen 文件对象列表。单个文件上传失败会记录日志但
        不中断其余文件的处理，最大程度保留可用的多模态输入。
        """
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
        for image_url in self.extract_image_urls(messages):
            try:
                _, file_obj = await self.upload_file_from_url(session, image_url)
                file_objects.append(file_obj)
            except Exception as exc:
                logger.warning("Upload remote image %s failed: %s", image_url, exc)
        return file_objects
