from __future__ import annotations

"""QwenClient 的文件与多模态媒体下载能力 + message 文件提取。

上传管线（STS/OSS/PUT/解析/时间线埋点）见 ``files_upload.py``；
``UploadMixin`` 通过继承 ``MediaDownloadMixin`` 同时具备两类能力。
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from upstream.qwen.auth.http import get_qwen_proxy
from upstream.qwen.chat.endpoints import (
    BASE_URL,
    GENERATED_IMAGE_DIR,
    GENERATED_VIDEO_DIR,
    USER_AGENT,
)
from upstream.qwen.chat.upload.storage import (
    DATA_URI_EXT_MAP,
    save_image_file,
    save_video_file,
)

logger = logging.getLogger("rogator")


class MediaDownloadMixin:
    """图片 / 视频下载 + 多模态 message 文件提取。"""

    @staticmethod
    def extract_base64_images(messages: List[Dict[str, Any]]) -> List[str]:
        """从 ``messages`` 中收集 ``data:`` 开头的 base64 图片 URI。"""

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

    @staticmethod
    def extract_remote_media_urls(messages: List[Dict[str, Any]]) -> List[str]:
        """从 ``messages`` 中收集非 base64 的远程媒体 URL。"""

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
                proxy=get_qwen_proxy(),
            ) as resp:
                if resp.status != 200:
                    return None
                return save_image_file(
                    await resp.read(),
                    resp.headers.get("Content-Type", "image/png"),
                    save_dir,
                )

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
                    proxy=get_qwen_proxy(),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Download video failed: HTTP %d", resp.status)
                        return None
                    return save_video_file(await resp.read(), save_dir)
        except Exception as exc:
            logger.warning("Download video exception: %s", exc)
            return None

    @staticmethod
    def ext_from_content_type(content_type: str) -> str:
        """从 Content-Type 推断文件扩展名；``image/*`` 用子类兜底，未知用 ``.bin``。"""
        ext = DATA_URI_EXT_MAP.get(content_type)
        if not ext and content_type.startswith("image/"):
            ext = f".{content_type.split('/')[-1]}"
        return ext or ".bin"

    async def prepare_message_files(
        self,
        session: Any,
        messages: List[Dict[str, Any]],
        extra_files: Optional[List[Tuple[bytes, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """按 ``extra_files → base64 → remote`` 顺序串行上传；单个失败不影响其余。

        上传函数由 ``UploadMixin`` 提供，本类只负责"哪几个文件要上传"的提取。
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
        for media_url in self.extract_remote_media_urls(messages):
            try:
                _, file_obj = await self.upload_file_from_url(session, media_url)
                file_objects.append(file_obj)
            except Exception as exc:
                logger.warning("Upload remote media %s failed: %s", media_url, exc)
        return file_objects