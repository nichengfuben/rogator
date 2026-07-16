from __future__ import annotations

"""Video generation and CDN helpers.

Merged from: video.py, cdn.py, media.py (video parts)
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp

from ..transport.endpoints import (
    BASE_URL,
    CHAT_PATH,
    GENERATED_VIDEO_DIR,
    SSE_TIMEOUT,
    TASK_STATUS_PATH,
    USER_AGENT,
    VIDEO_CDN_BASE,
    VIDEO_TASK_MAX_POLL_TIME,
    VIDEO_TASK_POLL_INTERVAL,
)
from ..crypto.crypto import build_headers
from ..compat.payloads import build_i2v_payload
from ..storage.storage import save_video_file

logger = logging.getLogger(__name__)


def build_cdn_video_url(
    user_id: str,
    video_type: str,
    message_id: str,
    task_id: str,
    token: str,
) -> str:
    """Build the fallback CDN URL for a generated video."""
    return (
        f"{VIDEO_CDN_BASE}/{user_id}/{video_type}/{message_id}/{task_id}.mp4?key={token}"
    )


class VideoService:
    """Submit, poll, and optionally download image-to-video jobs."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        proxy_resolver: Callable[[], Optional[str]],
        cookies_provider: Callable[[], dict],
        create_chat: Callable[[str, str, str], Awaitable[str]],
        cleanup_chat: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self._session = session
        self._resolve_proxy = proxy_resolver
        self._cookies = cookies_provider
        self._create_chat = create_chat
        self._cleanup_chat = cleanup_chat

    async def generate(
        self,
        prompt: str,
        image_url: str,
        token: str,
        user_id: str,
        model: str = "qwen-max-latest",
        size: str = "16:9",
        image_name: str = "source.png",
        download: bool = True,
    ) -> Dict[str, Any]:
        """Run the full image-to-video flow."""
        try:
            chat_id = await self._create_chat(token, model, "i2v")
        except Exception as exc:
            return {"success": False, "error": f"create i2v chat failed: {exc}"}
        try:
            submission = await self._submit_task(prompt, chat_id, model, image_url, image_name, size, token)
            if not submission.get("success"):
                return submission
            task_result = await self._poll_task_status(submission["task_id"], token, chat_id)
            video_url = task_result.get("content") or build_cdn_video_url(
                user_id=user_id,
                video_type="i2v",
                message_id=submission["message_id"],
                task_id=submission["task_id"],
                token=token,
            )
            result: Dict[str, Any] = {
                "success": True,
                "task_id": submission["task_id"],
                "message_id": submission["message_id"],
                "chat_id": chat_id,
                "video_url": video_url,
                "size": size,
            }
            if download and video_url:
                local_path = await self._download_video(video_url)
                if local_path:
                    result["local_path"] = local_path
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        finally:
            asyncio.ensure_future(self._cleanup_chat(chat_id, token))

    async def _submit_task(
        self,
        prompt: str,
        chat_id: str,
        model: str,
        image_url: str,
        image_name: str,
        size: str,
        token: str,
    ) -> Dict[str, Any]:
        async with self._session.post(
            f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}",
            json=build_i2v_payload(prompt, chat_id, model, image_url, image_name, size),
            headers=build_headers(token, chat_id=chat_id, cookies=self._cookies()),
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=SSE_TIMEOUT),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                return {"success": False, "error": f"HTTP {response.status}: {(await response.text())[:300]}"}
            data = await response.json()
            if not data.get("success"):
                return {"success": False, "error": str(data)}
            payload = data.get("data", {})
            message_id = payload.get("message_id", "")
            messages = payload.get("messages", [])
            task_id = ""
            if messages:
                task_id = ((messages[0].get("extra") or {}).get("wanx") or {}).get("task_id", "")
            if not task_id:
                return {"success": False, "error": "missing task_id in image-to-video response"}
            return {"success": True, "task_id": task_id, "message_id": message_id}

    async def _poll_task_status(self, task_id: str, token: str, chat_id: str) -> Dict[str, Any]:
        start = time.time()
        url = f"{BASE_URL}{TASK_STATUS_PATH.format(task_id=task_id)}"
        headers = build_headers(token, chat_id=chat_id, cookies=self._cookies())
        while time.time() - start < VIDEO_TASK_MAX_POLL_TIME:
            async with self._session.get(
                url,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
                proxy=self._resolve_proxy(),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get("task_status", "")
                    if status == "succeeded":
                        return data
                    if status == "failed":
                        raise RuntimeError(f"video task failed: {data.get('message', 'unknown')}" )
            await asyncio.sleep(VIDEO_TASK_POLL_INTERVAL)
        raise RuntimeError(f"video task polling timed out after {VIDEO_TASK_MAX_POLL_TIME} seconds")

    async def _download_video(self, video_url: str) -> Optional[str]:
        async with self._session.get(
            video_url,
            headers={
                "Accept": "*/*",
                "Origin": BASE_URL,
                "Referer": f"{BASE_URL}/",
                "User-Agent": USER_AGENT,
            },
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=SSE_TIMEOUT),
            proxy=self._resolve_proxy(),
        ) as response:
            if response.status != 200:
                return None
            return save_video_file(await response.read())


class VideoGenMixin:
    """Mixin providing video generation helpers."""

    async def _poll_once(self, url: str, headers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            async with self._session.get(
                url,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
                proxy=self._get_proxy_kwarg(),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                task_status = data.get("task_status", "")
                if task_status == "succeeded":
                    return data
                if task_status == "failed":
                    raise RuntimeError(f"任务失败: {data.get('message', '未知')}")
                return None
        except Exception as exc:
            if "任务失败" in str(exc):
                raise
            logger.debug("轮询异常: %s", exc)
            return None

    async def _poll_task_status(
        self,
        task_id: str,
        token: str,
        chat_id: str,
    ) -> Dict[str, Any]:
        """Poll an async media task until completion."""
        url = f"{BASE_URL}{TASK_STATUS_PATH.format(task_id=task_id)}"
        headers = build_headers(
            token,
            chat_id=chat_id,
            include_sse=False,
            cookies=self._cookies,
        )
        start = time.time()
        while time.time() - start < VIDEO_TASK_MAX_POLL_TIME:
            data = await self._poll_once(url, headers)
            if data is not None:
                return data
            await asyncio.sleep(VIDEO_TASK_POLL_INTERVAL)
        raise RuntimeError(f"任务轮询超时 ({VIDEO_TASK_MAX_POLL_TIME}s)")

    async def _submit_i2v_task(
        self,
        prompt: str,
        chat_id: str,
        model: str,
        image_url: str,
        image_name: str,
        size: str,
        token: str,
    ) -> tuple:
        """Submit an image-to-video task. Returns (task_id, message_id)."""
        payload = build_i2v_payload(
            prompt=prompt, chat_id=chat_id, model=model,
            image_url=image_url, image_name=image_name, size=size,
        )
        headers = build_headers(token, chat_id=chat_id, cookies=self._cookies)
        url = f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}"
        async with self._session.post(
            url, json=payload, headers=headers, ssl=False,
            timeout=aiohttp.ClientTimeout(total=SSE_TIMEOUT),
            proxy=self._get_proxy_kwarg(),
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                return False, f"HTTP {resp.status}: {err[:300]}"
            data = await resp.json()
            if not data.get("success"):
                return False, str(data)
            result_data = data.get("data", {})
            message_id = result_data.get("message_id", "")
            task_id = ""
            messages = result_data.get("messages", [])
            if messages:
                task_id = ((messages[0].get("extra") or {}).get("wanx") or {}).get("task_id", "")
            if not task_id:
                return False, "响应中未找到 task_id"
        return True, task_id, message_id

    async def _download_video(self, video_url: str) -> Optional[str]:
        """Download video from URL. Returns local path or None."""
        try:
            async with self._session.get(
                video_url,
                headers={
                    "Accept": "*/*", "Origin": BASE_URL,
                    "Referer": f"{BASE_URL}/", "User-Agent": USER_AGENT,
                },
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=SSE_TIMEOUT),
                proxy=self._get_proxy_kwarg(),
            ) as resp:
                if resp.status == 200:
                    return save_video_file(await resp.read(), GENERATED_VIDEO_DIR)
        except Exception as exc:
            logger.debug("视频下载失败: %s", exc)
        return None

    async def generate_video(
        self,
        prompt: str,
        image_url: str,
        token: str,
        user_id: str,
        model: str = "qwen-max-latest",
        size: str = "16:9",
        image_name: str = "source.png",
        download: bool = True,
    ) -> Dict[str, Any]:
        """Run the full image-to-video flow."""
        try:
            chat_id = await self._create_chat(token, model, "i2v")
        except Exception as exc:
            return {"success": False, "error": f"创建 i2v 对话失败: {exc}"}
        try:
            ok, *rest = await self._submit_i2v_task(
                prompt, chat_id, model, image_url, image_name, size, token,
            )
            if not ok:
                asyncio.ensure_future(self._cleanup_chat(chat_id, token))
                return {"success": False, "error": rest[0]}
            task_id, message_id = rest
            task_result = await self._poll_task_status(task_id, token, chat_id)
        except Exception as exc:
            asyncio.ensure_future(self._cleanup_chat(chat_id, token))
            return {"success": False, "error": str(exc)}
        video_url = task_result.get("content") or build_cdn_video_url(
            user_id=user_id, video_type="i2v",
            message_id=message_id, task_id=task_id, token=token,
        )
        result: Dict[str, Any] = {
            "success": True, "task_id": task_id, "message_id": message_id,
            "chat_id": chat_id, "video_url": video_url, "size": size,
        }
        if download and video_url:
            local_path = await self._download_video(video_url)
            if local_path:
                result["local_path"] = local_path
        asyncio.ensure_future(self._cleanup_chat(chat_id, token))
        return result
