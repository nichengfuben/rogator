from __future__ import annotations

"""DeepSeek 附件收集：base64 图片、远程 URL、splitter 文本。"""

import base64
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from upstream.deepseek.lib.protocol.headers import build_headers

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(
    r"^data:([^;,]+)?;base64,(.+)$",
    re.DOTALL,
)
_MIME_EXT_MAP: Dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
}


def _decode_data_uri(uri: str) -> Tuple[bytes, str]:
    match = _DATA_URI_RE.match(uri.strip())
    if not match:
        raise ValueError("invalid data URI")
    mime = (match.group(1) or "application/octet-stream").strip().lower()
    encoded = match.group(2)
    padding = (-len(encoded)) % 4
    if padding:
        encoded += "=" * padding
    ext = _MIME_EXT_MAP.get(mime, ".bin")
    filename = "upload_{}{}".format(uuid.uuid4().hex[:8], ext)
    return base64.b64decode(encoded), filename


def extract_base64_images(messages: Sequence[Dict[str, Any]]) -> List[Tuple[bytes, str]]:
    """从 OpenAI 风格消息中提取内联 base64 图片。"""
    out: List[Tuple[bytes, str]] = []
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
            if not candidate.startswith("data:"):
                continue
            try:
                out.append(_decode_data_uri(candidate))
            except Exception as exc:
                logger.warning("跳过无效 data URI 图片: %s", exc)
    return out


def extract_remote_media_urls(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """提取需先下载再上传的远程媒体 URL。"""
    seen: set = set()
    urls: List[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image_url":
                url_obj = part.get("image_url")
                candidate = (
                    str(url_obj.get("url", ""))
                    if isinstance(url_obj, dict)
                    else str(url_obj or "")
                )
            elif part_type in ("input_audio", "video_url"):
                nested = part.get(part_type) or {}
                candidate = str(nested.get("url", "")) if isinstance(nested, dict) else ""
            else:
                continue
            if candidate.startswith(("http://", "https://")) and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    return urls


async def _download_remote(url: str) -> Tuple[bytes, str]:
    async with aiohttp.ClientSession() as tmp:
        async with tmp.get(
            url,
            headers={"User-Agent": build_headers("")["user-agent"]},
            ssl=False,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError("download failed HTTP {}".format(resp.status))
            data = await resp.read()
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            content_type = content_type.split(";", 1)[0].strip().lower()
            ext = _MIME_EXT_MAP.get(content_type, ".bin")
            filename = "upload_{}{}".format(uuid.uuid4().hex[:8], ext)
            return data, filename


def collect_message_attachments(
    messages: Sequence[Dict[str, Any]],
    *,
    filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
) -> List[Tuple[bytes, str]]:
    """汇总消息内联图片与 splitter 文本附件。"""
    items: List[Tuple[bytes, str]] = []
    items.extend(extract_base64_images(messages))
    if filename and file_bytes:
        items.append((file_bytes, filename))
    return items


async def collect_message_attachments_async(
    messages: Sequence[Dict[str, Any]],
    *,
    filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
) -> List[Tuple[bytes, str]]:
    """异步版：额外下载远程 URL 媒体。"""
    items = collect_message_attachments(
        messages, filename=filename, file_bytes=file_bytes,
    )
    for url in extract_remote_media_urls(messages):
        try:
            items.append(await _download_remote(url))
        except Exception as exc:
            logger.warning("远程媒体下载失败 %s: %s", url[:48], exc)
    return items
