from __future__ import annotations

"""流式请求的文件上传与消息准备辅助函数。"""

import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from echotools.base.logger import get_logger

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from server.model.model_thinking import ThinkingRoute

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = get_logger("rogator")

_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _extract_page_urls(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    urls: List[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(".,;:)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


async def _upload_base64_images(
    client: "QwenClient", session: "QwenSession", image_uris: List[str],
) -> List[Any]:
    files: List[Any] = []
    for uri in image_uris:
        try:
            _, image_obj = await client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    return files


async def _upload_remote_media(
    client: "QwenClient", session: "QwenSession", media_urls: List[str],
) -> List[Any]:
    files: List[Any] = []
    for media_url in media_urls:
        try:
            _, media_obj = await client.upload_file_from_url(session, media_url)
            files.append(media_obj)
        except Exception as e:
            logger.debug("Remote media upload failed: %s", e)
    return files


async def _upload_text_attachment(
    client: "QwenClient",
    session: "QwenSession",
    filename: Optional[str],
    file_bytes: Optional[bytes],
) -> List[Any]:
    if not filename or not file_bytes:
        return []
    try:
        _, file_obj = await client.upload_file(session, file_bytes, filename)
        return [file_obj]
    except Exception as e:
        logger.warning("Upload failed: %s, sending truncated text without attachment", e)
        return []


async def _collect_uploaded_files(
    client: "QwenClient",
    session: "QwenSession",
    messages: List[Any],
    image_uris: List[str],
    media_urls: List[str],
    filename: Optional[str],
    file_bytes: Optional[bytes],
    send_text: str,
) -> List[Any]:
    from upstream.qwen.chat.upload.upstream_api import parse_urls

    files: List[Any] = []
    files.extend(await _upload_base64_images(client, session, image_uris))
    files.extend(await _upload_remote_media(client, session, media_urls))
    files.extend(await _upload_text_attachment(client, session, filename, file_bytes))
    page_urls = _extract_page_urls(send_text)
    if page_urls:
        try:
            files.extend(await parse_urls(client, session, page_urls))
        except Exception as exc:
            logger.debug("parse_urls failed: %s", exc)
    return files


async def prepare_stream(
    state: Any,
    client: "QwenClient",
    session: "QwenSession",
    messages: List[Any],
    model: str,
    tools: Optional[List[Any]],
    req_id: str,
    protocol_options: Optional[Any] = None,
    *,
    prompt_api: str = "openai",
) -> Tuple[List[Any], List[Any], ThinkingRoute]:
    injected, full_content, route = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    image_uris = client.extract_base64_images(messages)
    media_urls = client.extract_remote_media_urls(messages)
    final_messages, send_text, filename, file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=True, model=model,
    )
    files = await _collect_uploaded_files(
        client, session, messages, image_uris, media_urls, filename, file_bytes, send_text,
    )
    final_messages[0]["content"] = send_text
    return final_messages, files, route

