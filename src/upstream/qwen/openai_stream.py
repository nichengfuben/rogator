from __future__ import annotations

"""Qwen 上游 OpenAI 聊天流。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.base.logger import get_logger

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from server.formats import TokenExpiredError

logger = get_logger("rogator")


async def _upload_base64_images(client, session, image_uris: List[str]) -> List[Any]:
    files: List[Any] = []
    for uri in image_uris:
        try:
            _, image_obj = await client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    return files


async def _upload_remote_media(client, session, media_urls: List[str]) -> List[Any]:
    files: List[Any] = []
    for media_url in media_urls:
        try:
            _, media_obj = await client.upload_file_from_url(session, media_url)
            files.append(media_obj)
        except Exception as e:
            logger.debug("Remote media upload failed: %s", e)
    return files


async def _upload_text_attachment(
    client, session, filename: Optional[str], file_bytes: Optional[bytes],
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
    client, session, messages, image_uris, media_urls, filename, file_bytes,
) -> List[Any]:
    files: List[Any] = []
    files.extend(await _upload_base64_images(client, session, image_uris))
    files.extend(await _upload_remote_media(client, session, media_urls))
    files.extend(await _upload_text_attachment(client, session, filename, file_bytes))
    return files


async def _prepare_stream(
    state,
    client,
    session,
    messages,
    model,
    tools,
    req_id,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> Tuple[List, List, str, bool, str]:
    injected, full_content, qwen_enabled, qwen_mode, _use_entml = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )

    image_uris = client.extract_base64_images(messages)
    media_urls = client.extract_remote_media_urls(messages)
    final_messages, send_text, filename, file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=True,
    )
    files = await _collect_uploaded_files(
        client, session, messages, image_uris, media_urls, filename, file_bytes,
    )
    final_messages[0]["content"] = send_text
    chat_id = await client.create_chat(session, model)
    return final_messages, files, chat_id, qwen_enabled, qwen_mode


async def stream_openai_chat(
    state: Any,
    client: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    async with client.lease_valid_session() as session:
        if not session:
            raise TokenExpiredError("No valid session available")
        final_messages, uploaded_files, chat_id, qwen_enabled, qwen_mode = await _prepare_stream(
            state, client, session, messages, model, tools, req_id, protocol_options,
            prompt_api=prompt_api,
        )
        if files is not None:
            uploaded_files = files
        send_text = final_messages[0].get("content") or ""
        yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
        async for event in client.chat_completion(
            session, chat_id, final_messages, model, uploaded_files,
            qwen_thinking_enabled=qwen_enabled,
            qwen_thinking_mode=qwen_mode,
        ):
            yield event
