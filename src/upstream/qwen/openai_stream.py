from __future__ import annotations

"""Qwen 上游 OpenAI 聊天流。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.logger import get_logger

from handlers import extract_system_for_inject
from handlers.fncall_inject import inject_fncall_for_request
from handlers.openai.protocol import _inject_protocol_options
from handlers.openai.thinking import protocol_thinking_level
from handlers.openai.tools import convert_tools_to_openai
from server.formats import TokenExpiredError
from server.model.model_thinking import resolve_qwen_thinking

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
    messages,
    model,
    tools,
    req_id,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> Tuple[Any, List, List, str, bool, str]:
    qwen_enabled, qwen_mode, use_entml = resolve_qwen_thinking(
        model, protocol_thinking_level(protocol_options),
    )
    inject_options = _inject_protocol_options(protocol_options, use_entml)

    session = await client.get_valid_session()
    if not session:
        raise TokenExpiredError("No valid session available")

    image_uris = client.extract_base64_images(messages)
    media_urls = client.extract_remote_media_urls(messages)
    user_system_prompt, messages = extract_system_for_inject(messages)
    openai_tools = convert_tools_to_openai(tools)
    injected = inject_fncall_for_request(
        messages,
        openai_tools,
        state.protocol,
        req_id=req_id,
        api=prompt_api,
        model=model,
        lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=inject_options,
    )
    full_content = injected[0]["content"]
    final_messages = injected
    send_text, filename, file_bytes = state.splitter.split(full_content)
    files = await _collect_uploaded_files(
        client, session, messages, image_uris, media_urls, filename, file_bytes,
    )
    final_messages[0]["content"] = send_text
    chat_id = await client.create_chat(session, model)
    return session, final_messages, files, chat_id, qwen_enabled, qwen_mode


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
    prep = await _prepare_stream(
        state, client, messages, model, tools, req_id, protocol_options, prompt_api=prompt_api,
    )
    session, final_messages, uploaded_files, chat_id, qwen_enabled, qwen_mode = prep
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
