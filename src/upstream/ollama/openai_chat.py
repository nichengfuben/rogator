from __future__ import annotations

"""Ollama 上游 OpenAI chat 流适配与请求体归一化。"""

import base64
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from server.model.model_thinking import ThinkingRoute

logger = logging.getLogger("rogator")


def _try_extract_base64(url_obj: Any) -> Optional[str]:
    """从 image_url 对象中提取 data: URI 的 base64 数据。"""
    if not isinstance(url_obj, dict):
        return None
    url = url_obj.get("url", "")
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    _, _, b64data = url.partition(",")
    return b64data or None


def _extract_images_from_content(content: Any) -> Tuple[str, List[str]]:
    """从 OpenAI 格式 content 中提取文本和 base64 图片列表。"""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            t = part.get("text")
            if isinstance(t, str):
                text_parts.append(t)
        elif ptype == "image_url":
            b64 = _try_extract_base64(part.get("image_url"))
            if b64:
                images.append(b64)
    return "".join(text_parts), images


def build_chat_payload(
    messages: List[Dict[str, Any]],
    model: str,
    *,
    stream: bool = True,
) -> Dict[str, Any]:
    """将 OpenAI 格式消息转换为 Ollama /api/chat 请求体。"""
    ollama_msgs: list[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text, images = _extract_images_from_content(content)
        entry: Dict[str, Any] = {"role": role, "content": text}
        if images:
            entry["images"] = images
        ollama_msgs.append(entry)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": ollama_msgs,
        "stream": stream,
    }
    return payload


def _prepare_stream(
    state: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    model: str,
    protocol_options: Optional[Dict[str, Any]],
    prompt_api: str,
) -> Tuple[List[Dict[str, Any]], str, ThinkingRoute]:
    injected, full_content, route = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    final_messages, send_text, _filename, _file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=False, model=model,
    )
    if final_messages:
        final_messages[0]["content"] = send_text
    return final_messages, send_text, route


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
    del files  # Ollama 不支持附件上传
    final_messages, _send_text, _route = _prepare_stream(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    # Ollama 不支持 tools/thinking/search，忽略这些参数
    payload = build_chat_payload(final_messages, model, stream=True)
    async for event in client.stream_chat(payload):
        yield event
