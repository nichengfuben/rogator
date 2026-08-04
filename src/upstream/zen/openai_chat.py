from __future__ import annotations

"""Zen 上游 OpenAI chat 流适配。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from server.model.model_thinking import ThinkingRoute
from upstream.zen.payload import build_chat_payload, normalize_model_name


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
    del files  # Zen 无附件上传；超长由 gateway inject/截断处理
    model = normalize_model_name(model)
    final_messages, send_text, route = _prepare_stream(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    payload = build_chat_payload(
        final_messages,
        model,
        stream=True,
        thinking=bool(route.qwen_native_enabled),
    )
    async for event in client.stream_chat(payload):
        yield event
