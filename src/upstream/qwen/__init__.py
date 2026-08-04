from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from upstream.caps import load_capabilities

NAME = "qwen"

_DEFAULT_CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": True,
    "search": True,
    "count_tokens": True,
    "image_gen": True,
    "tts": True,
    "asr": True,
}

CAPABILITIES: Dict[str, bool] = load_capabilities(NAME, _DEFAULT_CAPABILITIES)


def create_client(splitter: Any = None) -> Any:
    from upstream.qwen.client import QwenClient

    return QwenClient(splitter)


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
    from upstream.qwen.completion_stream import stream_openai_chat as _stream

    async for event in _stream(
        state,
        client,
        messages,
        model,
        tools,
        req_id,
        protocol_options=protocol_options,
        prompt_api=prompt_api,
        files=files,
    ):
        yield event
