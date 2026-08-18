from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from upstream.caps import load_capabilities
from upstream.ollama.routes import DEFAULT_CAPABILITIES

NAME = "ollama"

CAPABILITIES: Dict[str, bool] = load_capabilities(NAME, DEFAULT_CAPABILITIES)


def create_client(splitter: Any = None) -> Any:
    from upstream.ollama.client import OllamaClient

    return OllamaClient(splitter)


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
    from upstream.ollama.openai_chat import stream_openai_chat as _stream

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
