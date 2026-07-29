from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

NAME = "deepseek"

_PLATFORM_SKIP = frozenset({"thinking", "tools", "native_tools"})

_DEFAULT_CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": True,
    "search": False,
    "count_tokens": True,
    "image_gen": False,
    "tts": False,
}


def _load_capability_overrides() -> Dict[str, bool]:
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return {}
    raw = _load_upstream_toml("deepseek")
    caps = raw.get("capabilities") if isinstance(raw, dict) else None
    if not isinstance(caps, dict):
        return {}
    out: Dict[str, bool] = {}
    for key, val in caps.items():
        if key in _PLATFORM_SKIP or key not in _DEFAULT_CAPABILITIES:
            continue
        out[str(key)] = bool(val)
    return out


CAPABILITIES: Dict[str, bool] = {**_DEFAULT_CAPABILITIES, **_load_capability_overrides()}


def create_client(splitter: Any = None) -> Any:
    from upstream.deepseek.client import DeepSeekClient

    return DeepSeekClient(splitter)


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
    from upstream.deepseek.openai_chat import stream_openai_chat as _stream

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
