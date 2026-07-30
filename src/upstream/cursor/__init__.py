from __future__ import annotations

import sys
import types
from typing import Any, AsyncGenerator, Dict, List, Optional

NAME = "cursor"
PERSIST_MODULE = "upstream.cursor.setup.persist"

_DEFAULT_CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": False,
    "search": False,
    "count_tokens": False,
    "image_gen": False,
    "tts": False,
}

_PLATFORM_SKIP = frozenset({"thinking", "tools", "native_tools"})


def _load_capability_overrides() -> Dict[str, bool]:
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return {}
    raw = _load_upstream_toml("cursor")
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
    from upstream.cursor.client import CursorClient

    return CursorClient(splitter)


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
    from upstream.cursor.chat.openai import stream_openai_chat as _stream

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


def _register_openai_chat_compat() -> None:
    from upstream.cursor.chat.openai import stream_openai_chat as _stream_fn

    mod = types.ModuleType("upstream.cursor.openai_chat")
    mod.stream_openai_chat = _stream_fn  # type: ignore[attr-defined]
    sys.modules.setdefault("upstream.cursor.openai_chat", mod)


_register_openai_chat_compat()
