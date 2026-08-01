from __future__ import annotations

"""Select an upstream by capabilities + model ownership, then random among matches."""

import random
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Sequence, Set

from core.types import CapabilityError, ModelNotAvailableError
from core.registry import UpstreamModule, UpstreamRegistry, get_registry, _PLATFORM_CAPS


def _caps_ok(mod: UpstreamModule, required: Iterable[str]) -> bool:
    for key in required:
        if key in _PLATFORM_CAPS:
            continue
        if not mod.capabilities.get(key, False):
            return False
    return True


def _model_ok(mod: UpstreamModule, model_id: str, models_by_upstream: Dict[str, Set[str]]) -> bool:
    owned = models_by_upstream.get(mod.name) or set()
    if not owned:
        # Unknown inventory: allow (single-upstream bootstrap / cache cold)
        return True
    return model_id in owned


def candidate_upstreams(
    *,
    model_id: str,
    required_capabilities: Sequence[str],
    models_by_upstream: Optional[Dict[str, Set[str]]] = None,
    registry: Optional[UpstreamRegistry] = None,
) -> List[UpstreamModule]:
    reg = registry or get_registry()
    inventory = models_by_upstream or {}
    out: List[UpstreamModule] = []
    for mod in reg.modules.values():
        if not _caps_ok(mod, required_capabilities):
            continue
        if not _model_ok(mod, model_id, inventory):
            continue
        out.append(mod)
    return out


def select_upstream(
    *,
    model_id: str,
    required_capabilities: Sequence[str] = ("chat",),
    models_by_upstream: Optional[Dict[str, Set[str]]] = None,
    registry: Optional[UpstreamRegistry] = None,
) -> UpstreamModule:
    cands = candidate_upstreams(
        model_id=model_id,
        required_capabilities=required_capabilities,
        models_by_upstream=models_by_upstream,
        registry=registry,
    )
    if not cands:
        # Distinguish empty registry vs filter miss
        reg = registry or get_registry()
        if not reg.modules:
            raise CapabilityError("No upstreams registered")
        raise ModelNotAvailableError(
            f"No upstream for model={model_id!r} caps={list(required_capabilities)!r}"
        )
    return random.choice(cands)


def _required_capabilities(
    tools: Optional[List[Dict[str, Any]]],
    messages: Sequence[Dict[str, Any]],
) -> tuple[str, ...]:
    caps: List[str] = ["chat"]
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    caps.append("vision")
                    break
    return tuple(dict.fromkeys(caps))


def resolve_upstream(
    state: Any,
    model: str,
    messages: Sequence[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> tuple[UpstreamModule, Any]:
    caps = _required_capabilities(tools, messages)
    mod = select_upstream(
        model_id=model,
        required_capabilities=caps,
        models_by_upstream=state._models_by_upstream(),
        registry=state._registry,
    )
    client = state.client_for(model, caps, upstream_name=mod.name)
    return mod, client


async def stream_openai_chat(
    state: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    mod, client = resolve_upstream(state, model, messages, tools)
    stream_fn = getattr(mod.module, "stream_openai_chat", None)
    if not callable(stream_fn):
        raise RuntimeError(f"upstream {mod.name} missing stream_openai_chat()")
    async for event in stream_fn(
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
