from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_EFFORT_SUFFIXES: Tuple[str, ...] = (
    "extra-high",
    "xhigh",
    "medium",
    "minimal",
    "high",
    "low",
    "max",
    "none",
    "auto",
)
_INVALID_ID_CHARS = frozenset("{}\'\"")
_THINKING_TOKEN = "thinking"


@dataclass(frozen=True)
class CursorModelIdentity:
    model_id: str
    base_name: str
    effort: Optional[str]
    thinking: bool
    fast: bool
    display_name: str


def is_valid_model_id(model_id: str) -> bool:
    text = str(model_id or "").strip()
    if not text or len(text) > 128:
        return False
    return not any(ch in text for ch in _INVALID_ID_CHARS)


def external_id_for(model_id: str) -> str:
    return str(model_id or "").strip().replace(".", "-")


def parse_cursor_model_id(model_id: str) -> Optional[CursorModelIdentity]:
    text = str(model_id or "").strip()
    if not is_valid_model_id(text):
        return None

    fast = False
    thinking = False
    effort: Optional[str] = None
    parts = text.split("-")

    if parts and parts[-1] == "fast":
        fast = True
        parts = parts[:-1]

    if _THINKING_TOKEN in parts:
        thinking = True
        parts = [p for p in parts if p != _THINKING_TOKEN]

    if parts:
        tail = parts[-1]
        if tail in _EFFORT_SUFFIXES:
            effort = tail
            parts = parts[:-1]
        elif len(parts) >= 2:
            pair = f"{parts[-2]}-{parts[-1]}"
            if pair in _EFFORT_SUFFIXES:
                effort = pair
                parts = parts[:-2]

    base_name = "-".join(parts) or text
    label_parts: List[str] = [base_name]
    if thinking:
        label_parts.append("thinking")
    if effort:
        label_parts.append(effort)
    if fast:
        label_parts.append("fast")
    display_name = " / ".join(label_parts)
    return CursorModelIdentity(
        model_id=text,
        base_name=base_name,
        effort=effort,
        thinking=thinking,
        fast=fast,
        display_name=display_name,
    )


def think_efforts_for(identity: CursorModelIdentity) -> Dict[str, Any]:
    if identity.thinking:
        return {
            "support": True,
            "valid_efforts": ["on"],
            "default_effort": "on",
        }
    if identity.effort:
        return {
            "support": True,
            "valid_efforts": [identity.effort],
            "default_effort": identity.effort,
            "off_effort": "none",
        }
    return {
        "support": True,
        "valid_efforts": ["low", "medium", "high", "xhigh", "max", "auto"],
        "default_effort": "medium",
        "off_effort": "none",
    }


def meta_for_model(model_id: str, api_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    identity = parse_cursor_model_id(model_id)
    if identity is None:
        return {
            "id": model_id,
            "object": "model",
            "owned_by": "cursor",
            "capabilities": {"chat": True, "thinking": True, "tools": True},
        }

    display = identity.display_name
    if isinstance(api_item, dict):
        for key in ("displayName", "displayNameShort", "name"):
            val = api_item.get(key)
            if isinstance(val, str) and val.strip():
                display = val.strip()
                break

    meta: Dict[str, Any] = {
        "id": identity.model_id,
        "object": "model",
        "owned_by": "cursor",
        "display_name": display,
        "cursor_base": identity.base_name,
        "capabilities": {
            "chat": True,
            "thinking": True,
            "tools": True,
        },
        "think_efforts": think_efforts_for(identity),
    }
    if identity.effort:
        meta["cursor_effort"] = identity.effort
    if identity.thinking:
        meta["cursor_thinking"] = True
    if identity.fast:
        meta["cursor_fast"] = True
    return meta


def normalize_model_id(raw: Any) -> Optional[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return text if is_valid_model_id(text) else None
    if isinstance(raw, dict):
        for key in ("modelId", "model_id", "id", "name"):
            val = raw.get(key)
            if isinstance(val, str):
                text = val.strip()
                if is_valid_model_id(text):
                    return text
    return None


def normalize_api_models(
    api_models: List[Any],
    *,
    extra_ids: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ids: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()

    def _add(raw: Any, api_item: Optional[Dict[str, Any]] = None) -> None:
        model_id = normalize_model_id(raw if api_item is None else api_item)
        if not model_id or model_id in seen:
            return
        seen.add(model_id)
        ids.append(model_id)
        meta[model_id] = meta_for_model(model_id, api_item if isinstance(api_item, dict) else None)

    for item in api_models:
        if isinstance(item, dict):
            _add(item, item)
        else:
            _add(item)

    for raw in extra_ids or []:
        _add(raw)

    return ids, meta
