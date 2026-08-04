from __future__ import annotations

"""Zen 请求体与 tools 归一化。"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rogator")


def normalize_model_name(model: str) -> str:
    if not model:
        return model
    if "-local" in model:
        base = model.replace("-local", "")
        logger.debug("Zen model normalized: %s -> %s", model, base)
        return base
    return model


def _normalize_tool_entry(t: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(t, dict):
        return None
    func = t.get("function")
    if isinstance(func, dict) and func.get("name"):
        return {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", "") or "",
                "parameters": func.get("parameters") or func.get("input_schema") or {},
            },
        }
    name = t.get("name")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": t.get("description", "") or "",
            "parameters": t.get("input_schema") or t.get("parameters") or {},
        },
    }


def normalize_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not tools or not isinstance(tools, list):
        return None
    normalized = [nt for nt in (_normalize_tool_entry(t) for t in tools) if nt]
    return normalized or None


def build_chat_payload(
    messages: List[Dict[str, Any]],
    model: str,
    *,
    stream: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Any = None,
    top_p: Any = None,
    max_tokens: Any = None,
    stop: Any = None,
    tool_choice: Any = None,
    thinking: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop:
        payload["stop"] = stop
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if thinking:
        payload["thinking"] = True
    return payload


def build_headers(*, stream: bool) -> Dict[str, str]:
    from upstream.zen.routes import USER_AGENT

    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": USER_AGENT,
    }
