from __future__ import annotations

"""SSE parsers and stream handler.

Merged from: sse.py, stream.py
"""

import json
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# SSE Parsers (from sse.py)
# ---------------------------------------------------------------------------


def _safe_loads(data_str: str) -> Optional[Any]:
    if not data_str or data_str == "[DONE]":
        return None
    try:
        return json.loads(data_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _parse_head_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "error" in data:
        return {"type": "error", "message": str(data["error"])}
    created = data.get("response.created")
    if isinstance(created, dict):
        return {
            "type": "response_created",
            "response_id": created.get("response_id", ""),
            "response_index": created.get("response_index"),
        }
    stopped = data.get("response.stopped")
    if isinstance(stopped, dict):
        return {
            "type": "response_stopped",
            "response_id": stopped.get("response_id", ""),
        }
    info = data.get("response.info")
    if isinstance(info, dict) and info.get("response_id"):
        return {
            "type": "response_info",
            "response_id": info.get("response_id", ""),
        }
    return None


def _build_answer(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if content and delta.get("status") != "finished":
        return {"type": "answer", "content": content}
    return None


def _build_thinking(delta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "thinking_summary",
        "status": delta.get("status") or "",
        "extra": delta.get("extra", {}),
    }


def _build_image_tool(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if delta.get("role") != "function" or delta.get("status") != "finished":
        return None
    extra = delta.get("extra", {})
    imgs = extra.get("image_list", extra.get("tool_result", []))
    urls = [
        img.get("image", "")
        for img in imgs
        if isinstance(img, dict) and img.get("image")
    ]
    if not urls:
        return None
    return {"type": "image_gen_tool", "urls": urls}


def _build_image_gen(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if not content:
        return None
    return {"type": "image_gen", "content": content, "extra": delta.get("extra", {})}


def _build_video_gen(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if not content:
        return None
    return {"type": "video_gen", "content": content}


def _build_other(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    phase = delta.get("phase")
    content = delta.get("content")
    status = delta.get("status")
    if phase is not None and phase != "" and content and status != "finished":
        return {"type": "other", "content": content}
    return None


def _build_web_search(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    content = delta.get("content")
    if content and delta.get("status") != "finished":
        return {"type": "thinking", "content": str(content)}
    extra = delta.get("extra") or {}
    info = extra.get("web_search_info") or []
    if isinstance(info, list) and info:
        parts = []
        for item in info:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("url") or "")
                snippet = str(item.get("snippet") or item.get("content") or "")
                parts.append(f"{title}: {snippet}".strip(": "))
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p)
        if text:
            return {"type": "thinking", "content": text}
    return None


def _dispatch_phase(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    phase = delta.get("phase")
    if phase == "KeepAlive":
        return None
    if phase == "answer":
        return _build_answer(delta)
    if phase == "think":
        content = delta.get("content")
        return {"type": "thinking", "content": content} if content else None
    if phase == "thinking_summary":
        return _build_thinking(delta)
    if phase == "web_search":
        return _build_web_search(delta)
    if phase == "image_gen_tool":
        return _build_image_tool(delta)
    if phase == "image_gen":
        return _build_image_gen(delta)
    if phase == "video_gen":
        return _build_video_gen(delta)
    return _build_other(delta)


def _parse_choice_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = data.get("usage")
    choices = data.get("choices", [])
    if not choices:
        return {"type": "usage", "data": usage} if usage else None
    delta = choices[0].get("delta", {})
    result = _dispatch_phase(delta)
    if usage:
        if result is None:
            return {"type": "usage", "data": usage}
        result["usage"] = usage
    return result


def parse_sse_event(data_str: str) -> Optional[Dict[str, Any]]:

    data = _safe_loads(data_str)
    if data is None:
        return None
    head = _parse_head_event(data)
    if head is not None:
        return head
    return _parse_choice_event(data)


class SseEventAssembler:
    # 对齐前端 kT/_T（eventsource-parser）：空行 dispatch，多条 data: 用 \n 合并。

    __slots__ = ("_data_parts",)

    def __init__(self) -> None:
        self._data_parts: List[str] = []

    def feed_line(self, line: str) -> Optional[str]:
        if line == "":
            return self._flush()
        if line.startswith(":"):
            return None
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data_parts.append(value)
            return None
        return None

    def flush_eof(self) -> Optional[str]:
        return self._flush()

    def _flush(self) -> Optional[str]:
        if not self._data_parts:
            return None
        payload = "\n".join(self._data_parts)
        self._data_parts.clear()
        return payload


def parse_sse_line(data_str: str) -> Optional[Union[str, Dict[str, Any]]]:
    """Map a raw SSE line into the public stream protocol."""
    event = parse_sse_event(data_str)
    if event is None:
        return None
    if event["type"] == "answer":
        return event.get("content", "")
    if event["type"] == "thinking":
        return {"thinking": event.get("content", "")}
    if event["type"] == "thinking_summary":
        extra = event.get("extra", {})
        titles: List[str] = extra.get("summary_title", {}).get("content", [])
        thoughts: List[str] = extra.get("summary_thought", {}).get("content", [])
        if not titles and not thoughts:
            return None
        parts: List[str] = []
        for index in range(max(len(titles), len(thoughts))):
            title = titles[index] if index < len(titles) else ""
            thought = thoughts[index] if index < len(thoughts) else ""
            if title or thought:
                parts.append(f"{title}: {thought}" if title else thought)
        return {"thinking": "\n".join(parts)} if parts else None
    if event["type"] == "usage":
        return {"usage": event.get("data", {})}
    return None
