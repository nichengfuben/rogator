from __future__ import annotations

"""SSE parsers and stream handler.

Merged from: sse.py, stream.py
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union

import aiohttp


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
        return {"type": "response_created", "response_id": created.get("response_id", "")}
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


def _dispatch_phase(delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    phase = delta.get("phase")
    if phase == "answer":
        return _build_answer(delta)
    if phase == "think":
        content = delta.get("content")
        return {"type": "thinking", "content": content} if content else None
    if phase == "thinking_summary":
        return _build_thinking(delta)
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
    """Parse one SSE ``data`` line into a structured event."""
    data = _safe_loads(data_str)
    if data is None:
        return None
    head = _parse_head_event(data)
    if head is not None:
        return head
    return _parse_choice_event(data)


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


# ---------------------------------------------------------------------------
# Stream Handler (from stream.py)
# ---------------------------------------------------------------------------


class StreamHandler:
    """Consume one SSE response and emit normalized stream items."""

    def __init__(self, download_image: Callable[[str], Awaitable[Optional[str]]]) -> None:
        self._download_image = download_image
        self.last_response_id: Optional[str] = None
        self._thinking_count = 0
        self._tail = b""

    async def stream(
        self,
        resp: aiohttp.ClientResponse,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Yield normalized items from a Qwen SSE response."""
        self.last_response_id = None
        self._thinking_count = 0
        self._tail = b""
        buffer = await resp.content.readany()
        if buffer:
            async for item in self._process_buffer(buffer):
                yield item
            buffer = self._tail
        async for raw in resp.content.iter_any():
            if not raw:
                continue
            buffer += raw
            async for item in self._process_buffer(buffer):
                yield item
            buffer = self._tail

    async def _process_buffer(
        self,
        buffer: bytes,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        lines = buffer.split(b"\n")
        self._tail = lines[-1]
        for line_bytes in lines[:-1]:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            event = parse_sse_event(line[5:].lstrip())
            if event is None:
                continue
            async for item in self._dispatch(event):
                yield item

    def _handle_error(self, event: Dict[str, Any]) -> None:
        raise RuntimeError(f"Qwen server error: {event.get('message', '')}")

    def _handle_answer(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = self._strip_tags(event.get("content", ""))
        return text if text else None

    def _handle_thinking(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = self._strip_tags(event.get("content", ""))
        return {"thinking": text} if text else None

    def _handle_thinking_summary(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"thinking": p} for p in self._iter_thinking_pieces(event)]

    async def _handle_image_gen_tool(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        calls = await asyncio.gather(
            *[self._build_single_image_call(url) for url in event.get("urls", [])]
        )
        return {"tool_calls": calls} if calls else None

    async def _handle_image_gen(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = event.get("content", "")
        if content:
            return {"tool_calls": [await self._build_single_image_call(content)]}
        return None

    def _handle_video_gen(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = event.get("content", "")
        if content:
            return {"tool_calls": [self._wrap_tool_call("qwen.video_gen", {"url": content})]}
        return None

    def _handle_usage(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {"usage": event.get("data", {})}

    def _handle_other(self, event: Dict[str, Any]) -> Optional[str]:
        content = self._strip_tags(event.get("content", ""))
        return content if content else None

    _SIMPLE_HANDLERS = {
        "response_created": "_handle_response_created",
        "answer": "_handle_answer",
        "thinking": "_handle_thinking",
        "thinking_summary": "_handle_thinking_summary",
        "image_gen_tool": "_handle_image_gen_tool",
        "image_gen": "_handle_image_gen",
        "video_gen": "_handle_video_gen",
        "usage": "_handle_usage",
        "other": "_handle_other",
    }

    async def _invoke_handler(self, handler_name: str, event: Dict[str, Any]) -> Union[str, Dict[str, Any], List, None]:
        handler = getattr(self, handler_name)
        if asyncio.iscoroutinefunction(handler):
            return await handler(event)
        return handler(event)

    async def _yield_handler_result(self, handler_name: str, event: Dict[str, Any]) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        result = await self._invoke_handler(handler_name, event)
        if result is None:
            return
        for item in (result if isinstance(result, list) else (result,)):
            yield item

    async def _dispatch(
        self,
        event: Dict[str, Any],
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        event_type = event.get("type", "")
        if event_type == "error":
            self._handle_error(event)
        elif event_type == "response_created":
            self.last_response_id = event.get("response_id")
        else:
            handler_name = self._SIMPLE_HANDLERS.get(event_type)
            if handler_name:
                async for item in self._yield_handler_result(handler_name, event):
                    yield item
        if event.get("usage") and event_type != "usage":
            yield {"usage": event["usage"]}

    def _iter_thinking_pieces(self, event: Dict[str, Any]) -> List[str]:
        if event.get("status") != "typing":
            return []
        extra = event.get("extra", {})
        titles = extra.get("summary_title", {}).get("content", [])
        thoughts = extra.get("summary_thought", {}).get("content", [])
        total = max(len(titles), len(thoughts))
        pieces: List[str] = []
        for index in range(self._thinking_count, total):
            title = titles[index] if index < len(titles) else ""
            thought = thoughts[index] if index < len(thoughts) else ""
            pieces.append(f"{title}: {thought}" if title else thought)
        self._thinking_count = total
        return pieces

    async def _build_single_image_call(self, url: str) -> Dict[str, Any]:
        local_path = await self._download_image(url)
        arguments: Dict[str, Any] = {"url": url}
        if local_path:
            arguments["local_path"] = local_path
        return self._wrap_tool_call("qwen.image_gen", arguments)

    @staticmethod
    def _strip_tags(content: str) -> str:
        return content.replace("<think>", "").replace("</think>", "")

    @staticmethod
    def _wrap_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
