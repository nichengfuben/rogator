from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from server.formats.constants import gen_chatcmpl_id, gen_msg_id, gen_tool_id
from server.formats.usage import build_usage_dict


def _openai_delta(
    *,
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {"role": "assistant"}
    if tool_calls is not None:
        delta["content"] = None
        delta["tool_calls"] = tool_calls
    elif reasoning is not None:
        delta["content"] = ""
        delta["reasoning"] = reasoning
        delta["reasoning_details"] = [{
            "type": "reasoning.text",
            "text": reasoning,
            "format": "unknown",
            "index": 0,
        }]
    elif content is not None:
        delta["content"] = content
    else:
        delta["content"] = ""
    return delta


def build_openai_chunk(
    model: str,
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chunk_id: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, Any]] = None,
    *,
    usage_null: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": chunk_id or gen_chatcmpl_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": _openai_delta(content=content, reasoning=reasoning, tool_calls=tool_calls),
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
    }
    if usage_null:
        result["usage"] = None
    elif usage is not None:
        result["usage"] = usage
    return result


def build_openai_stream_usage_chunk(
    model: str,
    chunk_id: str,
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    """OpenAI 流式：include_usage 时 finish 之后的独立 usage chunk（choices 为空）。"""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }


def openai_stream_include_usage(body: Optional[Dict[str, Any]]) -> bool:
    """请求是否启用了 stream_options.include_usage。"""
    if not body:
        return False
    opts = body.get("stream_options")
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("include_usage"))


def _assistant_message_body(
    content: str,
    reasoning: str,
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "refusal": None}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = [
            {
                "index": i,
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    else:
        message["content"] = content
    if reasoning:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{
            "type": "reasoning.text", "text": reasoning, "format": "unknown", "index": 0,
        }]
    return message


def build_openai_response(
    model: str,
    content: str,
    reasoning: str = "",
    finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建完整的非流式 OpenAI 响应，完全对齐 OpenCode 格式。"""
    if tool_calls:
        finish_reason = "tool_calls"
    message = _assistant_message_body(content, reasoning, tool_calls)
    return {
        "id": gen_chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": usage if usage is not None else build_usage_dict(),
        "cost": "0",
    }


def convert_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        return {
            "id": gen_msg_id(), "type": "message", "role": "assistant", "content": [],
            "model": response.get("model", ""), "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    message = choices[0].get("message", {})
    content_text = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    tool_calls = message.get("tool_calls") or []
    anth_content: List[Dict[str, Any]] = []
    if reasoning:
        anth_content.append({"type": "thinking", "thinking": reasoning})
    if content_text:
        anth_content.append({"type": "text", "text": content_text})
    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        try:
            args_json = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            args_json = {}
        tool_id = tc.get("id") or gen_tool_id()
        if not tool_id.startswith("toolu_"):
            tool_id = "toolu_" + tool_id
        anth_content.append({
            "type": "tool_use",
            "id": tool_id,
            "name": func.get("name", ""),
            "input": args_json,
        })
    if not anth_content:
        anth_content.append({"type": "text", "text": ""})
    usage = response.get("usage", {})
    return {
        "id": response.get("id", gen_msg_id()), "type": "message", "role": "assistant",
        "content": anth_content, "model": response.get("model", ""),
        "stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
