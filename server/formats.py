from __future__ import annotations

"""ID generation, format builders, and utilities for the Qwen adapter."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web


# ============================================================
# 全局常量
# ============================================================

PORT: int = 8932
MAX_CONCURRENT: int = 8
MAX_QUEUE_SIZE: int = 1000
PRELOGIN_ACCOUNT_COUNT: int = 3
MAX_CHARS: int = 1024000
REQUEST_TOTAL_TIMEOUT: float = 600.0
MODELS_FETCH_TIMEOUT: float = 60.0
LOGIN_TIMEOUT: float = 30.0
MAX_REQUEST_RESTARTS: int = 3
RESTART_DELAY: float = 1.0
DEFAULT_MODEL: str = "qwen3.7-max"
TOKEN_EXPIRE_HOURS: int = 12
TOKEN_EXPIRE_SECONDS: int = TOKEN_EXPIRE_HOURS * 3600
DATA_DIR: str = "data/qwen"
MODELS_CACHE_FILE: str = f"{DATA_DIR}/models.json"
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 10.0
SHUTDOWN_TOTAL_TIMEOUT: float = 15.0
RUNNER_SHUTDOWN_TIMEOUT: float = 10.0
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
DEFAULT_MODELS: List[str] = [
    "qwen3.7-max", "qwen3.6-plus", "qwen3.5-plus",
    "qwen3-max", "qwen3-235b-a22b",
]
CAPABILITIES: Dict[str, bool] = {
    "chat": True, "vision": True, "thinking": True,
    "search": True, "tools": True, "native_tools": True,
}
KEEPALIVE_INTERVAL: float = 5.0


# ============================================================
# ID 生成（对齐 OpenCode 格���）
# ============================================================

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:12]}"

def _gen_chatcmpl_id() -> str:
    return _gen_id("gen")

def _gen_request_id() -> str:
    return _gen_id("req")

def _gen_msg_id() -> str:
    return _gen_id("msg")

def _gen_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


# ============================================================
# 自定义异常
# ============================================================

class TokenExpiredError(Exception):
    """Token 过期，需要切换 session"""
    pass


# ============================================================
# 消息工具函数
# ============================================================

def _extract_text_from_content(content: Any) -> str:
    """从 content 中提取文本（支持 list 和 str 格式）。"""
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
        return ""
    return content if isinstance(content, str) else str(content)


def extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
    """提取最后一条 user 消息的 content。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text_from_content(msg.get("content", ""))
    return ""


# ============================================================
# 工具函数
# ============================================================

def _json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(
        data, status=status,
        dumps=lambda x: json.dumps(x, ensure_ascii=False),
    )

def _error_response(status: int, message: str, error_type: str = "invalid_request_error") -> web.Response:
    return _json_response({"error": {"message": message, "type": error_type, "code": status}}, status=status)


def _fix_tool_call_id(tc: Dict[str, Any]) -> Dict[str, Any]:
    """替换 echotools 硬编码的 call_0000 为唯一 UUID。"""
    raw_id = tc.get("id", "")
    if (not raw_id or
        raw_id == "call_0000" or
        raw_id == "toolu_call_0001" or
        raw_id.startswith("toolu_call_") or
        raw_id.startswith("call_")):
        call_id = _gen_tool_id()
    else:
        call_id = raw_id
    func = tc.get("function", {})
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "arguments": func.get("arguments", "{}"),
        }
    }


# ============================================================
# 长文本分割器
# ============================================================

class LongTextSplitter:
    def __init__(self, max_chars: int = MAX_CHARS):
        self.max_chars = max_chars

    def split(self, text: str):
        if len(text) <= self.max_chars:
            return text, None, None
        send_text = text[-self.max_chars:]
        remaining_text = text[:-self.max_chars]
        filename = f"remaining_{int(time.time())}_{uuid.uuid4().hex[:8]}.txt"
        return send_text, filename, remaining_text.encode("utf-8")


# ============================================================
# 格式构建（对�� OpenCode 输出格式）
# ============================================================

def build_openai_chunk(
    model: str,
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chunk_id: Optional[str] = None,
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

    result = {
        "id": chunk_id or _gen_chatcmpl_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None}],
    }
    return result


def _build_usage_dict() -> Dict[str, Any]:
    """构建默认 usage 字典（所有值为 0）。"""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _build_openai_message(
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """构建 OpenAI 响应中的 message 对象。"""
    message: Dict[str, Any] = {
        "role": "assistant",
        "refusal": None,
        "content": None if tool_calls else content,
    }
    if reasoning:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{
            "type": "reasoning.text",
            "text": reasoning,
            "format": "unknown",
            "index": 0,
        }]
    if tool_calls:
        message["tool_calls"] = [
            {
                "index": i,
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": tc.get("function", {}),
            }
            for i, tc in enumerate(tool_calls)
        ]
    return message


def build_openai_response(
    model: str, content: str, reasoning: str = "", finish_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """构建完整的非流式 OpenAI 响应，完全对齐 OpenCode 格式。"""
    if tool_calls:
        finish_reason = "tool_calls"

    message: Dict[str, Any] = {"role": "assistant", "refusal": None}

    if tool_calls:
        message["content"] = None
        message["tool_calls"] = []
        for tc in tool_calls:
            message["tool_calls"].append({
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                }
            })
    else:
        message["content"] = content

    if reasoning:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{
            "type": "reasoning.text", "text": reasoning, "format": "unknown", "index": 0
        }]

    return {
        "id": _gen_chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": _build_usage_dict(),
        "cost": "0",
    }

def convert_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        return {
            "id": _gen_msg_id(), "type": "message", "role": "assistant", "content": [],
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
        tool_id = tc.get("id") or _gen_tool_id()
        if not tool_id.startswith("toolu_"):
            tool_id = "toolu_" + tool_id
        anth_content.append({"type": "tool_use", "id": tool_id, "name": func.get("name", ""), "input": args_json})
    if not anth_content:
        anth_content.append({"type": "text", "text": ""})
    usage = response.get("usage", {})
    return {
        "id": response.get("id", _gen_msg_id()), "type": "message", "role": "assistant",
        "content": anth_content, "model": response.get("model", ""),
        "stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
    }
