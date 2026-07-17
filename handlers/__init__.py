from __future__ import annotations

"""HTTP request handlers — shared utilities, helpers and admin endpoints."""

import json
from typing import Any, Dict, List, Optional

from aiohttp import web

from echotools.fncall import inject_fncall
from echotools.logger import get_logger
from echotools.protocol.base import ToolProtocol

from server.formats import _error_response, _json_response, extract_last_user_content
from state import AppState

logger = get_logger("rogator")

# 工具调用指令（当有 tools 时拼接到 user_content 前）
TOOL_INSTRUCTION = """
<ultra_system_reminder>
## Function Definitions

All functions are defined inside a `<functions>` wrapper block. Each function is a `<tool>` tag containing `description`, `name`, and `parameters`. Each parameter is a `<parameter>` tag with `name`, `type`, `required`, and `<description>`.

**Function Invocation Syntax:**

When calling tools, respond with ONLY the following XML block format:

<entml:function_calls>
<entml:invoke name="tool_name">
<entml:parameters>
<param_name>value</param_name>
</entml:parameters>
</entml:invoke>
</entml:function_calls>

Multiple invocations can be stacked inside one `<entml:function_calls>` block for parallel execution.

## Function Call Instructions

Use parameter names exactly as defined. Put each parameter value directly between its opening and closing tags. Do not wrap values in quotes. Do not use JSON inside `<entml:parameters>`.

If you intend to call multiple tools and there are no dependencies between the calls, make all independent calls in the same function_calls block. Otherwise, wait for previous calls to finish to determine dependent values.
</ultra_system_reminder>
"""


class EmptyResponseError(Exception):
    """模型返回空响应"""
    pass


# ============================================================
# 辅助函数
# ============================================================

def replace_last_user_content(
    messages: List[Dict[str, Any]],
    new_content: str,
) -> List[Dict[str, Any]]:
    """替换最后一条 user 消息的 content，返回新列表。"""
    new_messages = list(messages)
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            new_messages[i] = {**new_messages[i], "content": new_content}
            break
    return new_messages


def fold_system_into_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 system 消息合并进最后一条 user 消息。"""
    sys_parts: List[str] = []
    non_sys: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                sys_parts.append(content if isinstance(content, str) else str(content))
        else:
            non_sys.append(msg)
    if not sys_parts:
        return messages
    sys_text = "\n\n".join(sys_parts)
    merged = list(non_sys)
    for i in range(len(merged) - 1, -1, -1):
        if merged[i].get("role") == "user":
            old = merged[i].get("content", "")
            old_text = old if isinstance(old, str) else str(old)
            merged[i] = {**merged[i], "content": sys_text + "\n\n" + old_text}
            return merged
    merged.insert(0, {"role": "user", "content": sys_text})
    return merged




# ============================================================
# 猴子补丁：inject_fncall 没工具时也返回 prompt
# ============================================================

_original_inject_fncall = inject_fncall


def _patched_inject_fncall(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    protocol: ToolProtocol,
    lang: str = "en",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """补丁：没工具时也构建 prompt，返回单条 user 消息。"""
    if not tools:
        current_user_message = extract_last_user_content(messages)
        prompt = f"<current_user_message>\n{current_user_message}\n</current_user_message>"
        return [{"role": "user", "content": prompt}]
    return _original_inject_fncall(messages, tools, protocol, lang=lang, **kwargs)


inject_fncall = _patched_inject_fncall


# ============================================================
# 全局状态
# ============================================================

_app_state: Optional[AppState] = None


def get_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state


# ============================================================
# Health / Admin handlers
# ============================================================

async def health_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "ok",
        "platform": "rogator",
        "timestamp": int(__import__("time").time()),
    })


async def list_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "qwen"}
            for m in state._models
        ],
    })


async def anthropic_list_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    now = int(__import__("time").time())
    return _json_response({
        "type": "list",
        "data": [
            {"type": "model", "id": m, "display_name": m, "created_at": now}
            for m in state._models
        ],
        "has_more": False,
    })


async def anthropic_root_handler(request: web.Request) -> web.Response:
    return web.Response(
        status=200,
        headers={
            "Content-Type": "application/json",
            "Anthropic-Version": "2023-06-01",
        },
        text=json.dumps({
            "type": "message",
            "version": "2023-06-01",
            "status": "ok",
            "endpoints": ["/v1/messages", "/anthropic/v1/messages"],
        }),
    )


async def count_tokens_handler(request: web.Request) -> web.Response:
    """估算请求消息的 token 数量（OpenAI / Anthropic 兼容）。"""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _json_response({"input_tokens": 0})
    messages = body.get("messages", []) or []
    system = body.get("system", "")
    total_chars = len(system) if isinstance(system, str) else 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    total_chars += len(part.get("text", ""))
    return _json_response({"input_tokens": _estimate_tokens_from_chars(total_chars)})


def _estimate_tokens_from_chars(total_chars: int) -> int:
    return max(0, total_chars // 3)


async def audio_speech_handler(request: web.Request) -> web.Response:
    """OpenAI 兼容的 TTS 端点，委托给 QwenClient.synthesize_tts。"""
    state = get_state()
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error_response(400, "Invalid JSON body")
    text = body.get("input", "")
    if not text:
        return _error_response(400, "Missing required field: input")
    model = body.get("model") or state.model
    session = await state.client.get_valid_session()
    if not session:
        return _error_response(503, "No valid Qwen session available")
    local_path = await state.client.synthesize_tts(text, session.token, model=model)
    if not local_path:
        return _error_response(502, "TTS synthesis failed")
    from pathlib import Path
    audio_bytes = Path(local_path).read_bytes()
    return web.Response(body=audio_bytes, content_type="audio/wav")


async def images_generations_handler(request: web.Request) -> web.Response:
    """OpenAI 兼容的图片生成端点（图生图/图生视频前置帧），委托给 QwenClient.generate_video。"""
    state = get_state()
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error_response(400, "Invalid JSON body")
    prompt = body.get("prompt", "")
    image_url = body.get("image") or body.get("image_url", "")
    if not prompt or not image_url:
        return _error_response(400, "Missing required fields: prompt, image")
    model = body.get("model") or state.model
    size = body.get("size", "16:9")
    session = await state.client.get_valid_session()
    if not session:
        return _error_response(503, "No valid Qwen session available")
    result = await state.client.generate_video(
        prompt, image_url, session.token, session.user_id, model=model, size=size,
    )
    if not result.get("success"):
        return _error_response(502, result.get("error", "Generation failed"))
    return _json_response({
        "created": int(__import__("time").time()),
        "data": [{"url": result.get("video_url", ""), "local_path": result.get("local_path", "")}],
    })


async def capabilities_handler(request: web.Request) -> web.Response:
    from server.formats import CAPABILITIES
    return _json_response({
        "platform": "rogator",
        "capabilities": CAPABILITIES,
        "protocol": "entml",
    })


async def status_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "running",
        "sessions": {"total": state.client.session_count},
        "scheduler": {"pending": state.scheduler.pending, "active": state.tracker.count},
        "models": {"count": len(state._models), "default": state.model},
    })


async def admin_refresh_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    await state.refresh_models()
    return _json_response({
        "status": "ok",
        "models": state._models,
        "count": len(state._models),
    })


async def admin_switch_session_handler(request: web.Request) -> web.Response:
    state = get_state()
    old = (
        state.client.current_session.username[:6]
        if state.client.current_session else "none"
    )
    new = await state.client.switch_to_next()
    return _json_response({
        "status": "ok",
        "previous": old,
        "current": new.username[:6] if new else "none",
    })


async def admin_sessions_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "sessions": [
            {"username": s.username[:6] + "***", "valid": s.is_valid}
            for s in state.client._sessions
        ],
        "total": state.client.session_count,
    })


# ============================================================
# 路由
# ============================================================

def setup_routes(app: web.Application) -> None:
    from handlers.anthro import anthropic_messages_handler
    from handlers.openai import openai_chat_handler

    routes = [
        ("GET", "/", health_handler),
        ("GET", "/health", health_handler),
        ("GET", "/v1/health", health_handler),
        ("GET", "/v1/models", list_models_handler),
        ("POST", "/v1/chat/completions", openai_chat_handler),
        ("POST", "/v1/messages/count_tokens", count_tokens_handler),
        ("GET", "/anthropic", anthropic_root_handler),
        ("POST", "/anthropic", anthropic_root_handler),
        ("POST", "/v1/messages", anthropic_messages_handler),
        ("POST", "/anthropic/v1/messages", anthropic_messages_handler),
        ("POST", "/anthropic/messages", anthropic_messages_handler),
        ("GET", "/anthropic/v1/models", anthropic_list_models_handler),
        ("POST", "/anthropic/v1/messages/count_tokens", count_tokens_handler),
        ("POST", "/v1/images/generations", images_generations_handler),
        ("POST", "/v1/audio/speech", audio_speech_handler),
        ("POST", "/v1/admin/refresh_models", admin_refresh_models_handler),
        ("POST", "/v1/admin/switch_session", admin_switch_session_handler),
        ("GET", "/v1/admin/sessions", admin_sessions_handler),
        ("GET", "/v1/capabilities", capabilities_handler),
        ("GET", "/v1/status", status_handler),
    ]
    for method, path, handler in routes:
        getattr(app.router, f"add_{method.lower()}")(path, handler)
