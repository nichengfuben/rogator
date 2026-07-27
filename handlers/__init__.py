from __future__ import annotations

"""HTTP request handlers — shared utilities, helpers and admin endpoints."""

import json
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.logger import get_logger

from server.formats import _error_response, _json_response
from state import AppState

logger = get_logger("rogator")


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


def normalize_message_content(content: Any) -> str:
    """将 message content（str / block 数组）规范为纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                text = str(block).strip()
                if text:
                    parts.append(text)
                continue
            btype = block.get("type")
            if btype in ("text", "input_text") or (btype is None and "text" in block):
                text = str(block.get("text") or "").strip()
            elif btype in ("thinking", "redacted_thinking"):
                text = str(block.get("thinking") or block.get("data") or "").strip()
            elif "text" in block:
                text = str(block.get("text") or "").strip()
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def extract_system_for_inject(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """提取 system 为 ``user_system_prompt``，返回 (prompt, 不含 system 的消息列表)。"""
    sys_parts: List[str] = []
    non_sys: List[Dict[str, Any]] = []
    for msg in messages or []:
        if (msg.get("role") or "user") == "system":
            text = normalize_message_content(msg.get("content")).strip()
            if text:
                sys_parts.append(text)
        else:
            non_sys.append(msg)
    return "\n\n".join(sys_parts), non_sys


def fold_system_into_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[已弃用] 将 system 合并进 user；请改用 ``extract_system_for_inject`` + inject 参数。"""
    user_system_prompt, non_sys = extract_system_for_inject(messages)
    if not user_system_prompt:
        return messages
    merged = list(non_sys)
    for i in range(len(merged) - 1, -1, -1):
        if merged[i].get("role") == "user":
            old_text = normalize_message_content(merged[i].get("content"))
            merged[i] = {
                **merged[i],
                "content": user_system_prompt + "\n\n" + old_text if old_text else user_system_prompt,
            }
            return merged
    merged.insert(0, {"role": "user", "content": user_system_prompt})
    return merged


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
    from server.model_catalog import build_openai_models_list

    state = get_state()
    return _json_response({
        "object": "list",
        "data": build_openai_models_list(state._models),
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


def _message_content_char_count(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
            total += len(part.get("text", ""))
    return total


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
        total_chars += _message_content_char_count(msg.get("content", ""))
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
