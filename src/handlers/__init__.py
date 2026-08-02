from __future__ import annotations

"""HTTP request handlers — shared utilities, helpers and admin endpoints."""

import json
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.base.logger import get_logger

from server.formats import (
    ClientDisconnectedError,
    _error_response,
    _json_response,
    client_disconnected_response,
    read_request_json,
)
from server.model.token_estimate import (
    estimate_anthropic_injected_input_tokens,
    estimate_anthropic_request_input_tokens,
)
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


def prepend_anthropic_system(
    messages: List[Dict[str, Any]],
    system: Any,
) -> List[Dict[str, Any]]:
    """将 Anthropic ``system`` 字段规范为首条 system 消息（支持 str / block 数组）。"""
    if system is None or system == "":
        return messages
    if isinstance(system, str):
        sys_text = system.strip()
    elif isinstance(system, list):
        sys_text = normalize_message_content(system).strip()
    else:
        sys_text = json.dumps(system, ensure_ascii=False).strip()
    if not sys_text:
        return messages
    return [{"role": "system", "content": sys_text}, *messages]


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
# 路由
# ============================================================

def setup_routes(app: web.Application) -> None:
    from handlers.shared.route_table import build_route_specs

    for method, path, handler in build_route_specs():
        getattr(app.router, f"add_{method.lower()}")(path, handler)
