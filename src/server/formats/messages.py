from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
        return ""
    return content if isinstance(content, str) else str(content)


def extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return extract_text_from_content(msg.get("content", ""))
    return ""


def build_qwen_message(
    user_content: str,
    model: str,
    files: Optional[List[Dict[str, Any]]] = None,
    *,
    thinking_enabled: bool = False,
    thinking_mode: str = "Fast",
    auto_thinking: bool = False,
    auto_search: bool = False,
) -> Dict[str, Any]:
    # Message timestamp is seconds.
    ts = int(time.time())
    feature_config: Dict[str, Any] = {
        "thinking_enabled": thinking_enabled,
        "output_schema": "phase",
        "research_mode": "normal",
        "auto_thinking": auto_thinking,
        "thinking_mode": thinking_mode,
        "auto_search": auto_search,
    }
    if thinking_enabled or thinking_mode == "Thinking":
        feature_config["thinking_format"] = "raw"
    return {
        "id": None,
        "fid": str(uuid.uuid4()),
        "parentId": None,
        "childrenIds": [str(uuid.uuid4())],
        "role": "user",
        "content": user_content,
        "user_action": "chat",
        "files": files or [],
        "timestamp": ts,
        "models": [model],
        "model": "",
        "chat_type": "t2t",
        "feature_config": feature_config,
        "extra": {"meta": {"subChatType": "t2t"}},
        "sub_chat_type": "t2t",
        "parent_id": None,
    }


def build_chat_payload(
    chat_id: str,
    model: str,
    qwen_message: Dict[str, Any],
    *,
    include_usage: bool = False,
) -> Dict[str, Any]:
    # Dual-write camelCase chatId/parentId with snake_case chat_id/parent_id.
    # 上游 web completions 无 stream_options；include_usage 仅显式需要时开启。
    ts = int(qwen_message.get("timestamp") or time.time())
    payload: Dict[str, Any] = {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chatId": chat_id,
        "parentId": "",
        "chat_id": chat_id,
        "chat_mode": "local",
        "model": model,
        "parent_id": None,
        "messages": [qwen_message],
        "timestamp": ts,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload
