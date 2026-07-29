from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


def extract_text_from_content(content: Any) -> str:
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
) -> Dict[str, Any]:
    return {
        "fid": str(uuid.uuid4()), "parentId": None, "childrenIds": [str(uuid.uuid4())],
        "role": "user", "content": user_content, "user_action": "chat",
        "files": files or [], "timestamp": int(time.time() * 1000), "models": [model],
        "chat_type": "t2t", "feature_config": {
            "thinking_enabled": thinking_enabled,
            "output_schema": "phase", "research_mode": "normal",
            "auto_thinking": auto_thinking,
            "thinking_mode": thinking_mode,
            "thinking_format": "raw",
            "auto_search": False,
        }, "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t",
    }


def build_chat_payload(
    chat_id: str,
    model: str,
    qwen_message: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stream": True, "version": "2.1", "incremental_output": True,
        "chat_id": chat_id, "chat_mode": "local", "model": model, "parent_id": None,
        "messages": [qwen_message], "timestamp": int(time.time() * 1000),
    }
