import secrets
from datetime import datetime
from typing import Any, Dict, Optional


def make_stream_id() -> str:

    return "{}-{}".format(
        datetime.now().strftime("%Y%m%d"),
        secrets.token_hex(8),
    )


def build_payload(
    session_id: str,
    prompt: str,
    model: str,
    *,
    stream_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构建 ``/api/v0/chat/completion`` 请求体。"""
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": "default",
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
        "action": None,
        "preempt": False,
    }


__all__ = [
    "build_payload",
    "make_stream_id",
]
