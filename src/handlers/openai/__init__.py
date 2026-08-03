from __future__ import annotations

"""OpenAI-compatible chat completion handlers."""

from handlers.openai.chat import _chat_once, _process_openai_non_stream
from handlers.openai.handler import openai_chat_handler
from handlers.openai.protocol import (
    _build_protocol_options,
    _inject_protocol_options,
    _map_to_thinking_level,
    protocol_thinking_level,
    thinking_level_is_active,
)
from handlers.openai.tools import _parse_tool_calls, convert_tools_to_openai

__all__ = [
    "openai_chat_handler",
    "_chat_once",
    "_process_openai_non_stream",
    "_parse_tool_calls",
    "convert_tools_to_openai",
    "protocol_thinking_level",
    "thinking_level_is_active",
]
