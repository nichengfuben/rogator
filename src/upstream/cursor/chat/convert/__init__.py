from __future__ import annotations

"""OpenAI ↔ Cursor 消息/模型转换。"""

from upstream.cursor.chat.convert.text import (
    IMPORTANT_MCP_TOOLS_ONLY,
    IMPORTANT_NO_TOOLS,
    build_cursor_turn,
    build_custom_system_prompt,
    build_prepend_user_messages,
    extract_system_texts,
    map_model,
    messages_to_prompt,
    prepend_system_to_prompt,
)
from upstream.cursor.chat.convert.history import (
    format_prior_context_user_text,
    format_tool_results_user_text,
    messages_to_cursor_history,
    split_prompt_and_history,
)
from upstream.cursor.chat.convert.tools import (
    _tool_name_match_keys,
    openai_tools_to_mcp,
    original_tool_names,
    restore_mcp_prefix_for_cursor,
    rewrite_tool_call_for_openai,
    split_mcp_tool_identity,
    strip_mcp_prefix,
)

__all__ = [
    "IMPORTANT_MCP_TOOLS_ONLY",
    "IMPORTANT_NO_TOOLS",
    "build_cursor_turn",
    "build_custom_system_prompt",
    "build_prepend_user_messages",
    "extract_system_texts",
    "format_prior_context_user_text",
    "format_tool_results_user_text",
    "map_model",
    "messages_to_cursor_history",
    "messages_to_prompt",
    "openai_tools_to_mcp",
    "original_tool_names",
    "prepend_system_to_prompt",
    "restore_mcp_prefix_for_cursor",
    "rewrite_tool_call_for_openai",
    "split_mcp_tool_identity",
    "split_prompt_and_history",
    "strip_mcp_prefix",
    "_tool_name_match_keys",
]
