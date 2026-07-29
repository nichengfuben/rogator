from __future__ import annotations

"""ID 生成、格式构建与 Qwen 适配工具（按职责拆分子模块）。"""

from server.formats.constants import (
    CAPABILITIES,
    DATA_DIR,
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    DEFAULT_USER_AGENT,
    KEEPALIVE_INTERVAL,
    LOGIN_TIMEOUT,
    MAX_CONCURRENT,
    MAX_QUEUE_SIZE,
    MAX_REQUEST_RESTARTS,
    MODELS_CACHE_FILE,
    MODELS_FETCH_TIMEOUT,
    PORT,
    PRELOGIN_ACCOUNT_COUNT,
    REQUEST_TOTAL_TIMEOUT,
    RESTART_DELAY,
    RUNNER_SHUTDOWN_TIMEOUT,
    SHUTDOWN_CANCEL_GRACE,
    SHUTDOWN_TOTAL_TIMEOUT,
    SHUTDOWN_WAIT_IDLE_TIMEOUT,
    TOKEN_EXPIRE_HOURS,
    TOKEN_EXPIRE_SECONDS,
    gen_chatcmpl_id,
    gen_msg_id,
    gen_request_id,
    gen_tool_id,
)
from server.formats.errors import (
    ClientDisconnectedError,
    PayloadTooLargeError,
    TokenExpiredError,
    UpstreamTimeoutError,
    client_disconnected_response,
    error_response,
    fix_tool_call_id,
    json_response,
    read_request_json,
)
from server.formats.messages import (
    build_chat_payload,
    build_qwen_message,
    extract_last_user_content,
    extract_text_from_content,
)
from server.formats.openai_build import (
    build_openai_chunk,
    build_openai_response,
    build_openai_stream_usage_chunk,
    convert_to_anthropic,
    openai_stream_include_usage,
)
from server.formats.usage import (
    UpstreamUsageTracker,
    build_usage_dict,
    normalize_upstream_usage,
    should_emit_anthropic_message_start,
)

# 兼容旧私有命名
_gen_chatcmpl_id = gen_chatcmpl_id
_gen_request_id = gen_request_id
_gen_msg_id = gen_msg_id
_gen_tool_id = gen_tool_id
_json_response = json_response
_error_response = error_response
_fix_tool_call_id = fix_tool_call_id
_build_usage_dict = build_usage_dict

__all__ = [
    "CAPABILITIES",
    "ClientDisconnectedError",
    "DATA_DIR",
    "DEFAULT_MODEL",
    "DEFAULT_MODELS",
    "DEFAULT_USER_AGENT",
    "KEEPALIVE_INTERVAL",
    "LOGIN_TIMEOUT",
    "MAX_CONCURRENT",
    "MAX_QUEUE_SIZE",
    "MAX_REQUEST_RESTARTS",
    "MODELS_CACHE_FILE",
    "MODELS_FETCH_TIMEOUT",
    "PORT",
    "PRELOGIN_ACCOUNT_COUNT",
    "PayloadTooLargeError",
    "REQUEST_TOTAL_TIMEOUT",
    "RESTART_DELAY",
    "RUNNER_SHUTDOWN_TIMEOUT",
    "SHUTDOWN_CANCEL_GRACE",
    "SHUTDOWN_TOTAL_TIMEOUT",
    "SHUTDOWN_WAIT_IDLE_TIMEOUT",
    "TOKEN_EXPIRE_HOURS",
    "TOKEN_EXPIRE_SECONDS",
    "TokenExpiredError",
    "UpstreamTimeoutError",
    "UpstreamUsageTracker",
    "build_chat_payload",
    "build_openai_chunk",
    "build_openai_response",
    "build_openai_stream_usage_chunk",
    "build_qwen_message",
    "client_disconnected_response",
    "convert_to_anthropic",
    "extract_last_user_content",
    "normalize_upstream_usage",
    "openai_stream_include_usage",
    "read_request_json",
    "should_emit_anthropic_message_start",
]
