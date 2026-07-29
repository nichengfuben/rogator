from __future__ import annotations

"""HTTP endpoints, constants, and settings for the Qwen adapter.

Merged from: endpoints.py, constants.py, settings.py
"""

from typing import Any, Dict, Final, List

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

BASE_URL: Final[str] = "https://chat.qwen.ai"
AUTH_BASE_URL: Final[str] = "https://auth.qwen.ai"
CHAT_ORIGIN: Final[str] = "https://chat.qwen.ai"
AUTH_API_PREFIX: Final[str] = "/api/v2"
CHAT_API_PREFIX: Final[str] = "/api/v2"
APP_VERSION: Final[str] = "0.2.64"
WEB_VERSION: Final[str] = "0.2.9"
API_VERSION: Final[str] = "2.1"
BAXIA_VERSION: Final[str] = "0.0.3"
BXUA_VERSION: Final[str] = BAXIA_VERSION
BAXIA_SDK_VERSION: Final[str] = BAXIA_VERSION
USE_LOCAL_MODE: Final[bool] = True
USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
USER_AGENT_MOBILE: Final[str] = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
)
SEC_CH_UA: Final[str] = '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"'
SEC_CH_UA_PLATFORM: Final[str] = '"macOS"'
FRONTEND_VERSION: Final[str] = WEB_VERSION
CUSTOM_BASE64_CHARS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"

SIGNIN_PATH: Final[str] = f"{AUTH_API_PREFIX}/auths/signin"
AUTH_CHECK_PATH: Final[str] = f"{AUTH_API_PREFIX}/user"
NEW_CHAT_PATH: Final[str] = f"{CHAT_API_PREFIX}/chats/new"
CHAT_PATH: Final[str] = f"{CHAT_API_PREFIX}/chat/completions"
STOP_CHAT_PATH: Final[str] = f"{CHAT_API_PREFIX}/chat/stop"
DELETE_CHAT_PATH: Final[str] = f"{CHAT_API_PREFIX}/chats/{{chat_id}}"
SETTINGS_PATH: Final[str] = f"{CHAT_API_PREFIX}/user/settings"
MODELS_PATH: Final[str] = f"{CHAT_API_PREFIX}/models"
TTS_PATH: Final[str] = f"{CHAT_API_PREFIX}/tts/completions"
TASK_STATUS_PATH: Final[str] = "/api/v1/tasks/status/{task_id}"
STS_TOKEN_PATHS: Final[List[str]] = [
    "/api/v1/files/getstsToken",
    "/api/v2/files/getstsToken",
]
VIDEO_CDN_BASE: Final[str] = "https://cdn.qwenlm.ai/output"

PERSIST_PATH: Final[str] = "persist/qwen/state.json"
MODELS_PERSIST_PATH: Final[str] = "persist/qwen/models.json"
TASK_TIMERS_PATH: Final[str] = "persist/qwen/task_timers.json"
PROXY_SELECTOR_PERSIST_PATH: Final[str] = "persist/qwen/proxy_selector.json"
GENERATED_IMAGE_DIR: Final[str] = "persist/qwen/generated_images"
GENERATED_VIDEO_DIR: Final[str] = "persist/qwen/generated_videos"
TTS_DIR: Final[str] = "persist/qwen/tts"
UPLOAD_TEMP_DIR: Final[str] = "persist/qwen/uploads"

LOGIN_BATCH_SIZE: Final[int] = 3
LOGIN_BATCH: Final[int] = LOGIN_BATCH_SIZE
LOGIN_CONCURRENCY: Final[int] = 1
LOGIN_POOL_SIZE: Final[int] = 8
LOGIN_SELECT_MIN: Final[int] = 2
LOGIN_SELECT_MAX: Final[int] = 5
INITIAL_LOGIN_MAX: Final[int] = 5
LOGIN_POLL_INTERVAL: Final[int] = 300
TOKEN_EXPIRY_MARGIN: Final[int] = 600
TOKEN_REFRESH_INTERVAL: Final[int] = 3600
COOKIE_REFRESH_INTERVAL: Final[int] = 1800
PERSIST_INTERVAL: Final[int] = 300
SSE_TIMEOUT: Final[int] = 300
TTS_TIMEOUT: Final[int] = 300
VIDEO_TASK_MAX_POLL_TIME: Final[int] = 900
VIDEO_TASK_POLL_INTERVAL: Final[int] = 5


# ---------------------------------------------------------------------------
# Constants (from constants.py)
# ---------------------------------------------------------------------------

MODELS: Final[List[str]] = [
    "qwen3.6-plus",
    "qwen3.5-plus",
    "qwen3.5-397b-a17b",
    "qwen3-max-2026-01-23",
    "qwen3-235b-a22b",
    "qwen3-vl-30b-a3b",
    "qwen3-vl-32b",
    "qwen3-vl-plus",
    "qwen3-coder-plus",
    "qwen3-omni-flash-2025-12-01",
    "qwen-plus-2025-07-28",
]

CAPS: Final[Dict[str, bool]] = {
    "chat": True,
    "vision": True,
    "thinking": True,
    "search": True,
    "image_gen": True,
    "image_edit": True,
    "audio_gen": True,
    "video_gen": True,
    "continuation": True,
    "artifacts": True,
}

SMART_PROXY_ENABLED: Final[bool] = True


# ---------------------------------------------------------------------------
# Settings (from settings.py)
# ---------------------------------------------------------------------------

DEFAULT_FULL_SETTINGS: Final[Dict[str, Any]] = {
    "ui": {
        "notificationEnabled": False,
        "theme": "dark",
        "language": "",
        "chatBubble": True,
        "showUsername": False,
        "widescreenMode": False,
        "title": {"auto": False},
        "autoTags": True,
        "largeTextAsFile": False,
        "splitLargeChunks": False,
        "scrollOnBranchChange": True,
        "responseAutoCopy": False,
        "models": [],
        "richTextInput": False,
    },
    "mcp_remind": False,
    "mcp": {
        "code-interpreter": False,
        "fire-crawl": False,
        "amap": False,
        "image-generation": False,
    },
    "memory": {
        "enable_memory": False,
        "enable_history_memory": False,
        "memory_version_reminder": False,
    },
    "reminder": {"project_version_reminder": False},
    "tts_speaker": {
        "speaker": "Cherry",
        "description": "一位阳光、积极、友好且自然的年轻女士",
        "url": "",
        "gender": "female",
    },
    "tts_speaker_v2": {
        "speaker": "Nini",
        "description": "像糯米糍一样软糯黏腻的嗓音",
        "url": "",
        "gender": "female",
        "is_personal": False,
        "speaker_id": "",
        "spk_name": "邻家妹妹",
    },
    "aipodcast": {"host": "", "guest": ""},
    "code_settings": {
        "custom_prompt": "",
        "diff_display": "split",
        "branch_format": "",
        "last_repo_choice": "",
        "last_branch_choice": "",
    },
    "manage_cookies": None,
    "personalization": {
        "name": "",
        "description": "",
        "style": None,
        "instruction": "",
        "enable_for_new_chat": False,
    },
    "tools_enabled": {
        "web_search": False,
        "web_extractor": False,
        "web_search_image": False,
        "image_gen_tool": True,
        "image_edit_tool": True,
        "code_interpreter": False,
        "bio": False,
        "history_retriever": False,
        "image_zoom_in_tool": False,
    },
}


__all__ = [
    "BASE_URL",
    "USER_AGENT",
    "USER_AGENT_MOBILE",
    "SEC_CH_UA",
    "FRONTEND_VERSION",
    "BAXIA_SDK_VERSION",
    "BXUA_VERSION",
    "CUSTOM_BASE64_CHARS",
    "MODELS",
    "CAPS",
    "SMART_PROXY_ENABLED",
    "MODELS_PERSIST_PATH",
    "PERSIST_PATH",
    "TASK_TIMERS_PATH",
    "PROXY_SELECTOR_PERSIST_PATH",
    "GENERATED_IMAGE_DIR",
    "GENERATED_VIDEO_DIR",
    "TTS_DIR",
    "UPLOAD_TEMP_DIR",
    "LOGIN_BATCH",
    "LOGIN_BATCH_SIZE",
    "LOGIN_CONCURRENCY",
    "LOGIN_POOL_SIZE",
    "LOGIN_SELECT_MIN",
    "LOGIN_SELECT_MAX",
    "INITIAL_LOGIN_MAX",
    "LOGIN_POLL_INTERVAL",
    "TOKEN_EXPIRY_MARGIN",
    "TOKEN_REFRESH_INTERVAL",
    "COOKIE_REFRESH_INTERVAL",
    "PERSIST_INTERVAL",
    "SSE_TIMEOUT",
    "TTS_TIMEOUT",
    "VIDEO_CDN_BASE",
    "VIDEO_TASK_MAX_POLL_TIME",
    "VIDEO_TASK_POLL_INTERVAL",
    "DEFAULT_FULL_SETTINGS",
]
