from __future__ import annotations

"""Zen（opencode.ai）上游常量。"""

from typing import Dict, Final, List

BASE_URL: Final[str] = "https://opencode.ai/zen/v1"
CHAT_PATH: Final[str] = "/chat/completions"
MODELS_PATH: Final[str] = "/models"

CONNECT_TIMEOUT: Final[float] = 60.0
STREAM_TOTAL_TIMEOUT: Final[float] = 600.0
STREAM_READ_TIMEOUT: Final[float] = 600.0
MODELS_FETCH_TIMEOUT: Final[float] = 60.0
MODELS_CACHE_TTL: Final[float] = 300.0

RETRY_COUNT: Final[int] = 2
USER_AGENT: Final[str] = "opencode/latest"

# 动态代理池后台刷新间隔（秒）；0 表示禁用
PROXY_REFRESH_INTERVAL: Final[float] = 86400.0

FALLBACK_MODEL: Final[str] = "mimo-v2.5-free"
FALLBACK_MODEL_ENABLED: Final[bool] = True
AUTO_REFRESH_MODELS: Final[bool] = False

DEFAULT_MODELS: Final[List[str]] = [
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
    "laguna-s-2.1-free",
]

DEFAULT_CAPABILITIES: Final[Dict[str, bool]] = {
    "chat": True,
    "vision": True,
    "search": False,
    "count_tokens": True,
    "image_gen": False,
    "tts": False,
}
