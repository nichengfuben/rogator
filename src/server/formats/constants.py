from __future__ import annotations

from typing import Dict, List

PORT: int = 8932
MAX_CONCURRENT: int = 8
MAX_QUEUE_SIZE: int = 1000
PRELOGIN_ACCOUNT_COUNT: int = 3
REQUEST_TOTAL_TIMEOUT: float = 600.0
MODELS_FETCH_TIMEOUT: float = 60.0
LOGIN_TIMEOUT: float = 30.0
MAX_REQUEST_RESTARTS: int = 3
RESTART_DELAY: float = 1.0
DEFAULT_MODEL: str = "qwen3.7-max"
TOKEN_EXPIRE_HOURS: int = 12
TOKEN_EXPIRE_SECONDS: int = TOKEN_EXPIRE_HOURS * 3600
DATA_DIR: str = "persist/qwen"
MODELS_CACHE_FILE: str = f"{DATA_DIR}/models.json"
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 3.0
SHUTDOWN_TOTAL_TIMEOUT: float = 8.0
RUNNER_SHUTDOWN_TIMEOUT: float = 10.0
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
DEFAULT_MODELS: List[str] = [
    "qwen3.8-max-preview",
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen3.5-plus",
    "qwen3.5-397b-a17b",
    "qwen3-max",
    "qwen3-max-2026-01-23",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-vl-30b-a3b",
    "qwen3-vl-32b",
    "qwen3-vl-plus",
    "qwen3-coder-plus",
    "qwen3-coder-30b-a3b-instruct",
    "qwen3-omni-flash",
    "qwen3-omni-flash-2025-12-01",
    "qwen2.5-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-omni-7b",
    "qwen2.5-coder-32b-instruct",
    "qwen-max-latest",
    "qwen-plus-2025-07-28",
    "qwen-plus-2025-09-11",
    "qwen-plus-2025-01-25",
    "qwen-turbo-2025-02-11",
]
CAPABILITIES: Dict[str, bool] = {
    "chat": True, "vision": True, "thinking": True,
    "search": True, "tools": True, "native_tools": True,
    "count_tokens": True, "image_gen": True, "tts": True,
}
KEEPALIVE_INTERVAL: float = 5.0


def gen_id(prefix: str) -> str:
    import time
    import uuid

    return f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:12]}"


def gen_chatcmpl_id() -> str:
    return gen_id("gen")


def gen_request_id() -> str:
    return gen_id("req")


def gen_msg_id() -> str:
    return gen_id("msg")


def gen_tool_id() -> str:
    import uuid

    return f"toolu_{uuid.uuid4().hex[:24]}"
