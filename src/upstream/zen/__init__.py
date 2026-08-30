from __future__ import annotations

"""Zen (opencode.ai) 上游聚合入口与公共常量。

`__init__.py` 承担两件事：

1. 暴露 NAME/CAPABILITIES 与 ``create_client``/``stream_openai_chat`` 等注册表 hook，
   这些 hook 被 ``server.model.platform_models`` 扫描以发现新上游。
2. 持有所有跨模块复用的常量（超时、URL、能力、模型列表）。常量原本位于
   ``routes.py``；合并到这里后 ``__init__.py`` 行数豁免，避免额外文件存在
   又被计入目录子项导致违规。

循环 import 的消除方式：``client/chat_stream/openai_chat/proxy`` 模块里
``from upstream.zen.routes import ...`` 改为 ``from upstream.zen import ...``，
常量通过本模块的 ``globals()`` 透出，调用方仍可用 ``from upstream.zen import NAME``。
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from upstream.caps import load_capabilities

logger = logging.getLogger("rogator")

NAME = "zen"

# === Base URLs / paths ===
BASE_URL: str = "https://opencode.ai/zen/v1"
CHAT_PATH: str = "/chat/completions"
MODELS_PATH: str = "/models"

# === Timeouts (seconds) ===
CONNECT_TIMEOUT: float = 60.0
STREAM_TOTAL_TIMEOUT: float = 600.0
STREAM_READ_TIMEOUT: float = 600.0
MODELS_FETCH_TIMEOUT: float = 120.0
MODELS_CACHE_TTL: float = 300.0

# === Retry / identity ===
RETRY_COUNT: int = 2
USER_AGENT: str = "opencode/latest"

# === Dynamic proxy pool ===
# 后台刷新间隔（秒）；0 表示禁用
PROXY_REFRESH_INTERVAL: float = 86400.0
# 动态池去重后选取最快的节点数；0 表示全部选取
DEFAULT_DYNAMIC_TOP_N: int = 20

# === Model registry defaults ===
FALLBACK_MODEL: str = "mimo-v2.5-free"
FALLBACK_MODEL_ENABLED: bool = True
AUTO_REFRESH_MODELS: bool = False

DEFAULT_MODELS: List[str] = [
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
    "laguna-s-2.1-free",
]

DEFAULT_CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": True,
    "search": False,
    "count_tokens": True,
    "image_gen": False,
    "tts": False,
}

CAPABILITIES: Dict[str, bool] = load_capabilities(NAME, DEFAULT_CAPABILITIES)


def get_constant(name: str) -> Optional[Any]:
    """``from upstream.zen import NAME`` 的轻量 helper，便于按需延后取。"""
    return globals().get(name)


def create_client(splitter: Any = None) -> Any:
    from upstream.zen.client import ZenClient

    return ZenClient(splitter)


async def stream_openai_chat(
    state: Any,
    client: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    from upstream.zen.openai_chat import stream_openai_chat as _stream

    async for event in _stream(
        state,
        client,
        messages,
        model,
        tools,
        req_id,
        protocol_options=protocol_options,
        prompt_api=prompt_api,
        files=files,
    ):
        yield event


__all__ = [
    "NAME",
    "BASE_URL",
    "CHAT_PATH",
    "MODELS_PATH",
    "CONNECT_TIMEOUT",
    "STREAM_TOTAL_TIMEOUT",
    "STREAM_READ_TIMEOUT",
    "MODELS_FETCH_TIMEOUT",
    "MODELS_CACHE_TTL",
    "RETRY_COUNT",
    "USER_AGENT",
    "PROXY_REFRESH_INTERVAL",
    "DEFAULT_DYNAMIC_TOP_N",
    "FALLBACK_MODEL",
    "FALLBACK_MODEL_ENABLED",
    "AUTO_REFRESH_MODELS",
    "DEFAULT_MODELS",
    "DEFAULT_CAPABILITIES",
    "CAPABILITIES",
    "create_client",
    "stream_openai_chat",
]