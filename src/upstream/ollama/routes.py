from __future__ import annotations

"""Ollama 上游常量与过滤规则。"""

import re
from typing import Dict, Final

CHAT_PATH: Final[str] = "/api/chat"
EMBED_PATH: Final[str] = "/api/embed"

CONNECT_TIMEOUT: Final[float] = 10.0
STREAM_TOTAL_TIMEOUT: Final[float] = 600.0
STREAM_READ_TIMEOUT: Final[float] = 600.0

REGISTRY_FILE: Final[str] = "persist/ollama/registry.json"

# 匹配模型名/IP 中的垃圾数据
SKIP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'(leak|test|attacker|rogue|probe|backup|__sec|defvul|127\.0\.0\.1|192\.168\.)',
    re.I,
)

DEFAULT_CAPABILITIES: Final[Dict[str, bool]] = {
    "chat": True,
    "vision": True,
    "search": False,
    "count_tokens": False,
    "image_gen": False,
    "tts": False,
}
