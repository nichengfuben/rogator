

# src/platforms/deepseek/core/protocol/constants.py
"""DeepSeek 平台常量定义"""

from typing import Dict, List

# ── 模型 ────────────────────────────────────────────────────────────────────
MODEL_PRO: str = "deepseek-v4-pro"
MODEL_FLASH: str = "deepseek-v4-flash"
MODEL_VISION: str = "deepseek-v4-vision"
MODELS: List[str] = [MODEL_PRO, MODEL_FLASH, MODEL_VISION]

# ── 能力字典 ─────────────────────────────────────────────────────────────────
_BASE_CAPS: Dict[str, bool] = {
    "chat": True,
    "completions": True,
    "responses": True,
    "thinking": False,
    "search": False,
    "tools": True,
    "continuation": True,
}

CAPS_PRO: Dict[str, bool] = dict(_BASE_CAPS)
CAPS_FLASH: Dict[str, bool] = dict(_BASE_CAPS)

# vision 模型：仅保留支持能力
CAPS_VISION: Dict[str, bool] = {
    "chat": True,
    "vision": True,
}

# 三模型能力并集（用于 /v1/models 输出）
CAPS: Dict[str, bool] = dict(_BASE_CAPS)
CAPS["vision"] = CAPS_VISION["vision"]

# ── 服务端点 ──────────────────────────────────────────────────────────────────
DEFAULT_HOST: str = "chat.deepseek.com"
HIF_LEIM_URL: str = "https://hif-leim.deepseek.com/query"
HIF_DLIQ_URL: str = "https://hif-dliq.deepseek.com/query"

# ── WASM PoW ──────────────────────────────────────────────────────────────────
from server.config.files import PROJECT_ROOT

_DEEPSEEK_PERSIST = PROJECT_ROOT / "persist" / "deepseek"
WASM_PATH: str = str(_DEEPSEEK_PERSIST / "sha3_wasm_bg.7b9ca65ddd.wasm")
WASM_URL: str = (
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
)
WASM_META: str = str(_DEEPSEEK_PERSIST / "wasm_meta.json")

# ── 其他 ──────────────────────────────────────────────────────────────────────
MAX_CONTINUE: int = 10
MAX_RETRIES: int = 3
FETCH_MODELS_ENABLED: bool = False
MODEL_FETCH_INTERVAL: int = 86400
HIF_REFRESH_INTERVAL: float = 2700.0  # 秒，45 分钟

# ── 公共请求头 ────────────────────────────────────────────────────────────────
COMMON_HEADERS: Dict[str, str] = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "content-type": "application/json",
    "sec-ch-ua": (
        '"Chromium";v="146","Not-A.Brand";v="24","Google Chrome";v="146"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "x-app-version": "20241129.1",
    "x-client-locale": "zh_CN",
    "x-client-platform": "web",
    "x-client-timezone-offset": "28800",
    "x-client-version": "2.0.0",
}

__all__ = [
    "MODEL_PRO",
    "MODEL_FLASH",
    "MODEL_VISION",
    "MODELS",
    "CAPS_PRO",
    "CAPS_FLASH",
    "CAPS_VISION",
    "CAPS",
    "DEFAULT_HOST",
    "HIF_LEIM_URL",
    "HIF_DLIQ_URL",
    "WASM_PATH",
    "WASM_URL",
    "WASM_META",
    "MAX_CONTINUE",
    "MAX_RETRIES",
    "FETCH_MODELS_ENABLED",
    "MODEL_FETCH_INTERVAL",
    "HIF_REFRESH_INTERVAL",
    "COMMON_HEADERS",
]
