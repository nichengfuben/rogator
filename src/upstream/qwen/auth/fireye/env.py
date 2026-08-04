from __future__ import annotations

"""静态指纹环境，对齐 dom_polyfill + build_fingerprint 字段。"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List

from upstream.qwen.chat.routes import SEC_CH_UA, USER_AGENT

_WEBGL_VENDOR: Final[str] = "Google Inc. (NVIDIA)"
_WEBGL_RENDERER: Final[str] = (
    "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0)"
)


@dataclass
class FingerprintEnv:
    user_agent: str = USER_AGENT
    platform: str = "Win32"
    language: str = "zh-CN"
    languages: List[str] = field(default_factory=lambda: ["zh-CN", "zh", "en"])
    hardware_concurrency: int = 16
    device_memory: int = 8
    max_touch_points: int = 0
    screen_width: int = 1920
    screen_height: int = 1080
    color_depth: int = 24
    pixel_ratio: float = 1.0
    timezone: str = "Asia/Shanghai"
    timezone_offset: int = -480
    sec_ch_ua: str = SEC_CH_UA
    webgl_vendor: str = _WEBGL_VENDOR
    webgl_renderer: str = _WEBGL_RENDERER
    canvas_seed: str = "rogator-fireye-canvas-v1"

    def canvas_hash(self) -> str:
        blob = "|".join(
            [
                self.canvas_seed,
                self.user_agent,
                self.webgl_vendor,
                self.webgl_renderer,
                f"{self.screen_width}x{self.screen_height}",
            ]
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def signal_bundle(self, fingerprint: str) -> Dict[str, Any]:
        return {
            "fp": fingerprint,
            "ua": self.user_agent,
            "lang": self.language,
            "langs": self.languages,
            "hc": self.hardware_concurrency,
            "dm": self.device_memory,
            "mtp": self.max_touch_points,
            "sw": self.screen_width,
            "sh": self.screen_height,
            "cd": self.color_depth,
            "dpr": self.pixel_ratio,
            "tz": self.timezone,
            "tzo": self.timezone_offset,
            "sec": self.sec_ch_ua,
            "wv": self.webgl_vendor,
            "wr": self.webgl_renderer,
            "cv": self.canvas_hash(),
        }

    def digest(self, fingerprint: str) -> bytes:
        payload = json.dumps(
            self.signal_bundle(fingerprint),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).digest()


def default_env() -> FingerprintEnv:
    return FingerprintEnv()
