from __future__ import annotations

"""模型 entml 思考协议映射表（persist/model_entml_thinking.jsonl）。"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    normalize_thinking_mode,
    resolve_thinking_injection,
)

logger = logging.getLogger("rogator")

_MODEL_MAP_FILE = Path(__file__).resolve().parent.parent / "persist" / "model_entml_thinking.jsonl"
_DEFAULT_ENTML = True


def _parse_line(line: str) -> Optional[Tuple[str, bool]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if ":" not in line:
        return None
    model_id, _, flag = line.partition(":")
    model_id = model_id.strip()
    flag = flag.strip().lower()
    if not model_id:
        return None
    if flag in ("true", "1", "yes"):
        return model_id, True
    if flag in ("false", "0", "no"):
        return model_id, False
    return None


def load_model_entml_map(path: Path | None = None) -> Dict[str, bool]:
    """加载 modelid:bool 映射；缺失文件或未知模型默认 True。"""
    p = path or _MODEL_MAP_FILE
    mapping: Dict[str, bool] = {}
    if not p.exists():
        logger.warning("Model entml map not found: %s", p)
        return mapping
    for line in p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed:
            model_id, uses_entml = parsed
            mapping[model_id] = uses_entml
    return mapping


_MODEL_ENTML_MAP: Dict[str, bool] = load_model_entml_map()


def uses_entml_thinking(model: str) -> bool:
    """True=用 entml 传 thinking（上游 Fast）；False=上游原生思考。"""
    return _MODEL_ENTML_MAP.get(model, _DEFAULT_ENTML)


def resolve_qwen_thinking(
    model: str,
    request_thinking_level: Optional[str],
) -> Tuple[bool, str, bool]:
    """返回 (qwen_thinking_enabled, qwen_thinking_mode, use_entml_protocol)。

    - entml 模型：上游永远 Fast，thinking 由 echotools `<thinking_behavior>` 注入。
    - 非 entml 模型：thinking_level 非 none 时上游开思考；none 时 Fast。
    """
    level = normalize_thinking_level(request_thinking_level)
    if level is None and request_thinking_level is not None:
        legacy = normalize_thinking_mode(request_thinking_level)
        if legacy == "off":
            level = "none"
        elif legacy == "on":
            level = "medium"
        elif legacy == "auto":
            level = "auto"
    level = level or "none"
    if uses_entml_thinking(model):
        return False, "Fast", True

    if level == "none" or resolve_thinking_injection({"thinking_level": level}) is None:
        return False, "Fast", False
    return True, "Thinking", False
