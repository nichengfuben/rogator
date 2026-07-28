from __future__ import annotations

"""fncall 注入包装：由 echotools inject 落盘 logs/prompts/{uuid7}.txt。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from echotools.fncall.prompt.inject import inject_fncall as _echotools_inject
from echotools.exec.protocol.base import ToolProtocol
from echotools.logger import get_logger

from server.config import CONFIG, LOG_DIR, PROJECT_ROOT

__all__ = ["inject_fncall_for_request", "prompt_dump_dir"]

logger = get_logger("rogator")

_PROMPTS_SUBDIR = "prompts"


def prompt_dump_dir() -> Path:
    return LOG_DIR / _PROMPTS_SUBDIR


def _get_dump_dir() -> Optional[str]:
    """record_prompt 或 print_prompt 为 true 时返回落盘目录。"""
    if CONFIG.record_prompt or CONFIG.print_prompt:
        return str(prompt_dump_dir())
    return None


def inject_fncall_for_request(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    protocol: ToolProtocol,
    *,
    req_id: str,
    api: str,
    model: str,
    lang: str = "zh",
    user_system_prompt: str = "",
    loop_detection_threshold: int = 3,
    protocol_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    dump_dir = _get_dump_dir()
    injected = _echotools_inject(
        messages=messages,
        tools=tools,
        protocol=protocol,
        lang=lang,
        user_system_prompt=user_system_prompt,
        loop_detection_threshold=loop_detection_threshold,
        dump_prompt=dump_dir is not None,
        dump_dir=dump_dir,
        protocol_options=protocol_options,
    )
    prompt = injected[0]["content"]
    logger.info(
        "inject prompt api=%s req_id=%s model=%s chars=%d tools=%d dump_dir=%s",
        api,
        req_id,
        model,
        len(prompt),
        len(tools or []),
        dump_dir,
    )
    return injected
