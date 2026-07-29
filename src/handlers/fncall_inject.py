from __future__ import annotations

"""fncall 注入包装：落盘 logs/prompts/{req_id}.txt（与 responses/{req_id}.txt 对齐）。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from echotools.fncall.prompt.inject import inject_fncall as _echotools_inject
from echotools.exec.protocol.base import ToolProtocol
from echotools.logger import get_logger

from server.config import CONFIG, LOG_DIR

__all__ = ["inject_fncall_for_request", "prompt_dump_dir"]

logger = get_logger("rogator")

_PROMPTS_SUBDIR = "prompts"


def prompt_dump_dir() -> Path:
    return LOG_DIR / _PROMPTS_SUBDIR


def _should_dump_prompt() -> bool:
    return bool(CONFIG.record_prompt or CONFIG.print_prompt)


def _dump_prompt(prompt: str, req_id: str) -> None:
    dump_dir = prompt_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"{req_id}.txt"
    path.write_text(prompt, encoding="utf-8")
    logger.debug("prompt 已写入 %s", path)


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
    injected = _echotools_inject(
        messages=messages,
        tools=tools,
        protocol=protocol,
        lang=lang,
        user_system_prompt=user_system_prompt,
        loop_detection_threshold=loop_detection_threshold,
        dump_prompt=False,
        dump_dir=None,
        protocol_options=protocol_options,
    )
    prompt = injected[0]["content"]
    if _should_dump_prompt():
        _dump_prompt(prompt, req_id)
    logger.info(
        "inject prompt api=%s req_id=%s model=%s chars=%d tools=%d dump_dir=%s",
        api,
        req_id,
        model,
        len(prompt),
        len(tools or []),
        str(prompt_dump_dir()) if _should_dump_prompt() else None,
    )
    return injected
