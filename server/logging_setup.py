from __future__ import annotations

"""Rogator 日志：控制台 + logs/{log_name}-{YYYYMMDD-HHmmss}.log。"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from echotools.logger import configure

from server.config import CONFIG, LOG_DIR

__all__ = ["setup_logging", "resolve_log_file_path"]


def resolve_log_file_path(*, log_name: Optional[str] = None) -> Optional[Path]:
    """生成带 log_name 前缀与时间戳的日志文件路径。"""
    if not CONFIG.log_to_file:
        return None
    name = (log_name or CONFIG.log_name or "rogator").strip() or "rogator"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / f"{name}-{stamp}.log"


def setup_logging(level: Optional[str] = None) -> Optional[Path]:
    log_level = (level or CONFIG.log_level or "INFO").upper()
    log_path = resolve_log_file_path()
    configure(
        level=log_level,
        color=CONFIG.log_color,
        log_file=str(log_path) if log_path is not None else None,
    )
    return log_path
