from __future__ import annotations

"""Rogator 日志：控制台 + logs/{log_name}-{YYYYMMDD-HHmmss}.log。"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from echotools.logger import configure

from server.config.app_config import CONFIG, LOG_DIR

__all__ = ["setup_logging", "resolve_log_file_path", "shutdown_logging", "resolve_access_log"]


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


def _wire_access_logger() -> logging.Logger:
    """让 aiohttp.access 走 root handler，输出与 rogator 主日志一致的格式。"""
    access = logging.getLogger("aiohttp.access")
    access.handlers.clear()
    access.propagate = True
    if access.level in (logging.NOTSET, 0):
        access.setLevel(logging.INFO)
    return access


def resolve_access_log(enabled: bool) -> Optional[logging.Logger]:
    """与 provider-core 一致：开启时显式传入 ``aiohttp.access`` logger，关闭时 ``None``。"""
    if not enabled:
        return None
    return _wire_access_logger()


def _close_handler(handler: logging.Handler) -> None:
    try:
        handler.acquire()
        try:
            handler.flush()
        finally:
            handler.release()
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass


def _detach_all_handlers() -> list[logging.Handler]:
    """从所有 logger 上移除 handler（公开 API，兼容 py3.8+）。"""
    seen: set[int] = set()
    handlers: list[logging.Handler] = []

    def take(logger: logging.Logger) -> None:
        for handler in logger.handlers[:]:
            hid = id(handler)
            if hid not in seen:
                seen.add(hid)
                handlers.append(handler)
            logger.removeHandler(handler)

    take(logging.getLogger())
    manager = logging.Logger.manager
    for logger_obj in list(manager.loggerDict.values()):
        if isinstance(logger_obj, logging.Logger):
            take(logger_obj)
    return handlers


def shutdown_logging() -> None:
    """进程退出前关闭日志 handler，避免 atexit 阶段 KeyboardInterrupt 噪音。"""
    handlers = _detach_all_handlers()
    handler_list = getattr(logging, "_handlerList", None)
    lock = getattr(logging, "_lock", None)
    if handler_list is not None and lock is not None:
        with lock:
            handler_list.clear()
    for handler in reversed(handlers):
        _close_handler(handler)
