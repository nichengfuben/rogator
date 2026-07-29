from __future__ import annotations

"""跨平台优雅关机：信号、asyncio 异常过滤、遗留任务清理。"""

import asyncio
import signal
import threading
from typing import Any, Optional

from echotools.logger import get_logger

logger = get_logger("rogator")

_WIN_SHUTDOWN_SOCKET_ERRORS = frozenset({64, 10054, 995, 10038})
_shutdown_lock = threading.Lock()
_shutdown_signal_count = 0


def _force_exit_after_repeat_interrupt() -> None:
    """第二次及以上中断：避免卡在 accept/清理阶段无法退出。"""
    logger.warning("Repeated interrupt during shutdown, forcing exit")
    raise SystemExit(130)


def _request_shutdown_once(state: Any, *, source: str) -> None:
    """首次中断：仅置位 shutdown_event（不提前设 _shutdown_requested，留给 state.shutdown）。"""
    global _shutdown_signal_count
    with _shutdown_lock:
        _shutdown_signal_count += 1
        if _shutdown_signal_count > 1:
            _force_exit_after_repeat_interrupt()
        already_set = state.shutdown_event.is_set()
    if already_set:
        return
    logger.info("%s received, shutting down...", source)
    state.shutdown_event.set()


def install_signal_handlers(state: Any) -> None:
    """注册 SIGINT/SIGTERM（Unix 用 asyncio；Windows 回退 signal.signal）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _async_handler(sig_name: str) -> None:
        _request_shutdown_once(state, source=f"Signal {sig_name}")

    def _sync_sigint_handler(_signum: int, _frame: Optional[Any]) -> None:
        try:
            _request_shutdown_once(state, source="Interrupt")
        except SystemExit:
            raise

    installed_async: set[int] = set()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _async_handler, sig_name)
            installed_async.add(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    sigint = getattr(signal, "SIGINT", None)
    if sigint is not None and sigint not in installed_async:
        try:
            signal.signal(sigint, _sync_sigint_handler)
        except (ValueError, OSError, AttributeError):
            pass


def reset_shutdown_signal_state_for_tests() -> None:
    global _shutdown_signal_count
    with _shutdown_lock:
        _shutdown_signal_count = 0


def install_asyncio_exception_handler(state: Any) -> None:
    """关机阶段忽略 Windows Proactor accept 等良性 socket 错误。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    default_handler = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        if state.is_shutting_down:
            exc = context.get("exception")
            if isinstance(exc, OSError):
                winerr = getattr(exc, "winerror", None)
                if winerr in _WIN_SHUTDOWN_SOCKET_ERRORS or exc.errno in _WIN_SHUTDOWN_SOCKET_ERRORS:
                    logger.debug("Ignoring shutdown socket error: %s", exc)
                    return
            message = context.get("message", "")
            if "Accept failed" in message:
                logger.debug("%s", message)
                return
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)


async def cancel_leftover_tasks() -> None:
    """取消并回收遗留 asyncio 任务。"""
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
