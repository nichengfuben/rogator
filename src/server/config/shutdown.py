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


def _request_shutdown_once(state: Any, *, source: str) -> None:
    """中断信号：幂等置位 shutdown_event（多次 Ctrl+C 与第一次等效）。"""
    with _shutdown_lock:
        if state.shutdown_event.is_set():
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
        _request_shutdown_once(state, source="Interrupt")

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
    """测试钩子（保留 API；信号处理已幂等，无需重置状态）。"""
    return


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


_LEFTOVER_TASK_CANCEL_TIMEOUT = 5.0


async def cancel_leftover_tasks(*, timeout: float = _LEFTOVER_TASK_CANCEL_TIMEOUT) -> None:
    """取消并回收遗留 asyncio 任务（有超时，避免 gather 永久阻塞）。"""
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in tasks:
        task.cancel()
    if not tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        remaining = sum(1 for t in tasks if not t.done())
        logger.warning(
            "Shutdown: %d leftover task(s) still running after %.1fs cancel wait",
            remaining,
            timeout,
        )
