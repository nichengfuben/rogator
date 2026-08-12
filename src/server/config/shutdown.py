from __future__ import annotations

"""跨平台优雅关机：信号、asyncio 异常过滤、遗留任务清理。"""

import asyncio
import logging
import signal
import sys
import threading
from typing import Any, Optional, Set

from echotools.base.logger import get_logger

logger = get_logger("rogator")

_WIN_SHUTDOWN_SOCKET_ERRORS = frozenset({64, 10054, 995, 10038})
_shutdown_lock = threading.Lock()
# Windows CTRL_CLOSE_EVENT handler 等待主线程完成清理的信号
_win_cleanup_done = threading.Event()
# 保留 SetConsoleCtrlHandler 回调引用，防止 GC 回收
_win_ctrl_handler_ref = None


def _request_shutdown_once(state: Any, *, source: str) -> None:
    """中断信号：幂等置位 shutdown_event（重复 Ctrl+C 与第一次等效）。"""
    with _shutdown_lock:
        if state.shutdown_event.is_set():
            return
        logger.info("%s received, shutting down...", source)
        state.shutdown_event.set()


def notify_shutdown_complete() -> None:
    """通知 Windows CTRL_CLOSE_EVENT handler 主线程已完成清理。"""
    _win_cleanup_done.set()


def _flush_logging_handlers() -> None:
    """强制 flush 所有 logging handler，确保日志写入磁盘。"""
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _install_windows_console_handler(state: Any) -> None:
    """注册 Windows SetConsoleCtrlHandler 捕获控制台关闭/注销/关机事件。

    Windows 对 CTRL_CLOSE_EVENT 仅给 5 秒清理时间；handler 在独立线程中运行，
    设置 shutdown_event 后等待主线程完成（最多 4.5s），再 flush 日志。
    """
    global _win_ctrl_handler_ref
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        logger.debug("ctypes unavailable, skipping Windows console handler")
        return

    kernel32 = ctypes.windll.kernel32
    HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    # CTRL_C_EVENT=0, CTRL_BREAK_EVENT=1, CTRL_CLOSE_EVENT=2,
    # CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
    _CONSOLE_CLOSE_EVENTS = frozenset({2, 5, 6})

    def _ctrl_handler(ctrl_type: int) -> bool:
        if ctrl_type not in _CONSOLE_CLOSE_EVENTS:
            return False
        event_names = {2: "CTRL_CLOSE_EVENT", 5: "CTRL_LOGOFF_EVENT", 6: "CTRL_SHUTDOWN_EVENT"}
        name = event_names.get(ctrl_type, f"CTRL_{ctrl_type}")
        try:
            logger.warning("[SHUTDOWN] %s received, initiating graceful shutdown...", name)
        except Exception:
            pass
        _request_shutdown_once(state, source=name)
        # 等待主线程完成 graceful shutdown，最多 4.5 秒（留 0.5s 余量）
        _win_cleanup_done.wait(timeout=4.5)
        _flush_logging_handlers()
        return True

    _win_ctrl_handler_ref = HandlerRoutine(_ctrl_handler)
    result = kernel32.SetConsoleCtrlHandler(_win_ctrl_handler_ref, True)
    if result:
        logger.info("Windows console close handler installed (CTRL_CLOSE/LOGOFF/SHUTDOWN)")
    else:
        logger.warning("SetConsoleCtrlHandler failed, console close will kill process immediately")


def install_signal_handlers(state: Any) -> None:
    """注册 SIGINT/SIGTERM（Unix 用 asyncio；Windows 回退 signal.signal + console handler）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _async_handler(sig_name: str) -> None:
        _request_shutdown_once(state, source=f"Signal {sig_name}")

    def _sync_sigint_handler(_signum: int, _frame: Optional[Any]) -> None:
        _request_shutdown_once(state, source="Interrupt")

    installed_async: Set[int] = set()
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

    _install_windows_console_handler(state)


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
