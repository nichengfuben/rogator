#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rogator - Qwen 长文本处理服务器 (entry point)"""

from __future__ import annotations

import argparse
import asyncio
import signal
import socket
import sys
from typing import Optional

from aiohttp import web

# ============================================================
# echotools 日志配置
# ============================================================
from echotools.logger import configure, get_logger

configure(
    level="DEBUG",
    color=True,
    show_time=True,
    show_level=True,
    show_name=True,
    time_format="%Y-%m-%d %H:%M:%S",
)
logger = get_logger("rogator")

# ============================================================
# 从模块导入
# ============================================================
from handlers import get_state, setup_routes
from server.config import CONFIG, load_config
from server.session_store import CLEANUP_INTERVAL
from state import AppState

# ============================================================
# 全局常量
# ============================================================

PORT: int = CONFIG.port
HOST: str = CONFIG.host
PRELOGIN_ACCOUNT_COUNT: int = CONFIG.prelogin
PRELOGIN_TIMEOUT: float = CONFIG.prelogin_timeout

APP_NAME: str = "Rogator"
APP_VERSION: str = "2.1.0"
APP_DESCRIPTION: str = "Qwen 长文本处理适配服务器"
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 10.0
SHUTDOWN_TOTAL_TIMEOUT: float = 15.0

# 显示参数
BANNER_WIDTH: int = 70


# ============================================================
# CLI 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=APP_DESCRIPTION,
    )
    parser.add_argument(
        "--port", type=int, default=PORT,
        help=f"服务器监听端口 (默认: {PORT})",
    )
    parser.add_argument(
        "--host", type=str, default=HOST,
        help=f"服务器监听地址 (默认: {HOST})",
    )
    parser.add_argument(
        "--prelogin", type=int, default=PRELOGIN_ACCOUNT_COUNT,
        help=f"预登录账户数 (默认: {PRELOGIN_ACCOUNT_COUNT})",
    )
    parser.add_argument(
        "--log-level", type=str, default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: DEBUG)",
    )
    return parser.parse_args()


# ============================================================
# 启动辅助函数
# ============================================================

def _check_port_in_use(port: int) -> bool:
    """检查端口是否被占用。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return False
    except OSError:
        return True


def _validate_config(port: int, prelogin_count: int) -> None:
    """验证配置参数。"""
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")
    if prelogin_count < 0:
        raise ValueError(f"Invalid prelogin count: {prelogin_count}")


def _install_signal_handlers(state: AppState) -> None:
    """注册系统信号处理器。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _handle(sig_name: str) -> None:
        logger.info("Signal %s received", sig_name)
        state.shutdown_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig:
            try:
                loop.add_signal_handler(sig, _handle, sig_name)
            except (NotImplementedError, RuntimeError):
                pass


async def _cancel_leftover_tasks() -> None:
    """取消所有遗留的 asyncio 任务。"""
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ============================================================
# 启动流程函数
# ============================================================

def _print_startup_info(state: AppState, port: int, prelogin_count: int) -> None:
    """打印启动信息横幅。"""
    logger.info("=" * BANNER_WIDTH)
    logger.info("%s - Qwen Server", APP_NAME)
    logger.info("  Version     : %s", APP_VERSION)
    logger.info("  Port        : %d", port)
    logger.info("  Model       : %s", state.model)
    logger.info("  Protocol    : %s", state.protocol.id)
    logger.info("  Sessions    : %d (max 12h)", state.client.session_count)
    logger.info("  Models      : %d", len(state._models))
    logger.info("  Cleanup     : startup + background (%ds)", int(CLEANUP_INTERVAL))
    logger.info("  ID Format   : gen-{timestamp}-{random12}")
    logger.info("=" * BANNER_WIDTH)


async def _prelogin_accounts(state: AppState, count: int) -> None:
    """预登录账户并刷新模型列表。"""
    logger.info("Prelogin %d accounts...", count)
    try:
        await asyncio.wait_for(
            state.client.prelogin_accounts(count),
            timeout=PRELOGIN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Prelogin timed out after %ds, continuing anyway",
            int(PRELOGIN_TIMEOUT),
        )
    except Exception as e:
        logger.error("Prelogin failed: %s", e)

    if state.client.session_count > 0:
        await state.refresh_models()
        # 随机打乱 session 顺序
        import random
        random.shuffle(state.client._sessions)


async def _run_server(app: web.Application, state: AppState, port: int) -> None:
    """启动 web 服务器并等待关机信号。"""
    runner = web.AppRunner(app)
    site: Optional[web.TCPSite] = None
    try:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        _install_signal_handlers(state)
        while not state.shutdown_event.is_set():
            try:
                await asyncio.wait_for(state.shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
    except KeyboardInterrupt:
        state.shutdown_event.set()
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
        raise
    finally:
        if site:
            try:
                await site.stop()
            except Exception:
                pass
        try:
            await asyncio.wait_for(state.shutdown(), timeout=SHUTDOWN_TOTAL_TIMEOUT)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            await runner.cleanup()
        except Exception:
            pass
        await _cancel_leftover_tasks()
        logger.info("Server stopped")


# ============================================================
# 异步入口
# ============================================================

async def main_async(
    port: int = PORT,
    host: str = HOST,
    prelogin_count: int | None = None,
) -> None:
    """服务器异步主入口。"""
    cfg = load_config()
    if prelogin_count is None:
        prelogin_count = cfg.prelogin
    _validate_config(port, prelogin_count)

    if _check_port_in_use(port):
        logger.error("Port %d already in use!", port)
        sys.exit(1)

    app = web.Application()
    setup_routes(app)
    state = get_state()
    state.client._prelogin_target = prelogin_count

    # 启动时立即清理过期/失效 session 并落盘
    removed = state.client.cleanup_expired_sessions()
    if removed:
        logger.info("Startup cleanup: removed %d expired/invalid session(s)", len(removed))

    await _prelogin_accounts(state, prelogin_count)
    state.start_background_tasks()
    _print_startup_info(state, port, prelogin_count)
    await _run_server(app, state, port)


# ============================================================
# 同步入口
# ============================================================

def main() -> None:
    """服务器主入口。"""
    args = parse_args()
    try:
        asyncio.run(main_async(
            port=args.port,
            host=args.host,
            prelogin_count=args.prelogin,
        ))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
