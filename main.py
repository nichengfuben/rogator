#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rogator - Qwen 长文本处理服务器 (entry point)"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for _entry in (_root / "src", _root):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)
import path_setup  # noqa: F401

import asyncio
import socket
from typing import Optional

from aiohttp import web

from server.config import CONFIG, load_config
from server.config.files import user_config_path, warn_if_config_version_mismatch
from server.config.logging_setup import setup_logging, shutdown_logging, resolve_access_log
from server.config.shutdown import (
    cancel_leftover_tasks,
    install_asyncio_exception_handler,
    install_signal_handlers,
)

# ============================================================
# echotools 日志：控制台 + logs/rogator.log
# ============================================================
from echotools.logger import get_logger

_LOG_FILE = setup_logging()
logger = get_logger("rogator")
warn_if_config_version_mismatch(user_config_path(), logger)

# ============================================================
# 从模块导入
# ============================================================
from handlers import get_state, setup_routes
from handlers.fncall_inject import prompt_dump_dir
from server.records.response_record import response_dump_dir
from server.client.session_store import CLEANUP_INTERVAL
from state import AppState

if _LOG_FILE is not None:
    logger.info("file logging enabled path=%s", _LOG_FILE)
logger.info(
    "prompt record=%s print=%s dir=%s pattern=logs/prompts/{req_id}.txt",
    CONFIG.record_prompt,
    CONFIG.print_prompt,
    prompt_dump_dir(),
)
logger.info(
    "response record=%s dir=%s pattern=logs/responses/{req_id}.txt (upstream think+answer, pre-entml)",
    CONFIG.record_response,
    response_dump_dir(),
)

# ============================================================
# 全局常量
# ============================================================

APP_NAME: str = "Rogator"
APP_VERSION: str = "2.2.1"
APP_DESCRIPTION: str = "Qwen 长文本处理适配服务器"
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 10.0
SHUTDOWN_TOTAL_TIMEOUT: float = 15.0

# 显示参数
BANNER_WIDTH: int = 70


# ============================================================
# 启动辅助函数
# ============================================================

def _check_port_in_use(host: str, port: int) -> bool:
    """检查端口是否被占用。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
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
    install_signal_handlers(state)


async def _cancel_leftover_tasks() -> None:
    await cancel_leftover_tasks()


# ============================================================
# 启动流程函数
# ============================================================

def _print_startup_info(state: AppState, host: str, port: int, prelogin_count: int) -> None:
    """打印启动信息横幅。"""
    logger.info("=" * BANNER_WIDTH)
    logger.info("%s - Qwen Server", APP_NAME)
    logger.info("  Version     : %s", APP_VERSION)
    logger.info("  Listen      : %s:%d", host, port)
    logger.info("  Model       : %s", state.model)
    logger.info("  Protocol    : %s", state.protocol.id)
    logger.info("  Sessions    : %d (max 12h)", state.client.session_count)
    logger.info("  Models      : %d", len(state._models))
    logger.info("  Max body    : %d bytes (%.1f MiB)", CONFIG.client_max_body_bytes, CONFIG.client_max_body_bytes / (1024 * 1024))
    logger.info("  Send full   : %s (no truncate / no OSS prefix)", CONFIG.send_full_prompt)
    logger.info("  Access log  : %s", CONFIG.access_log)
    logger.info("  Cleanup     : startup + background (%ds, auto prelogin)", int(CLEANUP_INTERVAL))
    logger.info("  ID Format   : gen-{timestamp}-{random12}")
    logger.info("=" * BANNER_WIDTH)


async def _prelogin_accounts(state: AppState, count: int) -> None:
    """预登录账户并刷新模型列表。"""
    logger.info("Prelogin %d accounts...", count)
    try:
        await asyncio.wait_for(
            state.client.prelogin_accounts(count),
            timeout=CONFIG.prelogin_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Prelogin timed out after %ds, continuing anyway",
            int(CONFIG.prelogin_timeout),
        )
    except Exception as e:
        logger.error("Prelogin failed: %s", e)

    if state.client.session_count > 0:
        await state.refresh_models()
        # 随机打乱 session 顺序
        import random
        random.shuffle(state.client._sessions)


async def _run_server(app: web.Application, state: AppState, host: str, port: int) -> None:
    """启动 web 服务器并等待关机信号。"""
    runner = web.AppRunner(
        app,
        access_log=resolve_access_log(CONFIG.access_log),
    )
    site: Optional[web.TCPSite] = None
    try:
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        _install_signal_handlers(state)
        install_asyncio_exception_handler(state)
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
        try:
            await asyncio.wait_for(state.shutdown(), timeout=SHUTDOWN_TOTAL_TIMEOUT)
        except (asyncio.TimeoutError, Exception):
            pass
        if site:
            try:
                await site.stop()
            except OSError as exc:
                logger.debug("site.stop during shutdown: %s", exc)
            except Exception:
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

async def main_async() -> None:
    """服务器异步主入口（配置来自 config.toml + template/config.toml）。"""
    cfg = load_config()
    port = cfg.port
    host = cfg.host
    prelogin_count = cfg.prelogin
    _validate_config(port, prelogin_count)

    if _check_port_in_use(host, port):
        logger.error("Port %s:%d already in use!", host, port)
        sys.exit(1)

    app = web.Application(client_max_size=cfg.client_max_body_bytes)
    setup_routes(app)
    state = get_state()
    state.client._prelogin_target = prelogin_count

    # 启动时立即清理过期/失效 session 并落盘
    removed = state.client.cleanup_expired_sessions()
    if removed:
        logger.info("Startup cleanup: removed %d expired/invalid session(s)", len(removed))

    await _prelogin_accounts(state, prelogin_count)
    state.start_background_tasks()
    _print_startup_info(state, host, port, prelogin_count)
    await _run_server(app, state, host, port)


# ============================================================
# 同步入口
# ============================================================

def main() -> None:
    """服务器主入口。"""
    exit_code = 0
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Shutdown requested (keyboard interrupt)")
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
        exit_code = 1
    finally:
        shutdown_logging()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
