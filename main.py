#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rogator 多上游 AI 适配服务器入口（OpenAI / Anthropic 兼容 API）。"""

import faulthandler

faulthandler.enable()

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for _entry in (_root / "src", _root):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)
import asyncio
import contextlib
import os
import time
from typing import Optional

from aiohttp import web

# echotools 日志：控制台 + logs/rogator.log
from echotools.base.logger import get_logger

import path_setup  # noqa: F401
from server.config import CONFIG, load_config
from server.config.files import user_config_path, warn_if_config_version_mismatch
from server.config.logging_setup import (
    resolve_access_log,
    setup_logging,
    shutdown_logging,
)
from server.config.reload import apply_session_pool_targets, start_config_watcher
from server.config.shutdown import (
    cancel_leftover_tasks,
    install_asyncio_exception_handler,
    install_signal_handlers,
    notify_shutdown_complete,
)
from server.config.startup_port import ensure_listen_port

_LOG_FILE = setup_logging()
logger = get_logger("rogator")
warn_if_config_version_mismatch(user_config_path(), logger)


from core.registry import load_upstreams
from core.session.store import CLEANUP_INTERVAL, valid_session_count
from handlers import get_state, setup_routes
from handlers.shared.fncall_inject import prompt_dump_dir
from server.records.response_record import response_dump_dir
from server.records.sse_record import sse_dump_dir
from state import AppState

load_upstreams()

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
logger.info(
    "sse record=%s dir=%s pattern=logs/sse/{req_id}.sse (upstream raw stream, pre-parse)",
    CONFIG.record_sse,
    sse_dump_dir(),
)


APP_NAME: str = "Rogator"
APP_VERSION: str = "2.3.5"
APP_DESCRIPTION: str = "多上游 AI 适配服务器（Qwen / DeepSeek 等）"
SHUTDOWN_CANCEL_GRACE: float = 0.3

# 显示参数
BANNER_WIDTH: int = 70


def _validate_config(port: int, prelogin_count: int) -> None:
    """验证配置参数。"""
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")
    if prelogin_count < 0:
        raise ValueError(f"Invalid prelogin count: {prelogin_count}")


def _install_signal_handlers(state: AppState) -> None:
    install_signal_handlers(state)


def _print_startup_info(
    state: AppState, host: str, port: int, prelogin_count: int
) -> None:
    """打印启动信息横幅。"""
    logger.info("=" * BANNER_WIDTH)
    logger.info("%s - %s", APP_NAME, APP_DESCRIPTION)
    logger.info("  Version     : %s", APP_VERSION)
    logger.info("  Listen      : %s:%d", host, port)
    logger.info("  Model       : %s", state.model)
    logger.info("  Protocol    : %s", state.protocol.id)
    session_parts = []
    for name, client in state._clients.items():
        sessions = getattr(client, "_sessions", None)
        if sessions is not None:
            session_parts.append(f"{name}={valid_session_count(sessions)}")
    logger.info("  Upstreams   : %s", ", ".join(state._registry.names()))
    logger.info("  Sessions    : %s (max 12h)", ", ".join(session_parts) or "none")
    logger.info("  Models      : %d", len(state._models))
    logger.info(
        "  Max body    : %d bytes (%.1f MiB)",
        CONFIG.client_max_body_bytes,
        CONFIG.client_max_body_bytes / (1024 * 1024),
    )
    logger.info(
        "  Send full   : %s (no truncate / no OSS prefix)", CONFIG.send_full_prompt
    )
    logger.info("  Access log  : %s", CONFIG.access_log)
    logger.info("  Models refresh: every %ds", int(CONFIG.models_refresh_interval))
    logger.info(
        "  Cleanup     : background session pool + %ds maintenance",
        int(CLEANUP_INTERVAL),
    )
    logger.info("  ID Format   : gen-{timestamp}-{random12}")
    logger.info("=" * BANNER_WIDTH)


async def _shutdown_step(
    label: str,
    coro,
    *,
    timeout: float,
    deadline: float,
) -> None:
    """单步关机；受整段硬 deadline 与各步 timeout 双重约束。"""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        logger.warning("Shutdown: hard deadline reached, skipping %s", label)
        return
    logger.info("Shutdown: %s...", label)
    try:
        await asyncio.wait_for(coro, timeout=min(timeout, remaining))
    except asyncio.TimeoutError:
        logger.warning("Shutdown: %s timed out (%.1fs cap)", label, timeout)
    except Exception as exc:
        logger.warning("Shutdown: %s failed: %s", label, exc)


async def _hard_exit_watchdog(deadline: float) -> None:
    delay = deadline - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    logger.error(
        "Shutdown: exceeded hard exit timeout (%.1fs), forcing exit",
        CONFIG.shutdown_hard_exit_timeout,
    )
    os._exit(1)


async def _graceful_shutdown(
    state: AppState, runner: web.AppRunner, site: Optional[web.TCPSite]
) -> None:
    deadline = time.monotonic() + CONFIG.shutdown_hard_exit_timeout
    watchdog = asyncio.create_task(_hard_exit_watchdog(deadline))
    try:
        await _shutdown_step(
            "draining active requests",
            state.shutdown(),
            timeout=CONFIG.shutdown_total_timeout,
            deadline=deadline,
        )
        if site is not None:
            await _shutdown_step(
                "stopping HTTP site",
                site.stop(),
                timeout=5.0,
                deadline=deadline,
            )
        await _shutdown_step(
            "cleaning up HTTP runner",
            runner.cleanup(),
            timeout=5.0,
            deadline=deadline,
        )
        await _shutdown_step(
            "cancelling leftover tasks",
            cancel_leftover_tasks(),
            timeout=6.0,
            deadline=deadline,
        )
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
    logger.info("Server stopped")
    notify_shutdown_complete()


async def _run_server(
    app: web.Application, state: AppState, host: str, port: int
) -> None:
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
        logger.info(
            "Listening on http://%s:%d (session login in background)", host, port
        )
        _install_signal_handlers(state)
        install_asyncio_exception_handler(state)
        state.start_background_tasks()
        state._bg_tasks.append(start_config_watcher(state))
        while not state.shutdown_event.is_set():
            try:
                await asyncio.wait_for(state.shutdown_event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
    except KeyboardInterrupt:
        state.shutdown_event.set()
    except SystemExit:
        state.shutdown_event.set()
        raise
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=True)
        raise
    finally:
        state.shutdown_event.set()
        await _graceful_shutdown(state, runner, site)


async def main_async() -> None:
    """服务器异步主入口（配置来自 config.toml + template/config.toml）。"""
    from upstream.qwen.media.proxy_toggle import get_proxy_toggle
    await get_proxy_toggle().initialize()
    cfg = load_config()
    CONFIG.swap(cfg)
    port = cfg.port
    host = cfg.host
    prelogin_count = cfg.prelogin
    _validate_config(port, prelogin_count)

    ensure_listen_port(host, port, force_kill=cfg.startup_force_kill_port)

    app = web.Application(client_max_size=cfg.client_max_body_bytes)
    setup_routes(app)
    state = get_state()
    apply_session_pool_targets(state, prelogin_count)
    for name, client in state._clients.items():
        cleanup = getattr(client, "cleanup_expired_sessions", None)
        if callable(cleanup):
            removed = cleanup()
            if removed:
                logger.info(
                    "Startup cleanup [%s]: removed %d expired/invalid session(s)",
                    name,
                    len(removed),
                )

    _print_startup_info(state, host, port, prelogin_count)
    await _run_server(app, state, host, port)


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
