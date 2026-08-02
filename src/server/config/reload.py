from __future__ import annotations

"""配置热重载：mtime 轮询 + 原子 swap + 运行时 apply。"""

import asyncio
import logging
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from echotools.base.logger import get_logger

from server.config.app_config import CONFIG, AppConfig, get_config, load_config
from server.config.files import USER_CONFIG_DIR, USER_UPSTREAM_DIR, user_config_path

if TYPE_CHECKING:
    from state import AppState

logger = get_logger("rogator")

# 变更后仅告警、需重启才生效的字段
RESTART_REQUIRED: frozenset[str] = frozenset({
    "host",
    "port",
    "upstream_enabled",
    "client_max_body_bytes",
    "access_log",
    "log_to_file",
    "log_name",
    "log_color",
})

_POLL_INTERVAL_S: float = 1.0
_DEBOUNCE_S: float = 0.5


def watched_config_paths() -> List[Path]:
    """参与合并的用户侧配置路径（存在与否均可，mtime 采集时跳过缺失）。"""
    paths: List[Path] = [user_config_path()]
    if USER_CONFIG_DIR.is_dir():
        paths.extend(sorted(USER_CONFIG_DIR.glob("*.toml")))
    if USER_UPSTREAM_DIR.is_dir():
        paths.extend(sorted(USER_UPSTREAM_DIR.glob("*.toml")))
    return paths


def _collect_mtimes(paths: Iterable[Path]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for path in paths:
        try:
            if path.is_file():
                out[str(path.resolve())] = path.stat().st_mtime
        except OSError:
            continue
    return out


def _diff_fields(old: AppConfig, new: AppConfig) -> Tuple[List[str], List[str]]:
    hot: List[str] = []
    restart: List[str] = []
    for f in fields(AppConfig):
        name = f.name
        if getattr(old, name) == getattr(new, name):
            continue
        if name in RESTART_REQUIRED:
            restart.append(name)
        else:
            hot.append(name)
    return hot, restart


def apply_session_pool_targets(state: "AppState", target: int) -> None:
    for client in state._clients.values():
        if hasattr(client, "_prelogin_target"):
            client._prelogin_target = target


def apply_runtime_config(old: AppConfig, new: AppConfig, state: Optional["AppState"]) -> None:
    """把可热更字段同步到运行时缓存（splitter / clients / 队列 / 日志级别）。"""
    import state as state_mod

    state_mod.MAX_QUEUE_SIZE = new.max_queue_size
    state_mod.QWEN_SEND_MAX_CHARS = new.qwen_send_max_chars
    state_mod.MODEL_CONTEXT_LENGTH = new.model_context_length

    if state is not None:
        splitter = getattr(state, "splitter", None)
        if splitter is not None:
            splitter.max_chars = new.qwen_send_max_chars
            splitter.send_full_prompt = new.send_full_prompt
        apply_session_pool_targets(state, new.prelogin)
        for client in state._clients.values():
            if hasattr(client, "_login_interval"):
                client._login_interval = new.login_interval
        wake = getattr(state.scheduler, "wake_for_config", None)
        if callable(wake):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(wake())
            except RuntimeError:
                pass

    if old.log_level != new.log_level:
        level = getattr(logging, new.log_level.upper(), None)
        if isinstance(level, int):
            logging.getLogger().setLevel(level)
            logging.getLogger("rogator").setLevel(level)


def reload_config(
    *,
    state: Optional["AppState"] = None,
    path: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> bool:
    """重新加载配置；成功 swap 返回 True，失败保留旧配置返回 False。"""
    try:
        new_cfg = load_config(path, template_path=template_path)
    except Exception as exc:
        logger.error("配置热重载失败，保留旧配置: %s", exc)
        return False

    old = get_config()
    if new_cfg == old:
        logger.debug("配置热重载: 无变化")
        return True

    hot, restart = _diff_fields(old, new_cfg)
    CONFIG.swap(new_cfg)
    apply_runtime_config(old, new_cfg, state)

    if hot:
        logger.info("配置已热重载: %s", ", ".join(hot))
    if restart:
        logger.warning(
            "以下配置已加载但需重启进程才生效: %s",
            ", ".join(restart),
        )
    if not hot and not restart:
        logger.info("配置已热重载")
    return True


async def config_watch_loop(
    state: "AppState",
    *,
    poll_interval: float = _POLL_INTERVAL_S,
    debounce: float = _DEBOUNCE_S,
) -> None:
    """轮询配置文件 mtime；变更后 debounce 再 reload。"""
    mtimes = _collect_mtimes(watched_config_paths())
    logger.info(
        "配置热重载已启用: poll=%.1fs debounce=%.1fs files=%d",
        poll_interval,
        debounce,
        len(mtimes),
    )
    while not state.is_shutting_down:
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=poll_interval)
            break
        except asyncio.TimeoutError:
            pass
        if state.is_shutting_down:
            break
        current = _collect_mtimes(watched_config_paths())
        if current == mtimes:
            continue
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=debounce)
            break
        except asyncio.TimeoutError:
            pass
        if state.is_shutting_down:
            break
        # debounce 后再采一次，避免编辑器半写
        current = _collect_mtimes(watched_config_paths())
        reload_config(state=state)
        mtimes = current


def start_config_watcher(state: "AppState") -> asyncio.Task:
    """启动配置监听后台任务。"""
    return asyncio.create_task(config_watch_loop(state), name="config_watch")
