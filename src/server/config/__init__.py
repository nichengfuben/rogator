from __future__ import annotations

from server.config.app_config import CONFIG, AppConfig, LOG_DIR, load_config, resolve_log_path
from server.config.files import (
    LEGACY_DIR_CONFIG,
    PROJECT_ROOT,
    TEMPLATE_DIR,
    USER_CONFIG_PATH,
    ensure_user_config_file,
    overlay_user_config,
    read_server_version,
    template_config_path,
    user_config_path,
    warn_if_config_version_mismatch,
)
from server.config.logging_setup import (
    resolve_access_log,
    resolve_log_file_path,
    setup_logging,
    shutdown_logging,
)
from server.config.shutdown import (
    cancel_leftover_tasks,
    install_asyncio_exception_handler,
    install_signal_handlers,
    reset_shutdown_signal_state_for_tests,
)

__all__ = [
    "CONFIG",
    "AppConfig",
    "LOG_DIR",
    "LEGACY_DIR_CONFIG",
    "PROJECT_ROOT",
    "TEMPLATE_DIR",
    "USER_CONFIG_PATH",
    "ensure_user_config_file",
    "load_config",
    "overlay_user_config",
    "read_server_version",
    "resolve_access_log",
    "resolve_log_file_path",
    "resolve_log_path",
    "setup_logging",
    "shutdown_logging",
    "cancel_leftover_tasks",
    "install_asyncio_exception_handler",
    "install_signal_handlers",
    "reset_shutdown_signal_state_for_tests",
    "template_config_path",
    "user_config_path",
    "warn_if_config_version_mismatch",
]
