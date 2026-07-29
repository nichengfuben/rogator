from __future__ import annotations

"""根目录 config.toml 与 template/ 路径、首次引导及加载时 overlay（不写回本地 config）。"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = PROJECT_ROOT / "template"
TEMPLATE_NAME = "config.toml"
UPSTREAM_CONFIG_TEMPLATE_NAME = "upstream_config.toml"
UPSTREAM_TEMPLATE_DIR_NAME = "upstream"
USER_CONFIG_NAME = "config.toml"
USER_CONFIG_PATH = PROJECT_ROOT / USER_CONFIG_NAME
USER_CONFIGS_DIR = PROJECT_ROOT / "configs"
LEGACY_DIR_CONFIG = PROJECT_ROOT / "config" / USER_CONFIG_NAME
LEGACY_UPSTREAM_DEFAULTS_NAME = "defaults.toml"


def template_config_path() -> Path:
    return TEMPLATE_DIR / TEMPLATE_NAME


def upstream_config_template_path() -> Path:
    return TEMPLATE_DIR / UPSTREAM_CONFIG_TEMPLATE_NAME


def upstream_template_dir() -> Path:
    return TEMPLATE_DIR / UPSTREAM_TEMPLATE_DIR_NAME


def user_config_path() -> Path:
    return USER_CONFIG_PATH


def _loads_toml(text: str) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        return _toml_loader.loads(text)
    return _toml_loader.loads(text.encode("utf-8"))


def overlay_user_config(
    template_raw: Dict[str, Any],
    user_raw: Dict[str, Any],
) -> Dict[str, Any]:
    """加载策略：以 template 为底，用户 config.toml 覆盖同路径键（仅内存，不改文件）。"""
    merged: Dict[str, Any] = dict(template_raw)
    for key, value in user_raw.items():
        base = merged.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            merged[key] = overlay_user_config(base, value)
        else:
            merged[key] = value
    return merged


def read_server_version(path: Path) -> str | None:
    """读取 ``[server].version``；文件不存在或字段缺失时返回 ``None``。"""
    if not path.is_file():
        return None
    raw = _loads_toml(path.read_text(encoding="utf-8"))
    server = raw.get("server")
    if not isinstance(server, dict):
        return None
    version = server.get("version")
    return str(version) if version is not None else None


def ensure_user_config_file() -> Path:
    """确保根目录 ``config.toml`` 存在；可从 ``config/config.toml`` 或模板复制。"""
    target = user_config_path()
    if target.is_file():
        return target
    if LEGACY_DIR_CONFIG.is_file():
        shutil.copy2(LEGACY_DIR_CONFIG, target)
        return target
    template = template_config_path()
    if not template.is_file():
        raise FileNotFoundError(
            f"模板缺失: template/{TEMPLATE_NAME}（无法创建 {USER_CONFIG_NAME}）"
        )
    shutil.copy2(template, target)
    return target


def warn_if_config_version_mismatch(user_path: Path, logger: Any) -> None:
    """``server.version`` 与模板不一致时仅打日志，不修改本地 config。"""
    template = template_config_path()
    if not template.is_file():
        return
    try:
        tpl_ver = read_server_version(template)
        usr_ver = read_server_version(user_path)
    except Exception as exc:
        logger.warning("无法读取配置版本号: %s", exc)
        return
    if tpl_ver is None:
        return
    if usr_ver is not None and usr_ver != tpl_ver:
        logger.warning(
            "config.toml 版本 (%s) 与 template/config.toml (%s) 不一致；"
            "请对照 template 自行更新本地配置（不会自动修改 config.toml）",
            usr_ver,
            tpl_ver,
        )
