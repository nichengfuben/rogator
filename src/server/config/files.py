from __future__ import annotations

"""config/ 目录：用户 config.toml、上游 TOML、model_registry 与 template 引导。"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

logger = logging.getLogger("rogator")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = PROJECT_ROOT / "template"
TEMPLATE_NAME = "config.toml"
UPSTREAM_CONFIG_TEMPLATE_NAME = "upstream_config.toml"
UPSTREAM_TEMPLATE_DIR_NAME = "upstream"
USER_CONFIG_NAME = "config.toml"
MODEL_REGISTRY_NAME = "model_registry.jsonl"

USER_CONFIG_DIR = PROJECT_ROOT / "config"
USER_CONFIG_PATH = USER_CONFIG_DIR / USER_CONFIG_NAME
USER_UPSTREAM_DIR = USER_CONFIG_DIR / "upstream"
MODEL_REGISTRY_FILE = USER_CONFIG_DIR / MODEL_REGISTRY_NAME

# 兼容旧路径
USER_CONFIGS_DIR = USER_CONFIG_DIR
LEGACY_CONFIGS_DIR = PROJECT_ROOT / "configs"
LEGACY_ROOT_CONFIG = PROJECT_ROOT / USER_CONFIG_NAME
LEGACY_DIR_CONFIG = USER_CONFIG_PATH
LEGACY_MODEL_REGISTRY = PROJECT_ROOT / "persist" / MODEL_REGISTRY_NAME
LEGACY_UPSTREAM_DEFAULTS_NAME = "defaults.toml"

_UPSTREAM_PER_PLATFORM = frozenset({"deepseek.toml", "qwen.toml"})


def template_config_path() -> Path:
    return TEMPLATE_DIR / TEMPLATE_NAME


def upstream_config_template_path() -> Path:
    return TEMPLATE_DIR / UPSTREAM_CONFIG_TEMPLATE_NAME


def upstream_template_dir() -> Path:
    return TEMPLATE_DIR / UPSTREAM_TEMPLATE_DIR_NAME


def user_config_path() -> Path:
    return USER_CONFIG_PATH


def _loads_toml(text: str) -> Dict[str, Any]:
    return _toml_loader.loads(text)


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


def _move_if_missing(src: Path, dest: Path) -> None:
    if not src.is_file() or dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def migrate_config_layout() -> None:
    """一次性迁移：configs/→config/、根 config.toml、persist/model_registry.jsonl。"""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)

    if LEGACY_CONFIGS_DIR.is_dir():
        for item in LEGACY_CONFIGS_DIR.iterdir():
            if not item.is_file():
                continue
            if item.name in _UPSTREAM_PER_PLATFORM:
                _move_if_missing(item, USER_UPSTREAM_DIR / item.name)
            else:
                _move_if_missing(item, USER_CONFIG_DIR / item.name)
        try:
            LEGACY_CONFIGS_DIR.rmdir()
        except OSError:
            pass
    elif LEGACY_CONFIGS_DIR.is_dir():
        remaining = [p for p in LEGACY_CONFIGS_DIR.iterdir() if p.name != "__pycache__"]
        if not remaining:
            shutil.rmtree(LEGACY_CONFIGS_DIR, ignore_errors=True)

    _move_if_missing(LEGACY_ROOT_CONFIG, USER_CONFIG_PATH)
    _move_if_missing(LEGACY_MODEL_REGISTRY, MODEL_REGISTRY_FILE)

    # 旧版：上游 TOML 直接在 config/ 根目录
    for name in _UPSTREAM_PER_PLATFORM:
        _move_if_missing(USER_CONFIG_DIR / name, USER_UPSTREAM_DIR / name)


def ensure_user_config_file() -> Path:
    """确保 ``config/config.toml`` 存在；可从旧根目录或模板复制。"""
    migrate_config_layout()
    target = user_config_path()
    if target.is_file():
        return target
    if LEGACY_ROOT_CONFIG.is_file():
        shutil.copy2(LEGACY_ROOT_CONFIG, target)
        return target
    template = template_config_path()
    if not template.is_file():
        raise FileNotFoundError(
            f"模板缺失: template/{TEMPLATE_NAME}（无法创建 config/{USER_CONFIG_NAME}）"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, target)
    return target


def warn_if_config_version_mismatch(user_path: Path, log: Any) -> None:
    """``server.version`` 与模板不一致时仅打日志，不修改本地 config。"""
    template = template_config_path()
    if not template.is_file():
        return
    try:
        tpl_ver = read_server_version(template)
        usr_ver = read_server_version(user_path)
    except Exception as exc:
        log.warning("无法读取配置版本号: %s", exc)
        return
    if tpl_ver is None:
        return
    if usr_ver is not None and usr_ver != tpl_ver:
        log.warning(
            "config/config.toml 版本 (%s) 与 template/config.toml (%s) 不一致；"
            "请对照 template 自行更新本地配置（不会自动修改 config/config.toml）",
            usr_ver,
            tpl_ver,
        )
