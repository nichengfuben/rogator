from __future__ import annotations

"""按 upstream 分桶的 persist 路径。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWN_UPSTREAMS: tuple[str, ...] = ("qwen", "deepseek", "cursor")


def persist_root(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / "persist"


def upstream_dir(upstream: str, root: Path | None = None) -> Path:
    return persist_root(root) / upstream.strip().lower()


def sessions_path(upstream: str, root: Path | None = None) -> Path:
    return upstream_dir(upstream, root) / "sessions.json"


def models_path(upstream: str, root: Path | None = None) -> Path:
    return upstream_dir(upstream, root) / "models.json"


def login_history_path(upstream: str, root: Path | None = None) -> Path:
    return upstream_dir(upstream, root) / "login_history.json"


def unified_sessions_path(root: Path | None = None) -> Path:
    return persist_root(root) / "sessions.json"


def unified_models_path(root: Path | None = None) -> Path:
    return persist_root(root) / "models.json"


def unified_login_history_path(root: Path | None = None) -> Path:
    return persist_root(root) / "login_history.json"
