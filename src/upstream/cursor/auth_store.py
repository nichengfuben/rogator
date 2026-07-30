from __future__ import annotations

"""Cursor 凭证：``persist/cursor/auth.toml``（Star Cursor 拉号后写入）。"""

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from core.persist.paths import upstream_dir

UPSTREAM = "cursor"
AUTH_FILENAME = "auth.toml"

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader


def cursor_dir(root: Path | None = None) -> Path:
    return upstream_dir(UPSTREAM, root)


def auth_path(root: Path | None = None) -> Path:
    return cursor_dir(root) / AUTH_FILENAME


def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        return _toml_loader.loads(text)
    return _toml_loader.loads(text.encode("utf-8"))


def _write_toml(path: Path, data: Dict[str, Any]) -> None:
    lines: list[str] = []
    for key, val in data.items():
        if val is None:
            continue
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key} = {val}")
        else:
            lines.append(f"{key} = {json.dumps(str(val), ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_auth(root: Path | None = None) -> Dict[str, Any]:
    return _read_toml(auth_path(root))


def get_access_token(root: Path | None = None) -> Optional[str]:
    auth = read_auth(root)
    return auth.get("access_token") or auth.get("accessToken")


def get_token_bundle(root: Path | None = None) -> Dict[str, str]:
    """返回 Agent 流所需的 token 与 machine id 字段。"""
    auth = read_auth(root)
    access = auth.get("access_token") or auth.get("accessToken") or ""
    machine_id = auth.get("machine_id") or auth.get("machineId") or str(uuid.uuid4())
    mac_machine_id = auth.get("mac_machine_id") or auth.get("macMachineId") or str(uuid.uuid4())
    return {
        "accessToken": str(access),
        "machineId": str(machine_id),
        "macMachineId": str(mac_machine_id),
    }


def write_auth(
    access_token: str,
    refresh_token: Optional[str] = None,
    *,
    email: str = "",
    card: str = "",
    machine_id: Optional[str] = None,
    mac_machine_id: Optional[str] = None,
    root: Path | None = None,
) -> bool:
    path = auth_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = read_auth(root)
        data["access_token"] = access_token
        data["refresh_token"] = refresh_token or access_token
        if email:
            data["email"] = email
        if card:
            data["card"] = card
        if machine_id:
            data["machine_id"] = machine_id
        elif not data.get("machine_id"):
            data["machine_id"] = str(uuid.uuid4())
        if mac_machine_id:
            data["mac_machine_id"] = mac_machine_id
        elif not data.get("mac_machine_id"):
            data["mac_machine_id"] = str(uuid.uuid4())
        _write_toml(path, data)
        return True
    except Exception:
        return False


def write_token_backup(pull_result: Dict[str, Any], root: Path | None = None) -> None:
    """可选：将 pull-token 原始响应写入备份文件。"""
    try:
        path = cursor_dir(root) / "token_backup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pull_result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
