from __future__ import annotations

"""Cursor 凭证：``persist/cursor/auth.json``（对齐 Cursor 桌面 auth.json）。"""

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from core.persist.migrate_util import write_json
from core.persist.paths import upstream_dir

UPSTREAM = "cursor"
AUTH_FILENAME = "auth.json"
LEGACY_AUTH_FILENAME = "auth.toml"

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader


def cursor_dir(root: Path | None = None) -> Path:
    return upstream_dir(UPSTREAM, root)


def auth_path(root: Path | None = None) -> Path:
    return cursor_dir(root) / AUTH_FILENAME


def _legacy_auth_path(root: Path | None = None) -> Path:
    return cursor_dir(root) / LEGACY_AUTH_FILENAME


def _read_legacy_toml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        return _toml_loader.loads(text)
    return _toml_loader.loads(text.encode("utf-8"))


def _normalize_auth(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    access = out.get("accessToken") or out.get("access_token") or ""
    refresh = out.get("refreshToken") or out.get("refresh_token") or access
    if access:
        out["accessToken"] = str(access)
        out["access_token"] = str(access)
    if refresh:
        out["refreshToken"] = str(refresh)
        out["refresh_token"] = str(refresh)
    machine = out.get("machineId") or out.get("machine_id") or str(uuid.uuid4())
    mac = out.get("macMachineId") or out.get("mac_machine_id") or str(uuid.uuid4())
    out["machineId"] = str(machine)
    out["machine_id"] = str(machine)
    out["macMachineId"] = str(mac)
    out["mac_machine_id"] = str(mac)
    return out


def read_auth(root: Path | None = None) -> Dict[str, Any]:
    path = auth_path(root)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return _normalize_auth(raw)
        except Exception:
            return {}
    legacy = _legacy_auth_path(root)
    if legacy.is_file():
        migrated = _normalize_auth(_read_legacy_toml(legacy))
        if migrated.get("accessToken"):
            write_json(path, migrated)
        return migrated
    return {}


def get_access_token(root: Path | None = None) -> Optional[str]:
    auth = read_auth(root)
    token = auth.get("accessToken") or auth.get("access_token")
    return str(token) if token else None


def get_token_bundle(root: Path | None = None) -> Dict[str, str]:
    auth = read_auth(root)
    access = auth.get("accessToken") or auth.get("access_token") or ""
    machine_id = auth.get("machineId") or auth.get("machine_id") or str(uuid.uuid4())
    mac_machine_id = auth.get("macMachineId") or auth.get("mac_machine_id") or str(uuid.uuid4())
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
        data["accessToken"] = access_token
        data["access_token"] = access_token
        refresh = refresh_token or access_token
        data["refreshToken"] = refresh
        data["refresh_token"] = refresh
        if email:
            data["email"] = email
        if card:
            data["card"] = card
        if machine_id:
            data["machineId"] = machine_id
            data["machine_id"] = machine_id
        elif not data.get("machineId"):
            mid = str(uuid.uuid4())
            data["machineId"] = mid
            data["machine_id"] = mid
        if mac_machine_id:
            data["macMachineId"] = mac_machine_id
            data["mac_machine_id"] = mac_machine_id
        elif not data.get("macMachineId"):
            mac = str(uuid.uuid4())
            data["macMachineId"] = mac
            data["mac_machine_id"] = mac
        write_json(path, _normalize_auth(data))
        return True
    except Exception:
        return False


def write_token_backup(pull_result: Dict[str, Any], root: Path | None = None) -> None:
    try:
        path = cursor_dir(root) / "token_backup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pull_result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
