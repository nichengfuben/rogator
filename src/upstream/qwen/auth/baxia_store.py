from __future__ import annotations

"""按账号持久化 Baxia 指纹与 bx-umidtoken（独立于 session cleanup）。"""

import json
import logging
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from core.persist.paths import PROJECT_ROOT
from core.session.accounts import accounts_for_upstream
from core.session.io import atomic_write_text
from upstream.qwen.auth.crypto import build_fingerprint, validate_bxumidtoken

logger = logging.getLogger("rogator")

_PROFILES_FILE = PROJECT_ROOT / "persist" / "qwen" / "baxia_profiles.json"
_lock = threading.Lock()


@dataclass(frozen=True)
class BaxiaProfile:
    fingerprint: str
    bx_umidtoken: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "fingerprint": self.fingerprint,
            "bx_umidtoken": self.bx_umidtoken,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BaxiaProfile":
        return BaxiaProfile(
            fingerprint=str(data.get("fingerprint") or ""),
            bx_umidtoken=str(data.get("bx_umidtoken") or ""),
        )


def baxia_profiles_path() -> Path:
    return _PROFILES_FILE


def _new_random_bx_umidtoken() -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "T2gA" + "".join(secrets.choice(alphabet) for _ in range(40))


def _new_profile() -> BaxiaProfile:
    return BaxiaProfile(
        fingerprint=build_fingerprint(),
        bx_umidtoken=_new_random_bx_umidtoken(),
    )


def _valid_profile_raw(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    fp = str(data.get("fingerprint") or "").strip()
    umid = str(data.get("bx_umidtoken") or "").strip()
    return bool(fp and validate_bxumidtoken(umid))


def _load_unlocked() -> Dict[str, Dict[str, str]]:
    path = baxia_profiles_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 Baxia profiles 失败，将重建缺失项: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for username, entry in raw.items():
        key = str(username).strip()
        if not key or not _valid_profile_raw(entry):
            continue
        prof = BaxiaProfile.from_dict(entry)
        out[key] = prof.to_dict()
    return out


def _save_unlocked(data: Dict[str, Dict[str, str]]) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(baxia_profiles_path(), payload)


def ensure_profiles(usernames: Iterable[str]) -> Tuple[int, int]:
    """补齐缺失 profile；已有且合法则保留。返回 (kept, created)。"""
    names = {str(u).strip() for u in usernames if str(u).strip()}
    if not names:
        return 0, 0
    with _lock:
        data = _load_unlocked()
        kept = 0
        created = 0
        for name in sorted(names):
            if name in data and _valid_profile_raw(data[name]):
                kept += 1
                continue
            data[name] = _new_profile().to_dict()
            created += 1
        if created:
            _save_unlocked(data)
        return kept, created


def ensure_pool_baxia_profiles() -> Tuple[int, int]:
    """启动时为 qwen 号池检查并补齐 Baxia profile。"""
    usernames = [acc.username for acc in accounts_for_upstream("qwen")]
    return ensure_profiles(usernames)


def regenerate_profile(username: str) -> BaxiaProfile:
    """登录时重新生成账号 Baxia 凭证（覆盖已有）。"""
    key = str(username or "").strip()
    if not key:
        raise ValueError("username required for Baxia profile regeneration")
    with _lock:
        data = _load_unlocked()
        prof = _new_profile()
        data[key] = prof.to_dict()
        _save_unlocked(data)
        logger.info("Baxia profile regenerated for %s", key[:6] + "***")
        return prof


def get_profile(username: str) -> BaxiaProfile:
    """取账号 profile；缺失时单条补齐（不覆盖已有）。"""
    key = str(username or "").strip()
    if not key:
        raise ValueError("username required for Baxia profile lookup")
    with _lock:
        data = _load_unlocked()
        entry = data.get(key)
        if entry and _valid_profile_raw(entry):
            return BaxiaProfile.from_dict(entry)
        prof = _new_profile()
        data[key] = prof.to_dict()
        _save_unlocked(data)
        logger.info("Baxia profile created for %s", key[:6] + "***")
        return prof
