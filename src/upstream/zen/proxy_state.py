from __future__ import annotations

"""Zen 节点状态持久化：与 proxy.py 的 NodeManager 解耦，避免大文件行数膨胀。"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("rogator")


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """写入 JSON；用同名目录临时文件 + ``os.replace`` 保证落盘原子性。

    写失败时清理临时文件，抛 ``OSError`` 由调用方处理。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(path.parent),
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            tmp_path = fh.name
        os.replace(tmp_path, str(path))
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load_state_payload(
    state_file: str,
    pool_size: int,
    describe_at: Any,
) -> tuple[int, Dict[str, float], int, int]:
    """读取节点状态文件，恢复 ``(current_index, muted_map, restored, expired)``。
    找不到文件或解析失败时返回 ``(0, {}, 0, 0)``，不抛异常。
    """
    path = Path(state_file)
    now = time.time()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0, {}, 0, 0
    except Exception as exc:
        logger.warning("zen NodeManager load failed: %s", exc)
        return 0, {}, 0, 0

    idx = int(data.get("current_node_index", 0) or 0)
    if idx < 0 or idx >= pool_size:
        idx = 0
    raw_muted = data.get("muted") or {}
    if not isinstance(raw_muted, dict):
        raw_muted = {}
    muted: Dict[str, float] = {}
    restored = 0
    expired = 0
    for desc, until in raw_muted.items():
        try:
            until_val = float(until)
        except (TypeError, ValueError):
            continue
        if until_val <= now:
            expired += 1
            continue
        muted[str(desc)] = until_val
        restored += 1
    logger.debug(
        "zen NodeManager restored index=%d (%s) mutes=%d expired=%d",
        idx, describe_at(idx), restored, expired,
    )
    return idx, muted, restored, expired


def save_state_payload(
    state_file: str,
    *,
    current_index: int,
    describe_at: Any,
    muted: Dict[str, float],
) -> None:
    """落盘当前节点状态；剔除已过期的 mute，避免历史条目堆积。"""
    now = time.time()
    live_muted = {
        desc: until for desc, until in muted.items() if until > now
    }
    payload = {
        "current_node_index": current_index,
        "current_node": describe_at(current_index),
        "updated_at": int(now),
        "muted": live_muted,
    }
    try:
        _write_json_atomic(Path(state_file), payload)
    except Exception as exc:
        logger.error("zen NodeManager save failed: %s", exc)