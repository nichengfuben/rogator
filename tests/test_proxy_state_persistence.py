"""Zen NodeManager 状态持久化（proxy_state.py）的最小覆盖。

覆盖三个不变量：
1. 文件缺失/损坏 → 返回全 0 元组，不抛异常
2. 读取时丢弃已过期的 mute，并统计 ``restored`` / ``expired``
3. 落盘时再过滤一次过期 mute，原子写
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from upstream.zen.proxy_state import (
    load_state_payload,
    save_state_payload,
)


def _describe(idx: int) -> str:
    return f"node-{idx}"


class TestLoadStatePayload:
    def test_missing_file_returns_zeros(self, tmp_path: Path) -> None:
        idx, muted, restored, expired = load_state_payload(
            str(tmp_path / "missing.json"), pool_size=3, describe_at=_describe,
        )
        assert idx == 0
        assert muted == {}
        assert restored == 0
        assert expired == 0

    def test_corrupted_file_returns_zeros(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not-json{{{", encoding="utf-8")
        idx, muted, restored, expired = load_state_payload(
            str(path), pool_size=3, describe_at=_describe,
        )
        assert (idx, muted, restored, expired) == (0, {}, 0, 0)

    def test_index_out_of_range_resets_to_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"current_node_index": 99, "muted": {}}),
            encoding="utf-8",
        )
        idx, _muted, _r, _e = load_state_payload(
            str(path), pool_size=2, describe_at=_describe,
        )
        assert idx == 0

    def test_expired_mutes_are_pruned(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        now = time.time()
        path.write_text(
            json.dumps(
                {
                    "current_node_index": 1,
                    "muted": {
                        "live": now + 3600,
                        "dead": now - 1,
                        "dead2": now - 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        idx, muted, restored, expired = load_state_payload(
            str(path), pool_size=3, describe_at=_describe,
        )
        assert idx == 1
        assert "live" in muted
        assert "dead" not in muted
        assert "dead2" not in muted
        assert restored == 1
        assert expired == 2

    def test_malformed_muted_entry_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "current_node_index": 0,
                    "muted": {"good": time.time() + 60, "bad": "not-a-float"},
                }
            ),
            encoding="utf-8",
        )
        _idx, muted, restored, expired = load_state_payload(
            str(path), pool_size=3, describe_at=_describe,
        )
        assert "good" in muted
        assert "bad" not in muted
        assert restored == 1
        assert expired == 0


class TestSaveStatePayload:
    def test_save_roundtrip_and_prunes_expired(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        now = time.time()
        save_state_payload(
            str(path),
            current_index=2,
            describe_at=_describe,
            muted={"alive": now + 60, "expired": now - 1},
        )
        assert path.exists()
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        assert data["current_node_index"] == 2
        assert data["current_node"] == "node-2"
        assert "alive" in data["muted"]
        assert "expired" not in data["muted"]

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text('{"old": true}', encoding="utf-8")
        save_state_payload(
            str(path),
            current_index=1,
            describe_at=_describe,
            muted={},
        )
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        assert "old" not in data
        assert data["current_node_index"] == 1

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "state.json"
        save_state_payload(
            str(path),
            current_index=0,
            describe_at=_describe,
            muted={},
        )
        assert path.exists()

    def test_load_after_save_restores(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        now = time.time()
        save_state_payload(
            str(path),
            current_index=1,
            describe_at=_describe,
            muted={"keep": now + 120},
        )
        idx, muted, restored, expired = load_state_payload(
            str(path), pool_size=3, describe_at=_describe,
        )
        assert idx == 1
        assert "keep" in muted
        assert restored == 1
        assert expired == 0