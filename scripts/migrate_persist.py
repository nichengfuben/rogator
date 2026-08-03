#!/usr/bin/env python3
"""将 persist 根目录统一文件迁移为按 upstream 分桶。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.persist.migrate import migrate_all, migrate_sessions_upstream  # noqa: E402
from core.persist.paths import KNOWN_UPSTREAMS, PROJECT_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移 persist 到按 upstream 分桶")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目根目录（默认：脚本所在仓库根）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅报告，不写入或归档",
    )
    parser.add_argument(
        "--force-sessions",
        action="store_true",
        help="即使分桶 sessions 已存在也强制从统一/备份文件重写",
    )
    args = parser.parse_args()
    persist = args.root / "persist"
    if not persist.is_dir():
        print(f"persist 目录不存在: {persist}", file=sys.stderr)
        return 1

    if args.dry_run:
        unified = {
            "login_history.json": (persist / "login_history.json").is_file(),
            "sessions.json": (persist / "sessions.json").is_file(),
            "models.json": (persist / "models.json").is_file(),
        }
        per_upstream = {}
        config_upstream = args.root / "config" / "upstream"
        for upstream in KNOWN_UPSTREAMS:
            d = persist / upstream
            per_upstream[upstream] = {
                name: (d / name).is_file()
                for name in (
                    "login_history.json",
                    "sessions.json",
                    "models.json",
                )
            }
            per_upstream[upstream]["accounts.csv"] = (
                config_upstream / upstream / "accounts.csv"
            ).is_file()
        print(json.dumps({"unified": unified, "per_upstream": per_upstream}, indent=2))
        return 0

    if args.force_sessions:
        restored: list[str] = []
        for upstream in KNOWN_UPSTREAMS:
            if migrate_sessions_upstream(upstream, args.root, force=True):
                restored.append(upstream)
        print(json.dumps({"sessions_restored": restored}, indent=2, ensure_ascii=False))
        return 0

    results = migrate_all(args.root, archive_unified=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
