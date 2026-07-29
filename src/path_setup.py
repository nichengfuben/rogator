"""将项目根目录与 ``src/`` 加入 ``sys.path``（须在 server/handlers 导入前执行）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def ensure_import_paths() -> None:
    for directory in (SRC, ROOT):
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


ensure_import_paths()
