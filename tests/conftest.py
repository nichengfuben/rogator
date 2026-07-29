"""pytest 启动时注入 src/ 与项目根目录到 sys.path。"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
for _entry in (_root / "src", _root):
    _path = str(_entry)
    if _path not in sys.path:
        sys.path.insert(0, _path)
import path_setup  # noqa: F401
