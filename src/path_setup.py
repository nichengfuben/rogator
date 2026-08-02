"""将项目根目录与 ``src/`` 加入 ``sys.path``（须在 server/handlers 导入前执行）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

_BROKEN_ECHOTOOLS: Final[frozenset[str]] = frozenset({"2.4.2"})


def ensure_import_paths() -> None:
    for directory in (SRC, ROOT):
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def ensure_echotools_importable() -> None:
    """echotools 2.4.2 存在 fncall 循环导入，启动前冒烟并给出修复提示。"""
    try:
        import echotools
    except ImportError as exc:
        raise RuntimeError("缺少依赖 echotools，请执行: pip install -r requirements.txt") from exc

    version = str(getattr(echotools, "__version__", "") or "")
    if version in _BROKEN_ECHOTOOLS:
        raise RuntimeError(
            f"echotools {version} 存在 fncall 循环导入缺陷，无法启动。"
            " 请执行: pip install \"echotools>=2.4.0,!=2.4.2\""
        )

    from echotools import FncallStreamParser, ToolProtocol, get_protocol  # noqa: F401
    from echotools.exec.fncall.protocols.entml_think.core import (  # noqa: F401
        normalize_thinking_level,
        resolve_thinking_injection,
    )


ensure_import_paths()
ensure_echotools_importable()
