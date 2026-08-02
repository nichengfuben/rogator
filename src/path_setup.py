"""将项目根目录与 ``src/`` 加入 ``sys.path``（须在 server/handlers 导入前执行）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

_BROKEN_ECHOTOOLS: Final[frozenset[str]] = frozenset({"2.4.2"})
_MIN_ECHOTOOLS: Final[Tuple[int, ...]] = (2, 4, 5)
_ECHOTOOLS_INSTALL_HINT = 'pip install "echotools>=2.4.5,!=2.4.2"'


def ensure_import_paths() -> None:
    for directory in (SRC, ROOT):
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _echotools_version_tuple(version: str) -> Tuple[int, ...]:
    parts: list[int] = []
    for seg in (version or "").strip().split(".")[:3]:
        try:
            parts.append(int(seg))
        except ValueError:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def ensure_echotools_importable() -> None:
    """启动前校验 echotools 版本与 fncall/tool_id 关键导出。"""
    try:
        import echotools
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖 echotools，请执行: pip install -r requirements.txt"
        ) from exc

    version = str(getattr(echotools, "__version__", "") or "")
    if version in _BROKEN_ECHOTOOLS:
        raise RuntimeError(
            f"echotools {version} 存在 fncall 循环导入缺陷，无法启动。"
            f" 请执行: {_ECHOTOOLS_INSTALL_HINT}"
        )
    if _echotools_version_tuple(version) < _MIN_ECHOTOOLS:
        raise RuntimeError(
            f"echotools {version or '(unknown)'} 过低，Rogator 需要 >= 2.4.5。"
            f" 请执行: {_ECHOTOOLS_INSTALL_HINT}"
        )

    from echotools import FncallStreamParser, ToolProtocol, get_protocol  # noqa: F401
    from echotools.base.ids import gen_tool_id  # noqa: F401
    from echotools.exec.fncall.protocols.entml_think.core import (  # noqa: F401
        normalize_thinking_level,
        resolve_thinking_injection,
    )
    from echotools.exec.fncall.tool_id import (  # noqa: F401
        ensure_toolu_tool_call_id,
        fix_tool_call_id,
    )


ensure_import_paths()
ensure_echotools_importable()
