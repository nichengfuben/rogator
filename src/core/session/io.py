from __future__ import annotations

"""磁盘原子写：tmp + os.replace，失败回退到直接覆盖，并跨盘 fallback。

``os.replace`` 在 Windows 共享盘 / NFS 跨盘时会抛 ``OSError``，那时退回到
``shutil.move``（跨盘）或 ``Path.replace``（不跨盘）。backoff 循环处理
防病毒软件临时持锁等短窗口阻塞。
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional


async def atomic_write_text_async(path: Path, content: str) -> None:
    from core.transport.blocking import run_blocking

    await run_blocking(atomic_write_text, path, content)


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except OSError:
        pass


def _replace_cross_device(tmp_path: Path, target: Path) -> None:
    try:
        os.replace(str(tmp_path), str(target))
        return
    except OSError as exc:
        if getattr(exc, "errno", None) not in (17, 18):  # EEXIST, ENOTEMPTY 不算跨盘
            cross = getattr(exc, "errno", None) in (16, 30)  # EBUSY / EACCES 也走 fallback
        else:
            cross = False
        try:
            shutil.move(str(tmp_path), str(target))
            return
        except OSError as move_exc:
            if not cross and move_exc.errno not in (16, 30):
                raise move_exc from exc
            try:
                target.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
            finally:
                _safe_unlink(tmp_path)
            return


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    backoff = (0.0, 0.025, 0.05, 0.1, 0.2)
    last_error: Optional[OSError] = None
    for wait in backoff:
        if wait:
            time.sleep(wait)
        try:
            tmp_path.write_text(content, encoding="utf-8")
            _replace_cross_device(tmp_path, path)
            return
        except OSError as exc:
            last_error = exc
            _safe_unlink(tmp_path)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        if last_error is not None and sys.platform == "win32":
            raise last_error from exc
        raise