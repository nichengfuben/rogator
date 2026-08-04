from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional


async def atomic_write_text_async(path: Path, content: str) -> None:
    from core.transport.blocking import run_blocking

    await run_blocking(atomic_write_text, path, content)


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
            os.replace(str(tmp_path), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        if last_error is not None and sys.platform == "win32":
            raise last_error from exc
        raise
