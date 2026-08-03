from __future__ import annotations

"""Call vendored fireyejs via Node/jsdom (no real browser)."""

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("rogator")

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "fireye"
_RUNNER = _VENDOR_DIR / "runner.js"
_NODE = os.environ.get("QWEN_FIREYE_NODE", "").strip() or shutil.which("node") or ""

_lock = threading.Lock()
_proc: Optional[subprocess.Popen[str]] = None


def fireye_available() -> bool:
    return bool(_NODE) and _RUNNER.is_file() and (_VENDOR_DIR / "fireyejs.js").is_file()


def _ensure_proc() -> subprocess.Popen[str]:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    if not fireye_available():
        raise RuntimeError("fireye runner unavailable (node/fireyejs missing)")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _proc = subprocess.Popen(
        [_NODE, str(_RUNNER)],
        cwd=str(_VENDOR_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    return _proc


def reset_fireye_worker() -> None:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            try:
                if _proc.stdin:
                    _proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                    _proc.stdin.flush()
            except Exception:
                pass
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None


def request_fireye_tokens(url: str = "") -> Dict[str, Any]:
    """Ask fireye runner for bx-ua / bx-umidtoken. Raises on hard failure."""
    payload = {"cmd": "token", "url": url or "https://chat.qwen.ai/api/v2/chat/completions"}
    with _lock:
        proc = _ensure_proc()
        assert proc.stdin and proc.stdout
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            _proc = None
            raise RuntimeError("fireye runner closed unexpectedly")
        data = json.loads(line)
        if not data.get("ok"):
            raise RuntimeError(f"fireye token error: {data}")
        return data
