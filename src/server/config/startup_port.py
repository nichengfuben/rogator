from __future__ import annotations

"""启动前端口占用检测与可选强制释放。"""

import logging
import socket
import sys
import time

from server.config.port_release import find_listen_pids, force_release_listen_port

logger = logging.getLogger("rogator")

_BIND_RETRY_SEC = 0.05
_BIND_RETRY_COUNT = 20


def _can_bind(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _wait_until_bind(host: str, port: int) -> bool:
    for _ in range(_BIND_RETRY_COUNT):
        if _can_bind(host, port):
            return True
        time.sleep(_BIND_RETRY_SEC)
    return False


def ensure_listen_port(host: str, port: int, *, force_kill: bool) -> None:
    """确保 host:port 可绑定；force_kill 时尝试终止占用该端口的进程。"""
    if _can_bind(host, port):
        return
    if not force_kill:
        pids = find_listen_pids(port)
        logger.error(
            "Port %s:%d already in use (PIDs: %s)! "
            "set server.startup_force_kill_port=true to auto-release",
            host,
            port,
            ",".join(str(pid) for pid in pids) or "unknown",
        )
        sys.exit(1)

    result = force_release_listen_port(port)
    if _wait_until_bind(host, port):
        if result.pids or result.released:
            logger.debug(
                "Released port %d (killed PIDs: %s)",
                port,
                ",".join(str(pid) for pid in result.pids) or "-",
            )
        return

    remaining = find_listen_pids(port)
    logger.error(
        "Port %s:%d still in use after force kill (PIDs: %s; detail=%s)",
        host,
        port,
        ",".join(str(pid) for pid in remaining) or "unknown",
        result.detail,
    )
    sys.exit(1)
