from __future__ import annotations

"""启动前端口占用检测与可选强制释放（对齐 Provider-Evo startup_force_kill_port）。"""

import logging
import socket
import sys

from echotools.exec.process.port import ensure_port_available

logger = logging.getLogger("rogator")


def _can_bind(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def ensure_listen_port(host: str, port: int, *, force_kill: bool) -> None:
    """确保 host:port 可绑定；force_kill 时尝试终止占用该端口的进程。"""
    if _can_bind(host, port):
        return
    if not force_kill:
        logger.error(
            "Port %s:%d already in use! (set server.startup_force_kill_port=true to auto-release)",
            host,
            port,
        )
        sys.exit(1)

    result = ensure_port_available(port, True)
    if result.released and _can_bind(host, port):
        killed = ",".join(str(pid) for pid in result.pids) or "-"
        logger.debug("Released port %d (killed PIDs: %s)", port, killed)
        return

    remaining = result.pids if result.occupied and not result.released else []
    if _can_bind(host, port):
        return

    logger.error(
        "Port %s:%d still in use after force kill (PIDs: %s)",
        host,
        port,
        ",".join(str(pid) for pid in remaining) or "unknown",
    )
    sys.exit(1)
