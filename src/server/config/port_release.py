from __future__ import annotations

"""跨 Linux 发行版的监听端口占用检测与释放。"""

import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set

logger = logging.getLogger("rogator")

_SS_PID_RE = re.compile(r"pid=(\d+)")
_LISTEN_STATE = "0A"


@dataclass(frozen=True)
class PortReleaseOutcome:
    port: int
    occupied: bool
    released: bool
    pids: List[int]
    detail: str


def _linux_port_hex(port: int) -> str:
    """/proc/net/tcp 本地端口字段（小端 hex）。"""
    return f"{port & 0xFF:02X}" f"{(port >> 8) & 0xFF:02X}"


def _parse_int_tokens(text: str) -> Set[int]:
    out: Set[int] = set()
    for token in text.replace(",", " ").split():
        token = token.strip()
        if not token.isdigit():
            continue
        pid = int(token)
        if pid > 0:
            out.add(pid)
    return out


def _run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )


def _find_pids_ss(port: int) -> Set[int]:
    pids: Set[int] = set()
    filters = (
        f"sport = :{port}",
        f"( sport = :{port} )",
    )
    flags = ("-ltnp", "-ltn")
    for family in ("", "6"):
        for flag in flags:
            for filt in filters:
                argv = ["ss", f"-H{flag}{family}", filt]
                try:
                    result = _run_command(argv)
                except OSError:
                    continue
                for line in (result.stdout or "").splitlines():
                    pids.update(int(m.group(1)) for m in _SS_PID_RE.finditer(line))
    return pids


def _find_pids_lsof(port: int) -> Set[int]:
    pids: Set[int] = set()
    templates = (
        ["lsof", "-nP", "-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        ["lsof", "-nP", "-i:{port}", "-sTCP:LISTEN", "-t"],
        ["lsof", "-ti", "tcp:{port}"],
        ["lsof", "-ti", ":{port}"],
    )
    for tmpl in templates:
        argv = [part.format(port=port) for part in tmpl]
        try:
            result = _run_command(argv)
        except OSError:
            continue
        pids.update(_parse_int_tokens(result.stdout or ""))
    return pids


def _find_pids_fuser(port: int) -> Set[int]:
    pids: Set[int] = set()
    for argv in (
        ["fuser", "-n", "tcp", str(port)],
        ["fuser", f"{port}/tcp"],
    ):
        try:
            result = _run_command(argv)
        except OSError:
            continue
        pids.update(_parse_int_tokens(result.stdout or ""))
        pids.update(_parse_int_tokens(result.stderr or ""))
    return pids


def _inodes_listening_on_port(port: int) -> Set[str]:
    if sys.platform not in ("linux", "linux2"):
        return set()
    needle = _linux_port_hex(port).upper()
    inodes: Set[str] = set()
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            cols = line.split()
            if len(cols) < 10 or cols[3] != _LISTEN_STATE:
                continue
            local_port = cols[1].split(":")[-1].upper()
            if local_port == needle:
                inodes.add(cols[9])
    return inodes


def _find_pids_proc(port: int) -> Set[int]:
    inodes = _inodes_listening_on_port(port)
    if not inodes:
        return set()
    pids: Set[int] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return pids
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        if not fd_dir.is_dir():
            continue
        try:
            for link in fd_dir.iterdir():
                try:
                    target = os.readlink(link)
                except OSError:
                    continue
                if not target.startswith("socket:["):
                    continue
                inode = target.split("[", 1)[-1].rstrip("]")
                if inode in inodes:
                    pids.add(int(entry.name))
                    break
        except OSError:
            continue
    return pids


def find_listen_pids(port: int) -> List[int]:
    """查找监听 port 的进程（多工具 fallback，适配不同 Linux）。"""
    if sys.platform == "win32":
        from echotools.exec.process.port import _find_pids_by_port  # noqa: SLF001

        return sorted(_find_pids_by_port(port))

    found: Set[int] = set()
    for finder in (_find_pids_ss, _find_pids_lsof, _find_pids_fuser, _find_pids_proc):
        found.update(finder(port))
        if found:
            break
    if not found:
        for finder in (_find_pids_lsof, _find_pids_fuser, _find_pids_proc):
            found.update(finder(port))
    return sorted(found)


def _kill_pid_unix(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        logger.warning("SIGTERM 失败 [%s]: %s", pid, exc)
        return False
    for _ in range(10):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            return True
    try:
        os.kill(pid, signal.SIGKILL)
        logger.warning("已 SIGKILL 占用端口的进程: %s", pid)
        return True
    except OSError as exc:
        logger.warning("SIGKILL 失败 [%s]: %s", pid, exc)
        return False


def _try_fuser_kill(port: int) -> None:
    for argv in (
        ["fuser", "-k", "-TERM", f"{port}/tcp"],
        ["fuser", "-k", f"{port}/tcp"],
    ):
        try:
            _run_command(argv)
        except OSError:
            continue


def force_release_listen_port(port: int) -> PortReleaseOutcome:
    """强制释放 port；无法解析 PID 时仍尝试 fuser -k。"""
    if sys.platform == "win32":
        from echotools.exec.process.port import ensure_port_available

        raw = ensure_port_available(port, True)
        return PortReleaseOutcome(
            port=raw.port,
            occupied=raw.occupied,
            released=raw.released,
            pids=list(raw.pids),
            detail=raw.detail,
        )

    pids = find_listen_pids(port)
    if not pids:
        _try_fuser_kill(port)
        time.sleep(0.3)
        pids = find_listen_pids(port)

    killed: List[int] = []
    for pid in pids:
        if _kill_pid_unix(pid):
            killed.append(pid)

    if not pids:
        _try_fuser_kill(port)

    for attempt in range(6):
        remaining = find_listen_pids(port)
        if not remaining:
            return PortReleaseOutcome(
                port, True, True, killed, "released listen port",
            )
        if attempt >= 3:
            for pid in remaining:
                if pid not in killed and _kill_pid_unix(pid):
                    killed.append(pid)
        else:
            _try_fuser_kill(port)
        time.sleep(0.25 * (attempt + 1))

    remaining = find_listen_pids(port)
    return PortReleaseOutcome(
        port,
        True,
        not remaining,
        killed,
        "failed to release all" if remaining else "released after wait",
    )
