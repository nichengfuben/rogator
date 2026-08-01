from __future__ import annotations

import fnmatch
import glob as glob_mod
import os
import re
import subprocess
from typing import Any, Callable, Dict, List

from upstream.cursor.stream.exec.common import finish, truncate

ExecFn = Callable[[Dict[str, Any], Dict[str, Any], float], List[Dict[str, Any]]]


def _handle_read(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["readArgs"]
    path = args.get("path", "")
    offset = args.get("offset", 0) or 0
    limit = args.get("limit", 0) or 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if offset > 0:
            lines = lines[offset:]
        if limit > 0:
            lines = lines[:limit]
        content = truncate("".join(lines))
        return [finish(base, start, "readResult", {"success": {"content": content, "lineCount": len(lines)}})]
    except FileNotFoundError:
        return [finish(base, start, "readResult", {"fileNotFound": {"path": path}})]
    except Exception as exc:
        return [finish(base, start, "readResult", {"error": {"message": str(exc)}})]


def _handle_write(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["writeArgs"]
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return [finish(base, start, "writeResult", {"success": {}})]
    except Exception as exc:
        return [finish(base, start, "writeResult", {"error": {"message": str(exc)}})]


def _handle_ls(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["lsArgs"]
    path = args.get("path", ".")
    try:
        entries = []
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            entry: Dict[str, Any] = {"name": item, "isDirectory": os.path.isdir(full)}
            if not entry["isDirectory"]:
                try:
                    entry["size"] = os.path.getsize(full)
                except OSError:
                    pass
            entries.append(entry)
        return [finish(base, start, "lsResult", {"success": {"entries": entries}})]
    except Exception as exc:
        return [finish(base, start, "lsResult", {"error": {"message": str(exc)}})]


def _grep_glob_matches(glob_pattern: str, path: str) -> List[str]:
    matched = glob_mod.glob(glob_pattern, recursive=True)
    if matched:
        return matched[:100]
    base_pat = os.path.basename(glob_pattern)
    root = path if os.path.isdir(path) else "."
    for r, _d, files in os.walk(root):
        for f in files:
            if fnmatch.fnmatch(f, base_pat):
                matched.append(os.path.join(r, f))
                if len(matched) >= 100:
                    return matched
    return matched


def _grep_file_lines(fpath: str, pattern: str, matches: List[Dict[str, Any]], *, with_path: bool) -> bool:
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if not re.search(pattern, line):
                    continue
                item: Dict[str, Any] = {"lineNumber": i, "content": line.rstrip()[:500]}
                if with_path:
                    item["filePath"] = fpath
                matches.append(item)
                if len(matches) >= 50:
                    return True
    except (OSError, UnicodeDecodeError):
        pass
    return len(matches) >= 50


def _grep_pattern_matches(pattern: str, path: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        _grep_file_lines(path, pattern, matches, with_path=False)
        return matches
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if _grep_file_lines(os.path.join(root, fname), pattern, matches, with_path=True):
                return matches
    return matches


def _handle_grep(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["grepArgs"]
    pattern = args.get("pattern", "")
    glob_pattern = args.get("glob", "")
    output_mode = args.get("outputMode", "files_with_matches")
    path = args.get("path", ".")
    if glob_pattern and not pattern:
        try:
            matched = _grep_glob_matches(glob_pattern, path)
            payload = {
                "success": {
                    "pattern": glob_pattern, "path": path, "outputMode": output_mode,
                    "workspaceResults": {path: {"files": {"files": matched, "totalFiles": len(matched), "clientTruncated": len(matched) >= 100, "ripgrepTruncated": False}}},
                },
            }
            return [finish(base, start, "grepResult", payload)]
        except Exception as exc:
            return [finish(base, start, "grepResult", {"error": {"message": str(exc)}})]
    if pattern:
        try:
            return [finish(base, start, "grepResult", {"success": {"matches": _grep_pattern_matches(pattern, path)}})]
        except Exception as exc:
            return [finish(base, start, "grepResult", {"error": {"message": str(exc)}})]
    return [finish(base, start, "grepResult", {"success": {"matches": []}})]


def _handle_delete(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    path = (msg.get("deleteArgs") or {}).get("path", "")
    try:
        if not os.path.exists(path):
            return [finish(base, start, "deleteResult", {"fileNotFound": {"path": path}})]
        if os.path.isdir(path):
            return [finish(base, start, "deleteResult", {"notFile": {"path": path}})]
        os.remove(path)
        return [finish(base, start, "deleteResult", {"success": {}})]
    except PermissionError:
        return [finish(base, start, "deleteResult", {"permissionDenied": {"path": path}})]
    except Exception as exc:
        return [finish(base, start, "deleteResult", {"error": {"message": str(exc)}})]


def _read_file_result(msg: dict, base: dict, start: float, args_key: str, result_key: str) -> List[Dict[str, Any]]:
    path = (msg.get(args_key) or {}).get("path", "")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = truncate(f.read())
        return [finish(base, start, result_key, {"success": {"content": content}})]
    except FileNotFoundError:
        err = {"fileNotFound": {"path": path}} if result_key == "redactedReadResult" else {"error": {"error": f"File not found: {path}"}}
        return [finish(base, start, result_key, err)]
    except Exception as exc:
        key = "message" if result_key == "redactedReadResult" else "error"
        return [finish(base, start, result_key, {"error": {key: str(exc)}})]


def _handle_pi_edit(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["piEditArgs"]
    path = args.get("path", "")
    edits = args.get("edits", [])
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        for edit in edits:
            old_text = edit.get("oldText", "")
            new_text = edit.get("newText", "")
            if not old_text:
                raise ValueError("oldText must not be empty")
            count = content.count(old_text)
            if count == 0:
                raise ValueError("oldText not found in file")
            if count > 1:
                raise ValueError(f"oldText found {count} times, must be unique")
            content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return [finish(base, start, "piEditResult", {"success": {}})]
    except FileNotFoundError:
        return [finish(base, start, "piEditResult", {"error": {"error": f"File not found: {path}"}})]
    except ValueError as exc:
        return [finish(base, start, "piEditResult", {"rejected": {"reason": str(exc)}})]
    except Exception as exc:
        return [finish(base, start, "piEditResult", {"error": {"error": str(exc)}})]


def _walk_match_files(path: str, pattern: str, limit: int) -> List[str]:
    matched: List[str] = []
    if not os.path.isdir(path):
        return matched
    for root, _dirs, files in os.walk(path):
        for fname in files:
            full = os.path.join(root, fname)
            if fnmatch.fnmatch(fname, pattern) or fnmatch.fnmatch(full, pattern):
                matched.append(full)
                if len(matched) >= limit:
                    return matched
    return matched


def _handle_pi_find(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg["piFindArgs"]
    try:
        files = _walk_match_files(args.get("path", "."), args.get("pattern", ""), 100)
        return [finish(base, start, "piFindResult", {"success": {"files": files}})]
    except Exception as exc:
        return [finish(base, start, "piFindResult", {"error": {"error": str(exc)}})]


def _handle_glob(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    args = msg.get("globToolArgs", {})
    try:
        files = _walk_match_files(args.get("path", "."), args.get("pattern", "**/*"), 100)
        return [finish(base, start, "globToolResult", {"success": {"files": files}})]
    except Exception as exc:
        return [finish(base, start, "globToolResult", {"error": {"error": str(exc)}})]


def _handle_blame(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    path = (msg.get("blameByFilePathArgs") or {}).get("path", "")
    try:
        proc = subprocess.run(["git", "blame", "--porcelain", path], capture_output=True, text=True, timeout=30)
        return [finish(base, start, "blameByFilePathResult", {"success": {"blame": truncate(proc.stdout)}})]
    except Exception as exc:
        return [finish(base, start, "blameByFilePathResult", {"error": {"error": str(exc)}})]


def _handle_pi_write(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    adapted = dict(msg)
    adapted["writeArgs"] = msg.get("piWriteArgs") or {}
    result = _handle_write(adapted, base, start)[0]
    result["piWriteResult"] = result.pop("writeResult")
    return [result]


def _handle_pi_grep(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    adapted = dict(msg)
    adapted["grepArgs"] = msg.get("piGrepArgs") or {}
    result = _handle_grep(adapted, base, start)[0]
    result["piGrepResult"] = result.pop("grepResult")
    return [result]


def _handle_pi_ls(msg: dict, base: dict, start: float) -> List[Dict[str, Any]]:
    adapted = dict(msg)
    adapted["lsArgs"] = msg.get("piLsArgs") or {}
    result = _handle_ls(adapted, base, start)[0]
    result["piLsResult"] = result.pop("lsResult")
    return [result]


FS_HANDLERS: Dict[str, ExecFn] = {
    "readArgs": _handle_read,
    "writeArgs": _handle_write,
    "lsArgs": _handle_ls,
    "grepArgs": _handle_grep,
    "deleteArgs": _handle_delete,
    "redactedReadArgs": lambda m, b, s: _read_file_result(m, b, s, "redactedReadArgs", "redactedReadResult"),
    "piReadArgs": lambda m, b, s: _read_file_result(m, b, s, "piReadArgs", "piReadResult"),
    "piWriteArgs": _handle_pi_write,
    "piGrepArgs": _handle_pi_grep,
    "piLsArgs": _handle_pi_ls,
    "piEditArgs": _handle_pi_edit,
    "piFindArgs": _handle_pi_find,
    "globToolArgs": _handle_glob,
    "blameByFilePathArgs": _handle_blame,
}
