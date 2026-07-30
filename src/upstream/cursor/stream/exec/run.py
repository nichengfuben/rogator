from __future__ import annotations

import os
import platform
import subprocess
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from upstream.cursor.stream.exec.common import finish, truncate

ExecFn = Callable[[Dict[str, Any], Dict[str, Any], float, Optional[Dict[str, Callable]]], List[Dict[str, Any]]]

_HOOK_RESPONSES: Dict[str, Dict[str, Any]] = {
    "preCompact": {"preCompact": {}},
    "subagentStart": {"subagentStart": {"permission": "allow"}},
    "subagentStop": {"subagentStop": {}},
    "beforeSubmitPrompt": {"beforeSubmitPrompt": {"continue": True}},
    "afterAgentResponse": {"afterAgentResponse": {}},
    "afterAgentThought": {"afterAgentThought": {}},
    "stop": {"stop": {}},
    "preToolUse": {"preToolUse": {"permission": "allow"}},
    "postToolUse": {"postToolUse": {}},
    "postToolUseFailure": {"postToolUseFailure": {}},
}


def _handle_mcp(
    msg: dict,
    base: dict,
    start: float,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    *,
    defer_mcp: bool = False,
) -> List[Dict[str, Any]]:
    args = msg["mcpArgs"]
    tool_name = args.get("name") or args.get("toolName") or ""
    tool_args = args.get("args", {})
    handler = (tool_handlers or {}).get(tool_name)
    if handler is None and tool_name:
        # 也允许短名 handler
        short = args.get("toolName") or ""
        if short and short != tool_name:
            handler = (tool_handlers or {}).get(short)
    if handler:
        try:
            response_text = handler(tool_args)
            payload = {"success": {"content": [{"text": {"text": response_text}}], "isError": False}}
        except Exception as exc:
            payload = {"error": {"error": str(exc)}}
    elif defer_mcp:
        # OpenAI 代理：真实结果由客户端下一轮 conversationHistory 回灌
        payload = {
            "success": {
                "content": [{"text": {"text": ""}}],
                "isError": False,
            },
        }
    else:
        payload = {"success": {"content": [{"text": {"text": f"Unknown tool: {tool_name}"}}], "isError": False}}
    return [finish(base, start, "mcpResult", payload)]


def _handle_shell(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    tool = "shellStreamArgs" if "shellStreamArgs" in msg else "shellArgs"
    args = msg.get("shellArgs") or msg.get("shellStreamArgs", {})
    cmd = args.get("command", "")
    cwd = args.get("workingDirectory", "") or None
    timeout = args.get("timeout", 30) or 30
    is_stream = tool == "shellStreamArgs"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        elapsed = int((__import__("time").time() - start) * 1000)
        if is_stream:
            out: List[Dict[str, Any]] = []
            if proc.stdout:
                m = dict(base)
                m["shellStream"] = {"stdout": {"data": truncate(proc.stdout)}}
                out.append(m)
            if proc.stderr:
                m = dict(base)
                m["shellStream"] = {"stderr": {"data": truncate(proc.stderr)}}
                out.append(m)
            m = dict(base)
            m["shellStream"] = {"exit": {"code": proc.returncode, "cwd": cwd or os.getcwd(), "localExecutionTimeMs": elapsed}}
            m["localExecutionTimeMs"] = elapsed
            out.append(m)
            return out
        return [finish(base, start, "shellResult", {
            "stdout": truncate(proc.stdout), "stderr": truncate(proc.stderr), "exitCode": proc.returncode,
        })]
    except subprocess.TimeoutExpired:
        return [finish(base, start, "shellResult", {"stdout": "", "stderr": f"Timed out after {timeout}s", "exitCode": -1})]
    except Exception as exc:
        return [finish(base, start, "shellResult", {"stdout": "", "stderr": str(exc), "exitCode": -1})]


def _handle_pi_bash(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    args = msg["piBashArgs"]
    cmd = args.get("command", "")
    timeout = args.get("timeout", 30) or 30
    cwd = args.get("workingDirectory", "") or None
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        output = truncate(proc.stdout + proc.stderr)
        if proc.returncode == 0:
            return [finish(base, start, "piBashResult", {"success": {"output": output}})]
        return [finish(base, start, "piBashResult", {"error": {"error": output, "exitCode": proc.returncode}})]
    except subprocess.TimeoutExpired:
        return [finish(base, start, "piBashResult", {"error": {"error": f"Timed out after {timeout}s"}})]
    except Exception as exc:
        return [finish(base, start, "piBashResult", {"error": {"error": str(exc)}})]


def _handle_mini_swe_bash(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    cmd = (msg.get("miniSweAgentBashArgs") or {}).get("command", "")
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return [finish(base, start, "miniSweAgentBashResult", {"success": {"output": truncate(proc.stdout + proc.stderr)}})]
    except Exception as exc:
        return [finish(base, start, "miniSweAgentBashResult", {"error": {"error": str(exc)}})]


def _fetch_url(msg: dict, base: dict, start: float, args_key: str, result_key: str) -> List[Dict[str, Any]]:
    url = (msg.get(args_key) or {}).get("url", "")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CursorMVP/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if result_key == "fetchResult":
                payload = {"success": {"url": url, "content": truncate(content), "statusCode": resp.status, "contentType": resp.headers.get("Content-Type", "")}}
            else:
                payload = {"success": {"content": truncate(content), "statusCode": resp.status}}
        return [finish(base, start, result_key, payload)]
    except Exception as exc:
        key = "message" if result_key == "fetchResult" else "error"
        return [finish(base, start, result_key, {"error": {key: str(exc)}})]


def _handle_git_diff(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    args = msg.get("gitDiffRequest", {})
    files = args.get("files", [])
    base_sha = args.get("baseSha", "")
    head_sha = args.get("headSha", "")
    base_branch = args.get("baseBranch", "")
    try:
        if base_sha and head_sha:
            cmd = ["git", "diff", base_sha, head_sha]
        elif base_branch:
            cmd = ["git", "diff", base_branch]
        else:
            cmd = ["git", "diff"]
        if files:
            cmd.extend(["--", *files])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        return [finish(base, start, "gitDiffResponse", {"diff": truncate(proc.stdout)})]
    except Exception as exc:
        return [finish(base, start, "gitDiffResponse", {"error": str(exc)})]


def _handle_execute_hook(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    request = (msg.get("executeHookArgs") or {}).get("request", {})
    hook_case = next((k for k in request if k != "case"), "")
    response = _HOOK_RESPONSES.get(hook_case, {})
    return [finish(base, start, "executeHookResult", {"response": response})]


def _handle_diagnostics(msg: dict, base: dict, start: float, _handlers=None) -> List[Dict[str, Any]]:
    path = (msg.get("diagnosticsArgs") or {}).get("path", "")
    if not os.path.exists(path):
        return [finish(base, start, "diagnosticsResult", {"fileNotFound": {"path": path}})]
    if not os.access(path, os.R_OK):
        return [finish(base, start, "diagnosticsResult", {"permissionDenied": {"path": path}})]
    return [finish(base, start, "diagnosticsResult", {"success": {"path": path, "totalDiagnostics": 0}})]


def _handle_request_context(base: dict, start: float) -> List[Dict[str, Any]]:
    shell = "powershell" if platform.system().lower() == "windows" else "bash"
    os_name = "windows" if platform.system().lower() == "windows" else platform.system().lower()
    return [finish(base, start, "requestContextResult", {"success": {"requestContext": {
        "env": {"operatingSystem": os_name, "defaultShell": shell},
    }}})]


RUN_HANDLERS: Dict[str, ExecFn] = {
    "mcpArgs": _handle_mcp,
    "shellArgs": _handle_shell,
    "shellStreamArgs": _handle_shell,
    "piBashArgs": _handle_pi_bash,
    "miniSweAgentBashArgs": _handle_mini_swe_bash,
    "fetchArgs": lambda m, b, s, _h=None: _fetch_url(m, b, s, "fetchArgs", "fetchResult"),
    "webFetchArgs": lambda m, b, s, _h=None: _fetch_url(m, b, s, "webFetchArgs", "webFetchResult"),
    "gitDiffRequest": _handle_git_diff,
    "executeHookArgs": _handle_execute_hook,
    "diagnosticsArgs": _handle_diagnostics,
}
