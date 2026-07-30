from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from upstream.cursor.stream.exec.common import base_msg, finish, tool_type
from upstream.cursor.stream.exec.fs import FS_HANDLERS
from upstream.cursor.stream.exec.run import RUN_HANDLERS, _handle_request_context
from upstream.cursor.stream.exec.stubs import is_stub_tool, stub_tool


def execute_tool(
    exec_msg: Dict[str, Any],
    tool_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    *,
    defer_mcp: bool = False,
) -> List[Dict[str, Any]]:
    """执行 execServerMessage 并返回 execClientMessage 内层 payload 列表。"""
    start = time.time()
    base = base_msg(exec_msg)
    tool = tool_type(exec_msg)
    if not tool:
        return [finish(base, start, "shellResult", {"stdout": "", "stderr": "Unknown exec type", "exitCode": -1})]
    if tool == "requestContextArgs":
        return _handle_request_context(base, start)
    if tool == "mcpArgs":
        return RUN_HANDLERS["mcpArgs"](exec_msg, base, start, tool_handlers, defer_mcp=defer_mcp)
    if tool in RUN_HANDLERS:
        return RUN_HANDLERS[tool](exec_msg, base, start, tool_handlers)
    if tool in FS_HANDLERS:
        return FS_HANDLERS[tool](exec_msg, base, start)
    if is_stub_tool(tool):
        return [stub_tool(exec_msg, base, start, tool)]
    return [finish(base, start, "shellResult", {"stdout": "", "stderr": f"Unsupported: {tool}", "exitCode": -1})]


def extract_tool_result_text(result: Dict[str, Any]) -> str:
    """从 exec 结果提取 MCP 文本，供 tool_result 事件使用。"""
    mr = result.get("mcpResult")
    if not isinstance(mr, dict):
        return ""
    success = mr.get("success")
    if not isinstance(success, dict):
        return ""
    for item in success.get("content") or []:
        if isinstance(item, dict) and "text" in item:
            text = item["text"]
            if isinstance(text, dict):
                return str(text.get("text") or "")
    return ""
