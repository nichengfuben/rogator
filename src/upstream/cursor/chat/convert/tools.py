from __future__ import annotations

"""MCP / OpenAI 工具名转换。"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from upstream.cursor.chat.tool_ids import normalize_tool_call_id


def split_mcp_tool_identity(name: str) -> Tuple[str, str, str]:
    """解析 ``mcp__<provider>__<tool>`` → (qualified_name, providerIdentifier, toolName)。"""
    name_s = str(name)
    if name_s.startswith("mcp__"):
        parts = name_s.split("__")
        if len(parts) >= 3 and parts[0] == "mcp" and parts[1]:
            provider = parts[1]
            tool_name = "__".join(parts[2:])
            if tool_name:
                return name_s, provider, tool_name
    return name_s, "", name_s


def strip_mcp_prefix(name: str) -> str:
    """请求者侧：去掉 ``mcp__`` 前缀（``mcp__a__b`` → ``a__b``）。"""
    s = str(name or "").strip()
    if s.startswith("mcp__"):
        return s[5:]
    return s


def restore_mcp_prefix_for_cursor(
    name: str,
    originals: Optional[Set[str]] = None,
) -> str:
    """请求者回传的工具名 → 注入 Cursor 时的名字。"""
    s = str(name or "").strip()
    if not s:
        return s
    if s.startswith("mcp__"):
        return s
    if originals:
        if s in originals:
            return s
        prefixed = f"mcp__{s}"
        if prefixed in originals:
            return prefixed
        for orig in originals:
            o = str(orig)
            if not o.startswith("mcp__"):
                continue
            if strip_mcp_prefix(o) == s:
                return o
            _, _, short = split_mcp_tool_identity(o)
            if short == s:
                return o
    if "__" in s:
        return f"mcp__{s}"
    return s


def _tool_name_match_keys(name: str) -> List[str]:
    """清单匹配用：原名 / 去 mcp__ / 短名。"""
    s = str(name or "").strip()
    if not s:
        return []
    keys = [s]
    stripped = strip_mcp_prefix(s)
    if stripped and stripped not in keys:
        keys.append(stripped)
    _, _, short = split_mcp_tool_identity(s)
    if short and short not in keys:
        keys.append(short)
    return keys


def original_tool_names(tools: Optional[List[Dict[str, Any]]]) -> Set[str]:
    names: Set[str] = set()
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if name:
            names.add(str(name))
    return names


def rewrite_tool_call_for_openai(
    tool_call: Dict[str, Any],
    *,
    allowed_originals: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """转发给请求者的工具名（mcp__ 去前缀；非 mcp 仅清单内）。"""
    if not tool_call:
        return None
    fn = dict(tool_call.get("function") or {})
    raw_name = str(fn.get("name") or "").strip()
    if not raw_name:
        return None
    is_mcp = raw_name.startswith("mcp__")
    if is_mcp:
        if allowed_originals is not None:
            if not any(k in allowed_originals for k in _tool_name_match_keys(raw_name)):
                return None
        out_name = strip_mcp_prefix(raw_name)
    else:
        if allowed_originals is not None and raw_name not in allowed_originals:
            return None
        out_name = raw_name
    if not out_name:
        return None
    out = dict(tool_call)
    out["function"] = {**fn, "name": out_name}
    out["type"] = out.get("type") or "function"
    if "id" in out:
        out["id"] = normalize_tool_call_id(out.get("id")) or out.get("id")
    return out


def openai_tools_to_mcp(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """转为 AgentRunRequest.mcpTools（对齐 McpToolDefinition；不改写名称）。"""
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        qualified, provider, tool_name = split_mcp_tool_identity(str(name))
        entry: Dict[str, Any] = {
            "name": qualified,
            "toolName": tool_name,
            "description": fn.get("description") or "",
            "inputSchemaJson": json.dumps(
                fn.get("parameters") or {"type": "object", "properties": {}},
                ensure_ascii=False,
            ),
        }
        if provider:
            entry["providerIdentifier"] = provider
        out.append(entry)
    return out
