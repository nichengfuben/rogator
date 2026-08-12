from __future__ import annotations

"""内置 ToolCall 提取 / proto Value 还原 / MCP args JSON。"""

import json
import uuid
from typing import Any, Dict, Optional, Tuple

from upstream.cursor.chat.tool_ids import normalize_tool_call_id

# interactionUpdate.toolCall 的 camelCase oneof → 请求者可见工具名
INTERACTION_ONEOF_TO_NAME: Dict[str, str] = {
    "shellToolCall": "Shell",
    "readToolCall": "Read",
    "grepToolCall": "Grep",
    "editToolCall": "Edit",
    "deleteToolCall": "Delete",
    "globToolCall": "Glob",
    "lsToolCall": "LS",
    "webSearchToolCall": "WebSearch",
    "webFetchToolCall": "WebFetch",
    "awaitToolCall": "Await",
    "semSearchToolCall": "SemSearch",
    "readLintsToolCall": "ReadLints",
    "updateTodosToolCall": "TodoWrite",
    "readTodosToolCall": "TodoRead",
    "taskToolCall": "Agent",
    "fetchToolCall": "WebFetch",
    "askQuestionToolCall": "AskQuestion",
    "generateImageToolCall": "GenerateImage",
    "switchModeToolCall": "SwitchMode",
}

# execServerMessage 参数字段 → (OpenAI 名, execClientMessage 回执字段)
EXEC_ARGS_TO_OPENAI: Dict[str, Tuple[str, str]] = {
    "shellArgs": ("Shell", "shellResult"),
    "shellStreamArgs": ("Shell", "shellResult"),
    "readArgs": ("Read", "readResult"),
    "writeArgs": ("Write", "writeResult"),
    "grepArgs": ("Grep", "grepResult"),
    "deleteArgs": ("Delete", "deleteResult"),
    "lsArgs": ("LS", "lsResult"),
    "globToolArgs": ("Glob", "globToolResult"),
    "fetchArgs": ("WebFetch", "fetchResult"),
    "webFetchArgs": ("WebFetch", "webFetchResult"),
    "piBashArgs": ("Shell", "piBashResult"),
    "piReadArgs": ("Read", "piReadResult"),
    "piWriteArgs": ("Write", "piWriteResult"),
    "piGrepArgs": ("Grep", "piGrepResult"),
    "piLsArgs": ("LS", "piLsResult"),
    "piEditArgs": ("Edit", "piEditResult"),
    "piFindArgs": ("Glob", "piFindResult"),
    "diagnosticsArgs": ("ReadLints", "diagnosticsResult"),
}


def is_mcp_tool_name(name: str) -> bool:
    return str(name or "").startswith("mcp__")


def unwrap_proto_value(value: Any) -> Any:
    """把 google.protobuf.Value / Struct 的 JSON 形态还原成普通 Python 值。"""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value and len(value) == 1:
        return value["stringValue"]
    if "numberValue" in value and len(value) == 1:
        return value["numberValue"]
    if "boolValue" in value and len(value) == 1:
        return value["boolValue"]
    if "nullValue" in value and len(value) == 1:
        return None
    if "structValue" in value:
        return unwrap_proto_struct(value["structValue"])
    if "listValue" in value:
        items = (value["listValue"] or {}).get("values") or []
        return [unwrap_proto_value(v) for v in items]
    if "fields" in value and all(isinstance(k, str) for k in value.keys()):
        if set(value.keys()) <= {"fields"} or (
            "fields" in value and not any(k.endswith("Value") for k in value)
        ):
            return unwrap_proto_struct(value)
    return {k: unwrap_proto_value(v) for k, v in value.items()}


def unwrap_proto_struct(struct: Any) -> Any:
    if not isinstance(struct, dict):
        return struct
    fields = struct.get("fields")
    if isinstance(fields, dict):
        return {k: unwrap_proto_value(v) for k, v in fields.items()}
    return {k: unwrap_proto_value(v) for k, v in struct.items()}


def mcp_args_to_json(args_obj: Any) -> str:
    if not isinstance(args_obj, dict):
        return "{}"
    plain = {k: unwrap_proto_value(v) for k, v in args_obj.items()}
    return json.dumps(plain, ensure_ascii=False)


def _openai_from_history_style(tc: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    if not (tc.get("toolName") or tc.get("argsJson")):
        return None
    name = str(tc.get("toolName") or "").strip()
    if not name:
        return None
    return {
        "id": normalize_tool_call_id(tc.get("toolCallId") or fallback_id),
        "type": "function",
        "function": {"name": name, "arguments": tc.get("argsJson") or "{}"},
    }


def _openai_from_mcp_call(tc: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    from upstream.cursor.chat.convert import strip_mcp_prefix

    mcp = tc.get("mcpToolCall") or {}
    args = mcp.get("args") or mcp
    if not isinstance(args, dict):
        return None
    if not (args.get("name") or args.get("toolName") or args.get("providerIdentifier")):
        return None
    provider = str(args.get("providerIdentifier") or "").strip()
    short = str(args.get("toolName") or "").strip()
    qualified = str(args.get("name") or "").strip()
    if not qualified and provider and short:
        qualified = f"mcp__{provider}__{short}"
    cursor_name = qualified or short or provider
    if not cursor_name:
        return None
    openai_name = (
        strip_mcp_prefix(cursor_name) if cursor_name.startswith("mcp__") else cursor_name
    )
    raw_args = args.get("args") or {}
    args_json = mcp_args_to_json(raw_args) if isinstance(raw_args, dict) else "{}"
    return {
        "id": normalize_tool_call_id(args.get("toolCallId") or fallback_id),
        "type": "function",
        "function": {"name": openai_name, "arguments": args_json},
    }


def _openai_from_builtin_oneof(tc: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    for key, name in INTERACTION_ONEOF_TO_NAME.items():
        if key not in tc:
            continue
        block = tc.get(key) or {}
        if not isinstance(block, dict):
            continue
        raw_args = block.get("args") or {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        tid = normalize_tool_call_id(
            raw_args.get("toolCallId") or block.get("toolCallId") or fallback_id
        )
        plain = {k: unwrap_proto_value(v) for k, v in raw_args.items() if k != "toolCallId"}
        return {
            "id": tid or str(fallback_id or uuid.uuid4()),
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(plain, ensure_ascii=False)},
        }
    return None


def openai_tool_from_agent_tool_call(tc: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    """从 interactionUpdate.toolCall（ToolCall oneof）提取 OpenAI function tool_call。"""
    if not tc:
        return None
    return (
        _openai_from_history_style(tc, fallback_id)
        or _openai_from_mcp_call(tc, fallback_id)
        or _openai_from_builtin_oneof(tc, fallback_id)
    )


def builtin_result_payload(result_field: str, text: str, *, is_error: bool = False) -> Dict[str, Any]:
    """把客户端 tool 文本回执压成 Cursor execClientMessage 载荷。"""
    body = text or ""
    if result_field == "shellResult":
        return {
            "stdout": "" if is_error else body,
            "stderr": body if is_error else "",
            "exitCode": 1 if is_error else 0,
        }
    if result_field in ("readResult", "piReadResult", "redactedReadResult"):
        if is_error:
            return {"error": {"message": body}}
        return {"success": {"content": body, "lineCount": body.count("\n") + (1 if body else 0)}}
    if result_field in ("writeResult", "deleteResult", "piWriteResult", "piEditResult"):
        if is_error:
            return {"error": {"message": body} if "Result" in result_field else {"error": body}}
        return {"success": {}}
    if result_field in ("grepResult", "piGrepResult"):
        if is_error:
            return {"error": {"message": body}}
        return {"success": {"matches": [], "output": body}}
    if result_field in ("lsResult", "piLsResult"):
        if is_error:
            return {"error": {"message": body}}
        return {"success": {"entries": [], "output": body}}
    if result_field in ("globToolResult", "piFindResult"):
        if is_error:
            return {"error": {"error": body}}
        return {"success": {"files": [], "output": body}}
    if result_field in ("fetchResult", "webFetchResult"):
        if is_error:
            return {"error": {"error": body}}
        return {"success": {"body": body}}
    if result_field == "piBashResult":
        if is_error:
            return {"error": {"error": body, "exitCode": 1}}
        return {"success": {"output": body}}
    if result_field == "diagnosticsResult":
        if is_error:
            return {"error": {"message": body}}
        return {"success": {"path": "", "totalDiagnostics": 0, "output": body}}
    if is_error:
        return {"error": {"message": body}}
    return {"success": {"content": [{"text": {"text": body}}]}}


def exec_tool_openai_view(exec_msg: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any], str]]:
    """返回 (openai_name, result_field, args_dict, tool_call_id_hint) 或 None。"""
    for key, (name, result_field) in EXEC_ARGS_TO_OPENAI.items():
        if key not in exec_msg:
            continue
        raw = exec_msg.get(key) or {}
        if not isinstance(raw, dict):
            raw = {}
        tid = str(raw.get("toolCallId") or exec_msg.get("execId") or exec_msg.get("id") or "").strip()
        return name, result_field, dict(raw), tid
    return None
