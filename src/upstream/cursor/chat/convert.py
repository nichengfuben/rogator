from __future__ import annotations

"""OpenAI ↔ Cursor 消息/模型转换。"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from upstream.cursor.setup.config import load_cursor_upstream_config

# 工具名由上游原样传入（不改写 mcp__）。IMPORTANT 约束「仅清单内工具」，
# 不强制 mcp__ 前缀——否则 Kimi 等客户端的 Shell/Read 会被模型当成不可用。
IMPORTANT_MCP_TOOLS_ONLY = """IMPORTANT:
You may use ONLY the tools provided in this request's tool list, calling them by their exact names.
Treat every other tool as unavailable — including any Cursor built-in shell, file, terminal, browser, editor, search, or tools not in that list.
When you need to act, call a tool from the provided list. Never invent a tool name."""

IMPORTANT_NO_TOOLS = """IMPORTANT:
No tools are available in this turn.
You must not call any tool — including built-in shell, file, terminal, browser, editor, search, or MCP tools.
Do not attempt tool use. Respond with plain assistant text only."""


def _message_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p and p.get("text") is not None:
                    parts.append(str(p.get("text") or ""))
                elif "content" in p and isinstance(p.get("content"), str):
                    parts.append(str(p.get("content") or ""))
                else:
                    parts.append(json.dumps(p, ensure_ascii=False))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    if isinstance(content, (dict, bool, int, float)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def extract_system_texts(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for msg in messages or []:
        if (msg.get("role") or "") != "system":
            continue
        text = _message_text(msg)
        if text:
            out.append(text)
    return out


def build_custom_system_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> str:
    """IMPORTANT 块置顶，其后拼接全部 system 消息。始终返回非空字符串。"""
    has_tools = bool(tools)
    preamble = IMPORTANT_MCP_TOOLS_ONLY if has_tools else IMPORTANT_NO_TOOLS
    systems = extract_system_texts(messages)
    if not systems:
        return preamble
    return preamble + "\n\n" + "\n\n".join(systems)


def prepend_system_to_prompt(system_text: str, prompt: str) -> str:
    """兼容旧路径：system 前缀 + body。

    新路径请用 ``build_cursor_turn``：system 进 history，UserMessage 只承载本轮用户话。
    """
    block = (system_text or "").strip()
    if not block:
        return prompt or ""
    wrapped = f"<system>\n{block}\n</system>"
    body = (prompt or "").strip()
    if not body:
        return wrapped
    return f"{wrapped}\n\n{body}"


def messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if not text and role != "assistant":
            continue
        if role == "system":
            continue
        if role == "assistant":
            if text:
                parts.append(f"<assistant>\n{text}\n</assistant>")
        elif role == "tool":
            parts.append(f"<tool_result name=\"{msg.get('name', '')}\">\n{text}\n</tool_result>")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def messages_to_cursor_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if role == "system":
            continue
        if role == "user":
            if not text:
                continue
            history.append({"user": {"content": [{"text": {"text": text}}]}})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if text:
                blocks.append({"text": {"text": text}})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({
                    "toolCall": {
                        "toolCallId": tc.get("id") or "",
                        "toolName": fn.get("name") or "",
                        "argsJson": fn.get("arguments") or "{}",
                    },
                })
            if blocks:
                history.append({"assistant": {"content": blocks}})
        elif role == "tool":
            history.append({
                "tool": {
                    "toolCallId": msg.get("tool_call_id") or "",
                    "toolName": msg.get("name") or "",
                    "content": [{"text": {"text": text}}],
                    "isError": bool(msg.get("is_error") or msg.get("isError")),
                },
            })
    return history


_TOOL_CONTINUE_PROMPT = "Continue based on the tool results."


def _embed_tool_results_prompt(messages: List[Dict[str, Any]]) -> str:
    """续轮 user 文本：固定续写句 + 明文 tool_result，避免仅靠 history 时模型看不见回执。"""
    blocks: List[str] = []
    for msg in messages or []:
        if (msg.get("role") or "") != "tool":
            continue
        name = str(msg.get("name") or "")
        text = _message_text(msg)
        blocks.append(f'<tool_result name="{name}">\n{text}\n</tool_result>')
    if not blocks:
        return _TOOL_CONTINUE_PROMPT
    return _TOOL_CONTINUE_PROMPT + "\n\n" + "\n\n".join(blocks)


def split_prompt_and_history(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """拆出本轮 user 文本 + Cursor conversationHistory。"""
    if not messages:
        return "", []
    last = messages[-1]
    last_role = last.get("role") or ""
    if last_role == "user" and _message_text(last):
        return _message_text(last), messages_to_cursor_history(messages[:-1])

    # 末条 user 但内容为空：回退到上一条非空 user，避免 UserMessage.text 为空
    if last_role == "user" and not _message_text(last):
        for msg in reversed(messages[:-1]):
            if (msg.get("role") or "") == "user" and _message_text(msg):
                idx = messages.index(msg)
                return _message_text(msg), messages_to_cursor_history(messages[:idx] + messages[idx + 1 : -1])
        return "", messages_to_cursor_history(messages[:-1])

    if last_role == "tool" or (
        last_role != "user"
        and any((m.get("role") or "") == "tool" for m in messages)
    ):
        return _embed_tool_results_prompt(messages), messages_to_cursor_history(messages)

    return messages_to_prompt(messages), []


def build_cursor_turn(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """构造 Cursor 本轮 UserMessage.text + conversationHistory。

    关键点：UserMessage.text **只放本轮用户话**（加 ``<user_query>``），
    IMPORTANT + 客户端 system 放进 history 首条，避免 30k system 淹没用户意图
    （模型会回 ``The user did not provide a specific query``）。
    """
    prompt, history = split_prompt_and_history(messages)
    system = build_custom_system_prompt(messages, tools).strip()
    hist: List[Dict[str, Any]] = list(history or [])
    if system:
        hist.insert(0, {
            "user": {"content": [{"text": {"text": f"<system>\n{system}\n</system>"}}]},
        })
    body = (prompt or "").strip()
    if body.startswith(_TOOL_CONTINUE_PROMPT):
        send_text = body
    elif body:
        send_text = f"<user_query>\n{body}\n</user_query>"
    else:
        send_text = (
            "<user_query>\n"
            "(No user text was found in this turn; ask the user what they need.)\n"
            "</user_query>"
        )
    return send_text, hist


def map_model(model: Optional[str]) -> str:
    """将 Rogator 内键映射为 Cursor Agent ``modelId``（保留 effort 后缀）。"""
    cfg = load_cursor_upstream_config()
    cursor_cfg = cfg.get("cursor") or {}
    models_cfg = cfg.get("models") or {}
    default = str(models_cfg.get("default") or cursor_cfg.get("default_model") or "composer-2.5-fast")
    if not model:
        return default

    from upstream.cursor.models.identity import is_valid_model_id

    if is_valid_model_id(model):
        return model

    mapping = models_cfg.get("mapping") or {}
    if isinstance(mapping, dict) and model in mapping:
        mapped = str(mapping[model])
        return mapped if is_valid_model_id(mapped) else model
    return model


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
    """过滤空名；有工具清单时只转发清单内工具（名称不改写）。"""
    if not tool_call:
        return None
    fn = dict(tool_call.get("function") or {})
    raw_name = str(fn.get("name") or "").strip()
    if not raw_name:
        return None
    if allowed_originals is not None and raw_name not in allowed_originals:
        return None
    out = dict(tool_call)
    out["function"] = {**fn, "name": raw_name}
    out["type"] = out.get("type") or "function"
    return out
