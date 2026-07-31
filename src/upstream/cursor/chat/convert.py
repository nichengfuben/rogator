from __future__ import annotations

"""OpenAI ↔ Cursor 消息/模型转换。"""

import json
import uuid
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
                # OpenAI / Anthropic / 部分客户端：{"type":"text","text":"..."}
                text = p.get("text")
                if text is not None and str(text).strip():
                    parts.append(str(text))
                    continue
                if p.get("type") in ("text", "input_text") and p.get("text") is not None:
                    parts.append(str(p.get("text") or ""))
                    continue
                if isinstance(p.get("content"), str) and str(p.get("content") or "").strip():
                    parts.append(str(p.get("content") or ""))
                    continue
                # 跳过纯图片/附件块，避免把整段 JSON 当成「用户话」淹没真实文本
                if p.get("type") in ("image_url", "image", "input_image", "file", "input_file"):
                    continue
                if "text" in p:
                    parts.append(str(p.get("text") or ""))
                elif "content" in p and isinstance(p.get("content"), str):
                    parts.append(str(p.get("content") or ""))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    if isinstance(content, (dict, bool, int, float)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _user_text(msg: Dict[str, Any]) -> str:
    """用户可见文本（去首尾空白）；空白视为空。"""
    return _message_text(msg).strip()


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

    新路径请用 ``build_cursor_turn``：system → prependUserMessages，UserMessage 只承载本轮用户明文。
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
            # 工具回执只走 conversationHistory；此处不拼 XML
            continue
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _tool_name_index(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """tool_call_id → toolName（OpenAI tool 消息常缺 name）。"""
    out: Dict[str, str] = {}
    for msg in messages or []:
        if (msg.get("role") or "") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            tid = str(tc.get("id") or "").strip()
            if not tid:
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "").strip()
            if name:
                out[tid] = name
    return out


def messages_to_cursor_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对齐 agent.v1.ConversationHistory（user / assistant / tool）。"""
    history: List[Dict[str, Any]] = []
    name_by_id = _tool_name_index(messages)
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if role == "system":
            continue
        if role == "user":
            if not text.strip():
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
            tid = str(msg.get("tool_call_id") or "")
            tname = str(msg.get("name") or "").strip() or name_by_id.get(tid, "")
            history.append({
                "tool": {
                    "toolCallId": tid,
                    "toolName": tname,
                    "content": [{"text": {"text": text}}],
                    "isError": bool(msg.get("is_error") or msg.get("isError")),
                },
            })
    return history


# 工具续轮：UserMessage.text 仅短续写；回执在 conversationHistory.tool
_TOOL_CONTINUE_PROMPT = "Continue based on the tool results."


def split_prompt_and_history(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """拆出本轮 UserMessage.text + Cursor conversationHistory。"""
    if not messages:
        return "", []
    last = messages[-1]
    last_role = last.get("role") or ""
    last_text = _user_text(last) if last_role == "user" else ""
    if last_role == "user" and last_text:
        return last_text, messages_to_cursor_history(messages[:-1])

    # 末条 user 但内容为空/空白：回退到上一条非空 user，避免 UserMessage.text 为空
    if last_role == "user" and not last_text:
        for msg in reversed(messages[:-1]):
            if (msg.get("role") or "") == "user" and _user_text(msg):
                idx = messages.index(msg)
                return _user_text(msg), messages_to_cursor_history(messages[:idx] + messages[idx + 1 : -1])
        return "", messages_to_cursor_history(messages[:-1])

    if last_role == "tool" or (
        last_role != "user"
        and any((m.get("role") or "") == "tool" for m in messages)
    ):
        # 官方：回执只进 history；text 不塞 <tool_result> XML
        return _TOOL_CONTINUE_PROMPT, messages_to_cursor_history(messages)

    return messages_to_prompt(messages), []


def build_prepend_user_messages(system_text: str) -> List[Dict[str, Any]]:
    """对齐 agent.v1.UserMessageAction.prepend_user_messages（非 customSystemPrompt）。

    customSystemPrompt 在 agentn 上会变成 ``--system-prompt`` 并报 unknown option；
    正规注入口是 prependUserMessages。
    """
    block = (system_text or "").strip()
    if not block:
        return []
    return [{
        "text": block,
        "messageId": str(uuid.uuid4()),
        "mode": 1,
    }]


def build_cursor_turn(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构造 Cursor 本轮 UserMessage.text + conversationHistory + prependUserMessages。

    对齐 agent.v1：
    - ``UserMessage.text`` = 本轮用户明文；工具续轮仅为短续写句（不塞 tool XML）
    - IMPORTANT + 客户端 system → ``prependUserMessages``
    - 工具回执 → ``conversationHistory`` 的 tool 消息（补全 toolName）
    """
    prompt, history = split_prompt_and_history(messages)
    system = build_custom_system_prompt(messages, tools).strip()
    prepend = build_prepend_user_messages(system)
    body = (prompt or "").strip()
    if body:
        send_text = body
    else:
        send_text = "(No user text was found in this turn; ask the user what they need.)"
    return send_text, list(history or []), prepend


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
