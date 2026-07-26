from __future__ import annotations

"""消息历史预处理：将 reasoning/thinking 嵌入 assistant 内容供 inject_fncall 构建。"""

import json
from typing import Any, Dict, List

from echotools.exec.fncall.protocols.entml_thinking_history import (
    apply_thinking_history_policy,
    extract_reasoning_text,
)

_THINKING_OPEN = "<entml:thinking>"
_THINKING_CLOSE = "</entml:thinking>"


def _embed_reasoning_in_content(content: Any, reasoning: str) -> Any:
    block = f"{_THINKING_OPEN}\n{reasoning.strip()}\n{_THINKING_CLOSE}"
    if isinstance(content, str):
        if content.strip():
            return f"{block}\n\n{content}"
        return block
    if isinstance(content, list):
        return [{"type": "text", "text": block}, *content]
    if content is None:
        return block
    text = str(content).strip()
    return f"{block}\n\n{text}" if text else block


def embed_reasoning_in_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """保留历史 assistant 的 reasoning/thinking，写入 entml:thinking 块。"""
    normalized = apply_thinking_history_policy(messages, include=True)
    out: List[Dict[str, Any]] = []
    for msg in normalized:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        reasoning = extract_reasoning_text(msg)
        if not reasoning:
            out.append(msg)
            continue
        new_msg = dict(msg)
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            new_msg.pop(key, None)
        content = new_msg.get("content")
        if isinstance(content, list):
            # 去掉 content 数组里原生 thinking 块，避免重复
            kept = [
                b for b in content
                if not (isinstance(b, dict) and str(b.get("type", "")).lower() in (
                    "thinking", "reasoning", "redacted_thinking"
                ))
            ]
            new_msg["content"] = _embed_reasoning_in_content(kept, reasoning)
        else:
            new_msg["content"] = _embed_reasoning_in_content(content, reasoning)
        out.append(new_msg)
    return out


def anthropic_thinking_to_text(block: Dict[str, Any]) -> str:
    """Anthropic thinking 块 → 纯文本。"""
    val = block.get("thinking") or block.get("data") or ""
    return str(val).strip()


def anthropic_content_block_to_text(block: Dict[str, Any]) -> str:
    btype = str(block.get("type", "")).lower()
    if btype in ("text", "") and "text" in block:
        return str(block.get("text") or "")
    if btype == "thinking":
        return anthropic_thinking_to_text(block)
    if btype == "reasoning":
        return str(block.get("text") or block.get("reasoning") or "")
    return ""


def merge_anthropic_assistant_blocks(content: List[Any]) -> str:
    """将 Anthropic assistant 内容块（含 thinking）合并为带 entml 标记的文本。"""
    parts: List[str] = []
    thinking_parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = str(block.get("type", "")).lower()
        if btype in ("thinking", "reasoning", "redacted_thinking"):
            text = anthropic_content_block_to_text(block)
            if text:
                thinking_parts.append(text)
        elif btype == "text" or "text" in block:
            text = str(block.get("text") or "")
            if text:
                parts.append(text)
    chunks: List[str] = []
    if thinking_parts:
        chunks.append(f"{_THINKING_OPEN}\n" + "\n".join(thinking_parts) + f"\n{_THINKING_CLOSE}")
    chunks.extend(parts)
    return "\n\n".join(c for c in chunks if c)
