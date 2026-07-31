#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比工具续轮 text 策略：真调模型看哪种能看见 TOKEN。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.auth.store import get_access_token
from upstream.cursor.chat.convert import messages_to_cursor_history
from upstream.cursor.chat.openai import stream_openai_chat
from upstream.cursor.client import CursorClient
from upstream.cursor.stream.agent import stream_cursor_agent
from upstream.cursor.auth.store import get_token_bundle
from upstream.cursor.chat.convert import (
    build_custom_system_prompt,
    build_prepend_user_messages,
    openai_tools_to_mcp,
)
from upstream.cursor.stream.exec.tool_filter import tool_filter_for_openai


ECHO = {
    "type": "function",
    "function": {
        "name": "mcp__smoke__echo",
        "description": "Echo text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


class _C:
    def __init__(self) -> None:
        self._conversation_id = None
        self._workspace = "X:/Project/Public/Qwen"
        self._inner = CursorClient()

    async def ensure_token(self) -> None:
        await self._inner.ensure_token()


async def run_with_text(
    *,
    label: str,
    text: str,
    messages_for_history: List[Dict[str, Any]],
) -> str:
    history = messages_to_cursor_history(messages_for_history)
    system = build_custom_system_prompt(messages_for_history, [ECHO]).strip()
    prepend = build_prepend_user_messages(system)
    allowed, exclude = tool_filter_for_openai(True)
    mcp = openai_tools_to_mcp([ECHO])
    print(f"\n=== {label} ===")
    print("text:", repr(text))
    print("history:", len(history), json.dumps(history, ensure_ascii=False)[:240])

    client = _C()
    await client.ensure_token()
    answer: List[str] = []
    thinking: List[str] = []
    t0 = time.time()
    async for ev in stream_cursor_agent(
        prompt=text,
        model="composer-2.5-fast",
        token=get_token_bundle(),
        conversation_id=None,
        conversation_history=history,
        workspace=client._workspace,
        prepend_user_messages=prepend,
        mcp_tools=mcp,
        allowed_tools=allowed,
        exclude_tools=exclude,
        defer_mcp=True,
    ):
        if ev.type == "text":
            answer.append(ev.text)
        elif ev.type == "thinking":
            thinking.append(ev.text)
        elif ev.type == "error":
            print("ERROR:", ev.error)
            return ""
    ans = "".join(answer)
    th = "".join(thinking)
    print(f"elapsed={time.time()-t0:.1f}s")
    print("thinking:", repr(th[:220]))
    print("answer:", repr(ans[:300]))
    saw = "TOKEN-7788" in ans or "7788" in ans or "TOKEN-7788" in th
    print("saw_token:", saw)
    return ans


async def main() -> int:
    if not get_access_token():
        print("FAIL: no token")
        return 2

    base = [
        {"role": "user", "content": "Call echo with TOKEN-7788"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "mcp__smoke__echo",
                    "arguments": '{"text":"TOKEN-7788"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "mcp__smoke__echo",
            "content": "echo-result:TOKEN-7788",
        },
    ]

    strategies: List[Tuple[str, str]] = [
        ("continue_working", "Continue working on the current task."),
        ("last_user", "Call echo with TOKEN-7788"),
        (
            "tool_in_text",
            "Tool result for mcp__smoke__echo: echo-result:TOKEN-7788\n"
            "Based on that tool result, reply with the echoed token only.",
        ),
    ]

    results = {}
    for name, text in strategies:
        ans = await run_with_text(label=name, text=text, messages_for_history=base)
        results[name] = ("TOKEN-7788" in ans) or ("7788" in ans)

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    # 至少一种策略能看见 token 才算摸清方向
    if any(results.values()):
        print("AT_LEAST_ONE_PASS")
        return 0
    print("ALL_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
