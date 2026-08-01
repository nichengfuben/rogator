#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live smoke: 并行 MCP 工具调用。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from upstream.cursor.auth.store import get_access_token
from upstream.cursor.chat.openai import stream_openai_chat
from upstream.cursor.client import CursorClient


class _SmokeClient:
    def __init__(self) -> None:
        self._conversation_id: Optional[str] = None
        self._workspace = "X:/Project/Public/Qwen"
        self._inner = CursorClient()

    async def ensure_token(self) -> None:
        await self._inner.ensure_token()


async def main() -> int:
    if not get_access_token():
        print("FAIL: no token")
        return 2

    tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp__smoke__echo_a",
                "description": "Echo channel A. Call in parallel with echo_b when asked.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp__smoke__echo_b",
                "description": "Echo channel B. Call in parallel with echo_a when asked.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "When the user asks for parallel echoes, you MUST call BOTH "
                "mcp__smoke__echo_a and mcp__smoke__echo_b in the SAME turn "
                "(parallel tool calls). Do not answer in prose first."
            ),
        },
        {
            "role": "user",
            "content": (
                "In one turn, call mcp__smoke__echo_a with text=\"A1\" AND "
                "mcp__smoke__echo_b with text=\"B2\" in parallel."
            ),
        },
    ]

    client = _SmokeClient()
    types: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    thinking = []
    t0 = time.time()
    async for ev in stream_openai_chat(
        None, client, messages, "composer-2.5-fast", tools, "smoke-parallel",
    ):
        types.append(str(ev.get("type")))
        if ev.get("type") == "thinking":
            thinking.append(str(ev.get("content") or ""))
        if ev.get("type") == "tool_call":
            tool_calls.append(ev.get("tool_call") or {})

    elapsed = time.time() - t0
    names = [((tc.get("function") or {}).get("name") or "") for tc in tool_calls]
    print(f"elapsed={elapsed:.2f}s types={types}")
    print("thinking:", "".join(thinking)[:300].replace("\n", " "))
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        print(f"tool[{i}]: {fn.get('name')!r} args={fn.get('arguments')!r} id={tc.get('id')!r}")

    ok = (
        len(tool_calls) >= 2
        and "mcp__smoke__echo_a" in names
        and "mcp__smoke__echo_b" in names
    )
    print("VERDICT:", "PASS" if ok else "FAIL", f"n={len(tool_calls)} names={names}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
