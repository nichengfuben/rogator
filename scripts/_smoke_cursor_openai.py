#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live smoke: Cursor upstream 思考 / 回复 / 工具调用。"""

from __future__ import annotations

import asyncio
import json
import sys
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


async def _collect(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "composer-2.5-fast",
    label: str,
) -> Dict[str, Any]:
    client = _SmokeClient()
    events: List[Dict[str, Any]] = []
    thinking_parts: List[str] = []
    answer_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    t0 = time.time()
    async for ev in stream_openai_chat(
        state=None,
        client=client,
        messages=messages,
        model=model,
        tools=tools,
        req_id=f"smoke-{label}",
    ):
        events.append(ev)
        et = ev.get("type")
        if et == "thinking":
            thinking_parts.append(str(ev.get("content") or ""))
        elif et == "answer":
            answer_parts.append(str(ev.get("content") or ""))
        elif et == "tool_call":
            tool_calls.append(ev.get("tool_call") or {})
    elapsed = time.time() - t0
    return {
        "label": label,
        "elapsed": round(elapsed, 2),
        "event_types": [e.get("type") for e in events],
        "thinking": "".join(thinking_parts),
        "answer": "".join(answer_parts),
        "tool_calls": tool_calls,
        "n_events": len(events),
    }


def _print_result(r: Dict[str, Any]) -> None:
    print(f"\n===== {r['label']} ({r['elapsed']}s, events={r['n_events']}) =====")
    print("types:", r["event_types"])
    th = r["thinking"]
    if th:
        print("thinking[:400]:", th[:400].replace("\n", " "))
    else:
        print("thinking: <empty>")
    ans = r["answer"]
    if ans:
        print("answer[:400]:", ans[:400].replace("\n", " "))
    else:
        print("answer: <empty>")
    if r["tool_calls"]:
        for i, tc in enumerate(r["tool_calls"]):
            fn = (tc.get("function") or {})
            print(f"tool_call[{i}]: name={fn.get('name')!r} args={fn.get('arguments')!r} id={tc.get('id')!r}")
    else:
        print("tool_calls: <none>")


async def main() -> int:
    if not get_access_token():
        print("FAIL: no access token in persist/cursor/auth.json")
        return 2

    echo_tool = {
        "type": "function",
        "function": {
            "name": "mcp__smoke__echo",
            "description": "Echo back the given text. Use this when the user asks to echo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"},
                },
                "required": ["text"],
            },
        },
    }

    # 1) 无工具：应有正文回复（可能带 thinking）
    r1 = await _collect(
        [
            {"role": "system", "content": "Be brief. Reply in one short sentence."},
            {"role": "user", "content": "Say exactly: hello-cursor-smoke"},
        ],
        tools=None,
        label="no_tools_reply",
    )
    _print_result(r1)

    # 2) 有工具：应发起 mcp__ 工具调用
    r2 = await _collect(
        [
            {"role": "system", "content": "You must use tools when available."},
            {
                "role": "user",
                "content": (
                    "Call the mcp__smoke__echo tool with text=\"ping-42\". "
                    "Do not answer in plain text first; call the tool."
                ),
            },
        ],
        tools=[echo_tool],
        label="with_tools_call",
    )
    _print_result(r2)

    # 3) 工具回灌：应基于结果正常回复
    tool_id = "call_smoke_1"
    if r2["tool_calls"]:
        tool_id = str(r2["tool_calls"][0].get("id") or tool_id)
        tool_name = str((r2["tool_calls"][0].get("function") or {}).get("name") or "mcp__smoke__echo")
        tool_args = str((r2["tool_calls"][0].get("function") or {}).get("arguments") or '{"text":"ping-42"}')
    else:
        tool_name = "mcp__smoke__echo"
        tool_args = '{"text":"ping-42"}'

    r3 = await _collect(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Call echo with ping-42"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": "echo-result:ping-42",
            },
        ],
        tools=[echo_tool],
        label="tool_continue_reply",
    )
    _print_result(r3)

    ok_reply = bool(r1["answer"].strip()) or bool(r1["thinking"].strip())
    # 无工具时至少要有 answer（thinking 单独不够算「回复」）
    ok_reply = bool(r1["answer"].strip())
    ok_tool = any(
        ((tc.get("function") or {}).get("name") or "").startswith("mcp__")
        for tc in r2["tool_calls"]
    )
    ok_continue = bool(r3["answer"].strip()) or bool(r3["tool_calls"])

    print("\n===== VERDICT =====")
    print("thinking_seen:", bool(r1["thinking"] or r2["thinking"] or r3["thinking"]))
    print("reply_ok:", ok_reply, "answer=", repr(r1["answer"][:120]))
    print("tool_call_ok:", ok_tool, "names=", [((t.get("function") or {}).get("name")) for t in r2["tool_calls"]])
    print("continue_ok:", ok_continue, "answer=", repr(r3["answer"][:120]), "tools=", len(r3["tool_calls"]))

    if ok_reply and ok_tool and ok_continue:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("FATAL:", type(exc).__name__, exc)
        raise
