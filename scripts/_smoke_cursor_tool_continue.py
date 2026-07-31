#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live smoke: 同流 park/resume 必须看见 tool result（TOKEN），不得 empty-query 失忆。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from upstream.cursor.auth.store import get_access_token
from upstream.cursor.chat.agent_session import find_parked_by_tool_ids
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
    label: str,
) -> Dict[str, Any]:
    client = _SmokeClient()
    thinking: List[str] = []
    answer: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    t0 = time.time()
    async for ev in stream_openai_chat(
        state=None,
        client=client,
        messages=messages,
        model="composer-2.5-fast",
        tools=tools,
        req_id=f"smoke-{label}",
    ):
        et = ev.get("type")
        if et == "prompt_meta":
            meta = ev
        elif et == "thinking":
            thinking.append(str(ev.get("content") or ""))
        elif et == "answer":
            answer.append(str(ev.get("content") or ""))
        elif et == "tool_call":
            tool_calls.append(ev.get("tool_call") or {})
    return {
        "label": label,
        "elapsed": round(time.time() - t0, 2),
        "thinking": "".join(thinking),
        "answer": "".join(answer),
        "tool_calls": tool_calls,
        "meta": meta,
    }


def _looks_like_empty_query_amnesia(text: str) -> bool:
    lower = (text or "").lower()
    needles = (
        "empty query",
        "你好",
        "需要我帮你做什么",
        "how can i help",
        "what can i help",
        "start of a",
    )
    return any(n in lower for n in needles)


async def main() -> int:
    if not get_access_token():
        print("FAIL: no access token")
        return 2

    echo_tool = {
        "type": "function",
        "function": {
            "name": "mcp__smoke__echo",
            "description": "Echo back the given text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }

    r_call = await _collect(
        [
            {"role": "system", "content": "Use tools when available. Be brief."},
            {
                "role": "user",
                "content": (
                    "Call mcp__smoke__echo with text=\"TOKEN-7788\". "
                    "Do not answer in plain text first."
                ),
            },
        ],
        tools=[echo_tool],
        label="call",
    )
    print("CALL answer:", repr(r_call["answer"][:160]))
    print("CALL tools:", [
        ((t.get("function") or {}).get("name"), (t.get("function") or {}).get("arguments"))
        for t in r_call["tool_calls"]
    ])
    if not r_call["tool_calls"]:
        print("FAIL: no tool call")
        return 1

    tc = r_call["tool_calls"][0]
    tid = str(tc.get("id") or "").strip()
    tname = str((tc.get("function") or {}).get("name") or "mcp__smoke__echo")
    targs = str((tc.get("function") or {}).get("arguments") or '{"text":"TOKEN-7788"}')
    if not tid:
        print("FAIL: empty tool_call id")
        return 1

    parked = find_parked_by_tool_ids([tid])
    print("PARKED after call:", bool(parked), "tid=", tid[:24])
    if parked is None:
        print("FAIL: expected ParkedRun after tool_call")
        return 1

    continue_msgs = [
        {"role": "system", "content": "Be brief. Use tool results."},
        {"role": "user", "content": "Call echo with TOKEN-7788"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tid,
                "type": "function",
                "function": {"name": tname, "arguments": targs},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tid,
            "name": tname,
            "content": "echo-result:TOKEN-7788",
        },
    ]

    r_cont = await _collect(continue_msgs, tools=[echo_tool], label="continue")
    ans = r_cont["answer"]
    print("CONTINUE meta:", r_cont["meta"])
    print("CONTINUE answer:", repr(ans[:300]))
    print("CONTINUE thinking[:200]:", repr(r_cont["thinking"][:200]))
    print("CONTINUE tools:", [
        ((t.get("function") or {}).get("name"), (t.get("function") or {}).get("arguments"))
        for t in r_cont["tool_calls"]
    ])

    resumed = bool(r_cont["meta"].get("resume_exec"))
    saw_token = ("TOKEN-7788" in ans) or ("7788" in ans)
    amnesia = _looks_like_empty_query_amnesia(ans) and not saw_token
    ok = resumed and bool(ans.strip()) and saw_token and not amnesia

    print("===== VERDICT (clean tid) =====")
    print("resume_exec:", resumed)
    print("saw_token:", saw_token)
    print("amnesia:", amnesia)
    print("PASS" if ok else "FAIL")
    if not ok:
        return 1

    # --- dirty tid（换行拼接）再跑一轮 ---
    r_call2 = await _collect(
        [
            {"role": "system", "content": "Use tools when available. Be brief."},
            {
                "role": "user",
                "content": (
                    "Call mcp__smoke__echo with text=\"TOKEN-9901\". "
                    "Do not answer in plain text first."
                ),
            },
        ],
        tools=[echo_tool],
        label="call-dirty",
    )
    if not r_call2["tool_calls"]:
        print("FAIL dirty: no tool call")
        return 1
    tc2 = r_call2["tool_calls"][0]
    tid2 = str(tc2.get("id") or "").strip()
    dirty = f"{tid2}\nfc_smoke_alias_1"
    tname2 = str((tc2.get("function") or {}).get("name") or "mcp__smoke__echo")
    targs2 = str((tc2.get("function") or {}).get("arguments") or '{"text":"TOKEN-9901"}')
    print("DIRTY tid=", repr(dirty[:48]))
    parked2 = find_parked_by_tool_ids([dirty])
    print("PARKED via dirty alias:", bool(parked2))
    if parked2 is None:
        print("FAIL dirty: ParkedRun not found via newline alias")
        return 1
    r_dirty = await _collect(
        [
            {"role": "system", "content": "Be brief. Use tool results."},
            {"role": "user", "content": "Call echo with TOKEN-9901"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tid2,
                    "type": "function",
                    "function": {"name": tname2, "arguments": targs2},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": dirty,
                "name": tname2,
                "content": "echo-result:TOKEN-9901",
            },
        ],
        tools=[echo_tool],
        label="continue-dirty",
    )
    ans_d = r_dirty["answer"]
    resumed_d = bool(r_dirty["meta"].get("resume_exec"))
    saw_d = ("TOKEN-9901" in ans_d) or ("9901" in ans_d)
    print("CONTINUE-DIRTY meta:", r_dirty["meta"])
    print("CONTINUE-DIRTY answer:", repr(ans_d[:300]))
    ok_d = resumed_d and saw_d and bool(ans_d.strip())
    print("===== VERDICT (dirty tid) =====")
    print("resume_exec:", resumed_d, "saw_token:", saw_d)
    print("PASS" if ok_d else "FAIL")
    if not ok_d:
        return 1

    # --- Plan mode system-reminder：本轮 text 仅 reminder，须锚定原任务 ---
    from upstream.cursor.chat.convert import split_prompt_and_history

    reminder_msgs = [
        {"role": "user", "content": "先跑 achecker.py 并制定全量交付计划"},
        {"role": "assistant", "content": "收到，开始规划。"},
        {
            "role": "user",
            "content": (
                "<system-reminder>\nPlan mode is active. The user indicated that "
                "they do not want you to execute yet — you MUST NOT make any edits "
                "or run any non-readonly tools.\n</system-reminder>"
            ),
        },
    ]
    prompt_r, _ = split_prompt_and_history(reminder_msgs)
    anchored = "achecker" in prompt_r.lower() and prompt_r.strip().startswith("先跑")
    print("REMINDER prompt head:", repr(prompt_r[:120]))
    r_plan = await _collect(reminder_msgs, tools=None, label="plan-reminder")
    ans_p = r_plan["answer"] + r_plan["thinking"]
    plan_amnesia = _looks_like_empty_query_amnesia(ans_p) and "achecker" not in ans_p.lower()
    remembers = ("achecker" in ans_p.lower()) or ("交付" in ans_p) or ("计划" in ans_p)
    print("PLAN answer:", repr(r_plan["answer"][:300]))
    print("PLAN thinking[:200]:", repr(r_plan["thinking"][:200]))
    ok_p = anchored and remembers and not plan_amnesia
    print("===== VERDICT (plan reminder) =====")
    print("anchored:", anchored, "remembers:", remembers, "amnesia:", plan_amnesia)
    print("PASS" if ok_p else "FAIL")
    return 0 if ok_p else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("FATAL:", type(exc).__name__, exc)
        raise
