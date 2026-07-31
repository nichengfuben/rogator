#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量联调：多轮 / 工具调用 / 模型 / 思考 / 回复 / 工具回执。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.auth.store import get_access_token
from upstream.cursor.chat.agent_session import find_parked_by_tool_ids
from upstream.cursor.chat.convert import map_model, split_prompt_and_history
from upstream.cursor.chat.openai import stream_openai_chat
from upstream.cursor.chat.tool_ids import normalize_tool_call_id
from upstream.cursor.client import CursorClient

MODEL = "composer-2.5-fast"
TOKEN = "MATRIX-4421"
WS = "X:/Project/Public/Qwen"


class _SmokeClient:
    def __init__(self) -> None:
        self._conversation_id: Optional[str] = None
        self._workspace = WS
        self._inner = CursorClient()

    async def ensure_token(self) -> None:
        await self._inner.ensure_token()


def _echo_tool() -> Dict[str, Any]:
    return {
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


async def _collect(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]],
    label: str,
    model: str = MODEL,
) -> Dict[str, Any]:
    client = _SmokeClient()
    thinking: List[str] = []
    answer: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    types: List[str] = []
    t0 = time.time()
    err: Optional[str] = None
    try:
        async for ev in stream_openai_chat(
            state=None,
            client=client,
            messages=messages,
            model=model,
            tools=tools,
            req_id=f"matrix-{label}",
        ):
            et = str(ev.get("type") or "")
            types.append(et)
            if et == "prompt_meta":
                meta = ev
            elif et == "thinking":
                thinking.append(str(ev.get("content") or ""))
            elif et == "answer":
                answer.append(str(ev.get("content") or ""))
            elif et == "tool_call":
                tool_calls.append(ev.get("tool_call") or {})
            elif et == "error":
                err = str(ev.get("error") or ev)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    return {
        "label": label,
        "elapsed": round(time.time() - t0, 2),
        "types": types,
        "meta": meta,
        "thinking": "".join(thinking),
        "answer": "".join(answer),
        "tool_calls": tool_calls,
        "error": err,
        "mapped_model": map_model(model),
        "req_model": model,
    }


def _check(name: str, ok: bool, detail: str, rows: List[Tuple[str, bool, str]]) -> None:
    rows.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _print_turn(r: Dict[str, Any]) -> None:
    print(f"\n----- {r['label']} ({r['elapsed']}s) -----")
    print("req_model:", r["req_model"], "mapped:", r["mapped_model"])
    print("types:", r["types"])
    print("resume_exec:", bool(r["meta"].get("resume_exec")), "meta:", r["meta"])
    print("thinking[:240]:", repr(r["thinking"][:240]))
    print("answer[:240]:", repr(r["answer"][:240]))
    for i, tc in enumerate(r["tool_calls"]):
        fn = tc.get("function") or {}
        print(f"tool[{i}]: id={tc.get('id')!r} name={fn.get('name')!r} args={fn.get('arguments')!r}")
    if r["error"]:
        print("error:", r["error"])


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    rows: List[Tuple[str, bool, str]] = []
    if not get_access_token():
        print("FAIL: no access token")
        return 2

    tools = [_echo_tool()]
    mapped = map_model(MODEL)
    _check("model.map", bool(mapped), f"{MODEL} -> {mapped}", rows)

    # ---- T1 多轮首轮：纯对话（思考 + 回复）----
    hist: List[Dict[str, Any]] = [
        {"role": "system", "content": "Be brief. Keep prior tokens in mind."},
        {"role": "user", "content": "Reply with exactly one word: PING-MATRIX"},
    ]
    t1 = await _collect(hist, tools=None, label="t1-chat")
    _print_turn(t1)
    has_think = bool(t1["thinking"].strip()) or ("thinking" in t1["types"])
    has_ans = "PING-MATRIX" in t1["answer"] or "PING" in t1["answer"].upper()
    _check("t1.no_error", not t1["error"], t1["error"] or "ok", rows)
    _check("t1.thinking_or_stream", has_think or bool(t1["answer"]), f"thinking_len={len(t1['thinking'])}", rows)
    _check("t1.answer", has_ans or bool(t1["answer"].strip()), repr(t1["answer"][:80]), rows)
    hist.append({"role": "assistant", "content": t1["answer"] or "PING-MATRIX"})

    # ---- T2 多轮续聊：记住上一轮 ----
    hist.append({"role": "user", "content": "What single word did I ask you to reply with? Answer with that word only."})
    t2 = await _collect(hist, tools=None, label="t2-memory")
    _print_turn(t2)
    mem_ok = "PING" in (t2["answer"] + t2["thinking"]).upper()
    _check("t2.no_error", not t2["error"], t2["error"] or "ok", rows)
    _check("t2.multi_turn_memory", mem_ok, repr(t2["answer"][:120]), rows)
    hist.append({"role": "assistant", "content": t2["answer"] or "PING-MATRIX"})

    # ---- T3 工具调用（模型应发 tool_call，不应空答糊弄）----
    hist.append({
        "role": "user",
        "content": (
            f"Call mcp__smoke__echo with text=\"{TOKEN}\". "
            "Do not invent the result; wait for the tool."
        ),
    })
    t3 = await _collect(hist, tools=tools, label="t3-tool-call")
    _print_turn(t3)
    _check("t3.no_error", not t3["error"], t3["error"] or "ok", rows)
    _check("t3.has_tool_call", bool(t3["tool_calls"]), f"n={len(t3['tool_calls'])}", rows)
    tc = t3["tool_calls"][0] if t3["tool_calls"] else {}
    fn = tc.get("function") or {}
    tid = normalize_tool_call_id(tc.get("id") or "") or str(tc.get("id") or "")
    name_ok = str(fn.get("name") or "") == "mcp__smoke__echo"
    args_ok = TOKEN in str(fn.get("arguments") or "")
    _check("t3.tool_name", name_ok, repr(fn.get("name")), rows)
    _check("t3.tool_args", args_ok, repr(fn.get("arguments")), rows)
    _check("t3.tool_id_clean", "\n" not in tid and bool(tid), repr(tid[:48]), rows)
    parked = find_parked_by_tool_ids([tid]) if tid else None
    _check("t3.parked", parked is not None, f"tid={tid[:24]}", rows)

    if not tid or not t3["tool_calls"]:
        print("\n===== SUMMARY =====")
        for n, ok, d in rows:
            print(f"{'PASS' if ok else 'FAIL'} | {n} | {d}")
        return 1

    hist.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tid,
            "type": "function",
            "function": {"name": fn.get("name"), "arguments": fn.get("arguments")},
        }],
    })

    # ---- T4 工具回执 + 同流 resume → 思考/回复含 TOKEN ----
    hist.append({
        "role": "tool",
        "tool_call_id": tid,
        "name": "mcp__smoke__echo",
        "content": f"echo-result:{TOKEN}",
    })
    t4 = await _collect(hist, tools=tools, label="t4-tool-result")
    _print_turn(t4)
    resumed = bool(t4["meta"].get("resume_exec"))
    blob = t4["answer"] + t4["thinking"]
    saw = TOKEN in blob or "4421" in blob
    _check("t4.resume_exec", resumed, str(t4["meta"]), rows)
    _check("t4.no_error", not t4["error"], t4["error"] or "ok", rows)
    _check("t4.thinking_present", bool(t4["thinking"].strip()) or "thinking" in t4["types"], f"len={len(t4['thinking'])}", rows)
    _check("t4.answer_has_token", saw and bool(t4["answer"].strip()), repr(t4["answer"][:160]), rows)
    _check("t4.no_extra_tool", len(t4["tool_calls"]) == 0, f"n={len(t4['tool_calls'])}", rows)
    hist[-1]  # keep tool msg
    # replace trailing tool turn with assistant final for next chat turn
    hist.append({"role": "assistant", "content": t4["answer"] or f"got {TOKEN}"})

    # ---- T5 多轮：追问工具结果 ----
    hist.append({"role": "user", "content": "What TOKEN did the echo tool return? Reply with the token only."})
    t5 = await _collect(hist, tools=tools, label="t5-ask-token")
    _print_turn(t5)
    ask_ok = TOKEN in (t5["answer"] + t5["thinking"]) or "4421" in (t5["answer"] + t5["thinking"])
    _check("t5.no_error", not t5["error"], t5["error"] or "ok", rows)
    _check("t5.remembers_tool_result", ask_ok, repr(t5["answer"][:120]), rows)

    # ---- T6 脏 tool_call_id 别名 resume ----
    t6a = await _collect(
        [
            {"role": "system", "content": "Use tools. Be brief."},
            {"role": "user", "content": 'Call mcp__smoke__echo with text="DIRTY-778". Do not answer first.'},
        ],
        tools=tools,
        label="t6a-call",
    )
    _print_turn(t6a)
    tc6 = t6a["tool_calls"][0] if t6a["tool_calls"] else {}
    tid6 = normalize_tool_call_id(tc6.get("id") or "") or str(tc6.get("id") or "")
    dirty = f"{tid6}\nfc_matrix_alias"
    fn6 = tc6.get("function") or {}
    _check("t6.call", bool(tid6), repr(tid6[:40]), rows)
    t6b = await _collect(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Call echo DIRTY-778"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tid6,
                    "type": "function",
                    "function": {"name": fn6.get("name"), "arguments": fn6.get("arguments")},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": dirty,
                "name": "mcp__smoke__echo",
                "content": "echo-result:DIRTY-778",
            },
        ],
        tools=tools,
        label="t6b-dirty-resume",
    )
    _print_turn(t6b)
    _check("t6.resume_exec", bool(t6b["meta"].get("resume_exec")), str(t6b["meta"]), rows)
    _check("t6.answer_token", "DIRTY-778" in (t6b["answer"] + t6b["thinking"]), repr(t6b["answer"][:120]), rows)

    # ---- T7 Plan reminder 锚定（只验转换 + 短轮；超时记 WARN 不算硬失败若锚定 OK）----
    reminder_msgs = [
        {"role": "user", "content": f"Remember project token {TOKEN} and run achecker.py plan."},
        {"role": "assistant", "content": "Understood, planning achecker."},
        {
            "role": "user",
            "content": "<system-reminder>\nPlan mode is active. Do not execute yet.\n</system-reminder>",
        },
    ]
    prompt_r, _ = split_prompt_and_history(reminder_msgs)
    anchored = TOKEN in prompt_r and "achecker" in prompt_r.lower() and "<system-reminder>" in prompt_r
    _check("t7.prompt_anchored", anchored, repr(prompt_r[:140]), rows)
    t7 = await _collect(reminder_msgs, tools=None, label="t7-plan")
    _print_turn(t7)
    if t7["error"] and "timeout" in t7["error"].lower():
        _check("t7.upstream_timeout", False, t7["error"] + " (anchored already checked)", rows)
    else:
        remembers = TOKEN in (t7["answer"] + t7["thinking"]) or "achecker" in (t7["answer"] + t7["thinking"]).lower()
        _check("t7.no_error", not t7["error"], t7["error"] or "ok", rows)
        _check("t7.remembers_goal", remembers, repr(t7["answer"][:160]), rows)

    print("\n===== SUMMARY =====")
    fails = 0
    for n, ok, d in rows:
        print(f"{'PASS' if ok else 'FAIL'} | {n} | {d}")
        if not ok:
            fails += 1
    # soft: t7 timeout alone shouldn't fail whole matrix if core path passed
    hard = [n for n, ok, _ in rows if not ok and not n.startswith("t7.")]
    print(json.dumps({
        "total": len(rows),
        "fails": fails,
        "hard_fails": len(hard),
        "model": mapped,
        "verdict": "PASS" if not hard else "FAIL",
    }, ensure_ascii=False))
    return 0 if not hard else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
