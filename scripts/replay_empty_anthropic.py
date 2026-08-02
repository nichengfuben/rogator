#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重放 Anthropic 空响应：用已落盘 inject prompt 直连 Qwen，打印上游 SSE 事件。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.store import valid_session_count
from upstream.qwen.chat.store import load_upstream_sessions
from upstream.qwen.client import QwenClient

PROMPT_PATH = ROOT / "logs" / "prompts" / "req-1785688676-374650893af8.txt"
MODEL = "qwen3.7-max"


async def _replay(*, full_prompt: bool) -> int:
    if not PROMPT_PATH.is_file():
        print(f"缺少 prompt: {PROMPT_PATH}", flush=True)
        return 2

    sessions, _ = load_upstream_sessions("qwen")
    valid = [s for s in sessions if s.is_valid and not s.is_expired()]
    if not valid:
        print("无有效 Qwen session", flush=True)
        return 2
    session = valid[0]
    print(f"session={session.username[:6]} valid_pool={valid_session_count(sessions)}", flush=True)

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"prompt_chars={len(prompt)} full_prompt={full_prompt}", flush=True)

    class _Splitter:
        max_chars = 120_000 if not full_prompt else 0
        send_full_prompt = full_prompt

    client = QwenClient(_Splitter())
    await client._ensure_http_session()
    messages = [{"role": "user", "content": prompt}]
    events: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    chat_id = await client.create_chat(session, MODEL)
    print(f"chat_id={chat_id[:16]}…", flush=True)
    try:
        async for event in client.chat_completion(session, chat_id, messages, MODEL):
            events.append(event)
            etype = event.get("type")
            content = event.get("content", "")
            preview = str(content)[:120].replace("\n", "\\n") if content else ""
            extra = ""
            if etype == "usage":
                extra = json.dumps(event.get("data", {}), ensure_ascii=False)[:200]
            elif etype == "response_created":
                extra = str(event.get("response_id", ""))[:40]
            print(f"  [{len(events):03d}] {etype} {extra or preview}", flush=True)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1

    elapsed = time.perf_counter() - t0
    answers = [e.get("content", "") for e in events if e.get("type") == "answer"]
    thinking = [e.get("content", "") for e in events if e.get("type") == "thinking"]
    usage = [e for e in events if e.get("type") == "usage"]
    print(
        f"done elapsed={elapsed:.2f}s events={len(events)} "
        f"answer_chars={sum(len(x) for x in answers)} "
        f"thinking_chars={sum(len(x) for x in thinking)} usage_events={len(usage)}",
        flush=True,
    )
    if not answers and not thinking:
        print("结论: 上游 SSE 无 thinking/answer → Anthropic 侧空响应", flush=True)
    return 0


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    full = mode != "truncate"
    raise SystemExit(asyncio.run(_replay(full_prompt=full)))


if __name__ == "__main__":
    main()
