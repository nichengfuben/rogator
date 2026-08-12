#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zen 上游工具调用探测：验证 tools 参数正确传递且上游正常响应。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from upstream.zen.client import ZenClient
from upstream.zen.payload import build_chat_payload, normalize_tools


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    },
]

MESSAGES = [
    {"role": "user", "content": "北京今天天气怎么样？请使用 get_weather 工具查询。"},
]


async def main() -> int:
    client = ZenClient(splitter=None)
    await client._ensure_http_session()

    model = "qwen3-coder-plus"
    tools = normalize_tools(TOOLS)
    payload = build_chat_payload(
        MESSAGES,
        model,
        stream=True,
        tools=tools,
    )

    print(f"model={model}")
    print(f"tools_in_payload={'tools' in payload}")
    print(f"tools_count={len(payload.get('tools', []))}")

    events: list[dict] = []
    err = ""
    try:
        async for event in client.stream_chat(payload):
            etype = event.get("type", "")
            print(f"  event: type={etype} ", end="")
            if etype == "answer":
                content = event.get("content", "")
                print(f"len={len(content)}")
            elif etype == "thinking":
                content = event.get("content", "")
                print(f"len={len(content)}")
            elif etype == "usage":
                print(f"data={event.get('data')}")
            elif etype == "tool_call":
                tc = event.get("tool_call", {})
                idx = tc.get("index", "?")
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                print(f"index={idx} name={name} args_len={len(args)}")
            else:
                print(f"raw={json.dumps(event, ensure_ascii=False)[:200]}")
            events.append(event)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {err}")

    has_answer = any(e.get("type") == "answer" for e in events)
    has_error = any(e.get("type") == "error" for e in events)

    print(f"\ntotal_events={len(events)}")
    print(f"has_answer={has_answer}")
    print(f"has_error={has_error}")
    if err:
        print(f"error_detail={err}")
        await client.shutdown()
        return 1

    if has_answer and not has_error:
        print("OK - tools 参数已正确传递，上游正常响应")
    elif not has_answer:
        print("WARN - 未收到 answer 事件，请检查上游返回")
    await client.shutdown()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
