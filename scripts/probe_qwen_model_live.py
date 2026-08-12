#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短 prompt 实网探测：登录 → 建聊 → completions，打印 Cookie 键与结果。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import accounts_for_upstream
from upstream.qwen.auth.crypto import build_headers
from upstream.qwen.client import QwenClient
from upstream.qwen.chat.routes import CHAT_PATH


async def main() -> int:
    pool = accounts_for_upstream("qwen")
    if not pool:
        print("NO_ACCOUNT")
        return 2
    account = pool[0]
    client = QwenClient(splitter=None)
    await client._ensure_http_session()
    print(f"login user={account.username[:6]}...")
    session = await client._perform_login(account)
    if not session:
        print("LOGIN_FAIL")
        return 2
    print(f"login ok user_id={str(session.user_id)[:8]}...")

    model = "qwen3-coder-plus"
    cookies = client.cookies_for_session(session, thinking_mode="Fast")
    probe_headers = build_headers(
        session.token,
        include_sse=True,
        api_path=CHAT_PATH,
        cookies=cookies,
    )
    cookie_header = probe_headers.get("Cookie", "")
    keys = [p.split("=", 1)[0] for p in cookie_header.split("; ") if p]
    print("cookie_keys=", ",".join(keys))
    print("has_Authorization=", "Authorization" in probe_headers)

    chat_id = await client.create_chat(session, model)
    print(f"chat_id={chat_id[:12]}...")
    messages = [{"role": "user", "content": "只用中文回复两个字：你好"}]
    parts: list[str] = []
    err = ""
    try:
        async for event in client.chat_completion(
            session,
            chat_id,
            messages,
            model,
            qwen_thinking_enabled=False,
            qwen_thinking_mode="Fast",
        ):
            if event.get("type") == "answer" and event.get("content"):
                parts.append(str(event["content"]))
                if sum(len(x) for x in parts) > 80:
                    break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    text = "".join(parts)
    print(f"answer={text[:120]!r}")
    print(f"harvested={sorted(client.cookie_jar)}")
    if err:
        print(f"ERROR={err}")
        await client.shutdown()
        return 1
    if not text.strip():
        print("EMPTY_ANSWER")
        await client.shutdown()
        return 1
    print("OK")
    await client.shutdown()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
