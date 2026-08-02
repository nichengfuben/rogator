#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen 直连：流式聊天 + cancel，验证 stop / delete。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

import upstream.qwen.chat.chat as qwen_chat
from core.session.accounts import accounts_for_upstream
from upstream.qwen.client import QwenClient


async def main() -> int:
    pool = accounts_for_upstream("qwen")
    if not pool:
        print("无 Qwen 账号配置", flush=True)
        return 2
    account = pool[0]

    stop_called = False
    delete_called = False
    stop_detail: dict[str, Any] = {}
    delete_detail: dict[str, Any] = {}

    real_stop = qwen_chat.stop_upstream_generation
    real_delete = qwen_chat.delete_upstream_chat

    async def _tracked_stop(client, session, chat_id, response_id=""):
        nonlocal stop_called, stop_detail
        stop_called = True
        stop_detail = {
            "chat_id": chat_id,
            "response_id": response_id,
            "user": session.username[:6],
        }
        print(
            f"[stop] chat_id={chat_id[:12]} response_id={response_id or '(empty)'}",
            flush=True,
        )
        return await real_stop(client, session, chat_id, response_id)

    async def _tracked_delete(client, session, chat_id):
        nonlocal delete_called, delete_detail
        delete_called = True
        delete_detail = {"chat_id": chat_id, "user": session.username[:6]}
        print(f"[delete] chat_id={chat_id[:12]}", flush=True)
        return await real_delete(client, session, chat_id)

    client = QwenClient(splitter=None)
    await client._ensure_http_session()

    print(f"登录 Qwen ({account.username[:6]}…)…", flush=True)
    session = await client._perform_login(account)
    if not session:
        print("登录失败", flush=True)
        return 2
    print("登录成功", flush=True)

    model = "qwen3.7-max"
    messages = [{"role": "user", "content": "用中文写一首关于春天的短诗，至少八行。"}]
    chunks: List[str] = []

    async def _consume(chat_id: str):
        async for event in client.chat_completion(
            session, chat_id, messages, model,
        ):
            if event.get("type") == "answer" and event.get("content"):
                chunks.append(str(event["content"]))
                print(event["content"], end="", flush=True)

    with patch.object(qwen_chat, "stop_upstream_generation", _tracked_stop), patch.object(
        qwen_chat, "delete_upstream_chat", _tracked_delete,
    ):
        print(f"Qwen 流式测试 ({session.username[:6]}…)…", flush=True)
        chat_id = await client.create_chat(session, model)
        print(f"chat_id={chat_id[:12]}", flush=True)
        task = asyncio.create_task(_consume(chat_id))
        for _ in range(200):
            if chunks:
                break
            if task.done():
                break
            await asyncio.sleep(0.05)
        if not chunks:
            try:
                await asyncio.wait_for(task, timeout=120.0)
            except asyncio.TimeoutError:
                task.cancel()
                print("超时未收到首包", flush=True)
                return 1
            print(f"\n完整结束 chunks={len(chunks)}", flush=True)
            return 0

        print("\n--- cancel ---", flush=True)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=30.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    await asyncio.sleep(1.0)
    print(
        f"\n首包后 cancel: chunks={len(chunks)}, "
        f"stop={stop_called}, delete={delete_called}, "
        f"stop_detail={stop_detail or 'none'}",
        flush=True,
    )
    await client.shutdown()
    return 0 if chunks and stop_called and delete_called else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
