#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek 直连：流式聊天 + cancel，验证 stop_stream。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, List

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import accounts_for_upstream
from upstream.deepseek.lib.adapter.client import DeepseekClient
from upstream.deepseek.lib.adapter.helpers.pmtutil import Account as DsAccount
from upstream.deepseek.lib.user.userapi import login


async def main() -> int:
    pool = accounts_for_upstream("deepseek")
    if not pool:
        print("无 DeepSeek 账号配置", flush=True)
        return 2
    account = pool[0]

    stop_called = False
    stop_detail: dict[str, Any] = {}
    real_stop = None

    import upstream.deepseek.lib.session.sessapi as sessapi

    async def _tracked_stop(http, token, chat_session_id, message_id):
        nonlocal stop_called, stop_detail
        stop_called = True
        stop_detail = {
            "chat_session_id": str(chat_session_id),
            "message_id": str(message_id),
        }
        print(
            f"[stop_stream] session={stop_detail['chat_session_id'][:16]} "
            f"msg={stop_detail['message_id']}",
            flush=True,
        )
        assert real_stop is not None
        return await real_stop(http, token, chat_session_id, message_id)

    real_stop = sessapi.stop_stream

    async with aiohttp.ClientSession() as http:
        print(f"登录 DeepSeek ({account.username[:6]}…)…", flush=True)
        token, user_id, _did = await login(http, account.username, account.password)
        print("登录成功", flush=True)

        inner = DeepseekClient()
        ds_acc = DsAccount(
            username=account.username,
            password=account.password,
            token=token,
            user_id=user_id,
        )
        await inner.init_immediate(http, accounts=[ds_acc])
        await inner.background_setup(login_accounts=False)

        cands = await inner.candidates()
        if not cands:
            print("无 candidate", flush=True)
            return 2
        candidate = cands[0]

        chunks: List[str] = []

        async def _consume():
            async for chunk in inner.complete(
                candidate,
                [{"role": "user", "content": "用中文写一首关于春天的短诗，至少八行。"}],
                "deepseek-v4-flash",
                stream=True,
            ):
                if isinstance(chunk, str) and chunk:
                    chunks.append(chunk)
                    print(chunk, end="", flush=True)

        with __import__("unittest.mock").mock.patch.object(
            sessapi, "stop_stream", side_effect=_tracked_stop,
        ):
            task = asyncio.create_task(_consume())
            for _ in range(100):
                if chunks:
                    break
                await asyncio.sleep(0.05)
            if not chunks:
                try:
                    await asyncio.wait_for(task, timeout=120.0)
                except asyncio.TimeoutError:
                    task.cancel()
                    print("超时未收到首包", flush=True)
                    return 1
                print(f"\n完整结束，共 {len(chunks)} 块（未 cancel）", flush=True)
                return 0

            print("\n--- cancel ---", flush=True)
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=30.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await asyncio.sleep(1.0)
        print(
            f"\n首包后 cancel: chunks={len(chunks)}, stop_called={stop_called}, "
            f"detail={stop_detail or 'none'}",
            flush=True,
        )
        await inner.close()
        return 0 if chunks and stop_called else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
