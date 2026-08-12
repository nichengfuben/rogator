#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rogator 超限路径：split → 附件上传 → completions（仅 fireye 路径）。"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import Account, accounts_for_upstream
from state import LongTextSplitter
from upstream.qwen.client import QwenClient


def _sync_proxy(proxy: str) -> None:
    if not proxy:
        return
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


def _load_ext_account(index: int = 0) -> Account:
    import re

    path = Path(
        os.environ.get(
            "QWEN_EXT_ACCOUNTS",
            str(ROOT / "config" / "upstream" / "qwen" / "ext_accounts.toml"),
        )
    )
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(
        r'\[\[accounts\]\]\s*\nusername\s*=\s*"([^"]+)"\s*\npassword\s*=\s*"([^"]+)"',
        text,
    )
    if index >= len(blocks):
        raise IndexError(f"account index {index} out of range ({len(blocks)})")
    user, pwd = blocks[index]
    return Account(username=user, password=pwd)


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-file",
        default=str(ROOT / "logs" / "prompts" / "req-1785757818-d8255a751e4d.txt"),
    )
    parser.add_argument("--model", default="qwen3-coder-plus")
    parser.add_argument("--max-chars", type=int, default=1024000)
    parser.add_argument("--ext-account", type=int, default=0)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--max-events", type=int, default=20)
    args = parser.parse_args(argv)

    proxy = (args.proxy or "").strip()
    if proxy:
        _sync_proxy(proxy)
        print(f"proxy={proxy}", flush=True)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    splitter = LongTextSplitter(max_chars=args.max_chars, send_full_prompt=False)
    send_text, filename, file_bytes = splitter.split(prompt)
    print(
        f"prompt={len(prompt)} max_chars={args.max_chars} "
        f"send={len(send_text)} attach={'yes' if file_bytes else 'no'} "
        f"attach_bytes={len(file_bytes or b'')} name={filename}",
        flush=True,
    )
    if file_bytes is None:
        print("未触发附件分割（prompt 未超限或 send_full_prompt）", flush=True)

    try:
        account = _load_ext_account(args.ext_account)
    except Exception:
        pool = accounts_for_upstream("qwen")
        if not pool:
            print("无可用账号", flush=True)
            return 2
        account = pool[0]
    print(f"account={account.username[:6]}… model={args.model}", flush=True)

    client = QwenClient(splitter=splitter)
    await client._ensure_http_session()
    t0 = time.time()
    session = await client._perform_login(account)
    if not session:
        print("login FAIL", flush=True)
        return 2
    print(f"login OK ({int((time.time() - t0) * 1000)}ms)", flush=True)

    files: List[Dict[str, Any]] = []
    if filename and file_bytes:
        t1 = time.time()
        url, file_obj = await client.upload_file(session, file_bytes, filename)
        files.append(file_obj)
        print(
            f"upload OK ({int((time.time() - t1) * 1000)}ms) "
            f"file_class={file_obj.get('file_class')} status={file_obj.get('status')} "
            f"greenNet={file_obj.get('greenNet')!r} "
            f"parse={(file_obj.get('file') or {}).get('meta', {}).get('parse_meta')} "
            f"url={str(url)[:80]}",
            flush=True,
        )

    chat_id = await client.create_chat(session, args.model)
    print(f"chat_id={chat_id[:12]}…", flush=True)

    messages = [{"role": "user", "content": send_text}]
    hits = 0
    first = ""
    kind = ""
    err = ""
    t2 = time.time()
    try:
        async for event in client.chat_completion(
            session,
            chat_id,
            messages,
            args.model,
            files,
            qwen_thinking_enabled=True,
            qwen_thinking_mode="Thinking",
        ):
            et = str(event.get("type") or "")
            if et == "error":
                err = str(event.get("message") or event)[:300]
                break
            if et in ("answer", "thinking", "thinking_summary"):
                hits += 1
                if not kind:
                    kind = et
                    first = str(event.get("content") or "")[:120]
                if hits >= args.max_events:
                    break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:320]
    elapsed = int((time.time() - t2) * 1000)
    try:
        await client.cleanup_chat(session, chat_id)
    except Exception:
        pass

    if err:
        print(f"[FAIL] error={err!r} elapsed={elapsed}ms", flush=True)
        return 1
    if hits:
        print(
            f"[OK] outcome={kind} snippet={first!r} events={hits} elapsed={elapsed}ms",
            flush=True,
        )
        return 0
    print(f"[FAIL] no_model_content elapsed={elapsed}ms", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
