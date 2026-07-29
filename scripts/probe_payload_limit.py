"""Binary-search chat.qwen.ai HTTP 413 body limit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
import path_setup  # noqa: F401

from core.crypto.crypto import build_headers, build_login_headers, hash_password
from core.transport.routes import AUTH_BASE_URL, BASE_URL, CHAT_PATH, NEW_CHAT_PATH
from server.formats import build_chat_payload, build_qwen_message

MODEL = os.environ.get("PROBE_MODEL", "qwen3.5-plus")
MI_B = 1024 * 1024


def _credentials() -> Tuple[str, str]:
    email = os.environ.get("GENERALUSR", "").strip()
    password = os.environ.get("GENERALPWD", "").strip()
    if not email or not password:
        raise SystemExit("GENERALUSR/GENERALPWD not set")
    return email, password


async def _login(session: aiohttp.ClientSession, email: str, password: str) -> str:
    async with session.post(
        f"{AUTH_BASE_URL}/api/v2/auths/signin",
        json={"email": email, "password": hash_password(password), "remember_me": True},
        headers=build_login_headers(),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"login HTTP {resp.status}: {body[:300]}")
        data = json.loads(body)
        token = str((data.get("data") or {}).get("access_token", ""))
        if not token:
            raise RuntimeError(f"missing token: {data}")
        return token


async def _create_chat(session: aiohttp.ClientSession, token: str) -> str:
    async with session.post(
        f"{BASE_URL}{NEW_CHAT_PATH}",
        json={
            "title": "413 probe",
            "models": [MODEL],
            "chat_mode": "local",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000),
            "project_id": "",
        },
        headers=build_headers(token, include_version=False),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"create chat HTTP {resp.status}: {body[:300]}")
        data = json.loads(body)
        chat_id = str((data.get("data") or {}).get("id", ""))
        if not chat_id:
            raise RuntimeError(f"no chat_id: {data}")
        return chat_id


def _payload(chat_id: str, content: str) -> dict:
    msg = build_qwen_message(content, MODEL, thinking_enabled=True, thinking_mode="Thinking")
    return build_chat_payload(chat_id, MODEL, msg)


def _body_bytes(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


async def _probe_chars(
    session: aiohttp.ClientSession,
    token: str,
    chat_id: str,
    n: int,
    *,
    timeout_s: int,
) -> Tuple[int, int, str]:
    content = "x" * n
    payload = _payload(chat_id, content)
    body_bytes = _body_bytes(payload)
    async with session.post(
        f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}",
        json=payload,
        headers=build_headers(token, chat_id=chat_id, include_sse=True),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=timeout_s, sock_read=min(timeout_s, 120)),
    ) as resp:
        status = resp.status
        if status == 200:
            await resp.release()
            return status, body_bytes, ""
        body = await resp.text()
        return status, body_bytes, body[:300]


async def _binary_search(
    session: aiohttp.ClientSession,
    token: str,
    lo: int,
    hi: int,
    *,
    timeout_s: int,
) -> Tuple[int, int, int, int, str]:
    chat_id = await _create_chat(session, token)
    print(f"binary search chat_id={chat_id}  range=[{lo:,}, {hi:,}]")
    ok_chars = lo
    ok_bytes = 0
    fail_chars = hi
    fail_bytes = 0
    fail_snippet = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        status, body_bytes, snippet = await _probe_chars(
            session, token, chat_id, mid, timeout_s=timeout_s,
        )
        print(f"  mid={mid:,} body={body_bytes:,} ({body_bytes / MI_B:.4f} MiB) -> HTTP {status}")
        if status == 413:
            fail_chars = mid
            fail_bytes = body_bytes
            fail_snippet = snippet
            hi = mid - 1
        elif status == 200:
            ok_chars = mid
            ok_bytes = body_bytes
            lo = mid + 1
        else:
            raise RuntimeError(f"unexpected HTTP {status}: {snippet!r}")
        await asyncio.sleep(0.3)
    return ok_chars, ok_bytes, fail_chars, fail_bytes, fail_snippet


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=int, default=21_900_000)
    parser.add_argument("--hi", type=int, default=22_050_000)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    email, password = _credentials()
    print(f"model={MODEL}")
    async with aiohttp.ClientSession() as session:
        token = await _login(session, email, password)
        print("login ok")
        ok_c, ok_b, fail_c, fail_b, snippet = await _binary_search(
            session, token, args.lo, args.hi, timeout_s=args.timeout,
        )
        print("\n=== result ===")
        print(f"max_ok_content_chars: {ok_c:,}")
        print(f"max_ok_body_bytes:    {ok_b:,} ({ok_b / MI_B:.6f} MiB)")
        print(f"first_413_at_chars:   {fail_c:,}")
        print(f"first_413_body_bytes: {fail_b:,} ({fail_b / MI_B:.6f} MiB)")
        print(f"413 snippet: {snippet[:200]!r}")
        # 推荐配置：留 256 KiB 余量
        margin = 256 * 1024
        rec = max(0, ok_b - margin)
        print(f"\nrecommended qwen_send_max_chars (body margin 256KiB): {rec:,}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
