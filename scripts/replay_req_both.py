#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重放指定 req 的 prompt：fireye 路径 vs 短 bx-ua 路径各跑一遍 completions。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

import aiohttp

from core.session.accounts import Account
from upstream.qwen.auth.baxia_runtime import get_baxia_tokens, reset_baxia_runtime
from upstream.qwen.auth.crypto import (
    build_headers_async,
    build_login_headers,
    generate_bxua,
    generate_fingerprint,
    hash_password,
)
from upstream.qwen.chat.routes import (
    APP_VERSION,
    BASE_URL,
    CHAT_ORIGIN,
    CHAT_PATH,
    NEW_CHAT_PATH,
    SEC_CH_UA,
    SEC_CH_UA_PLATFORM,
    USER_AGENT,
)
from upstream.qwen.chat.sse import parse_sse_event
from upstream.qwen.chat.upload.payload import build_new_chat_payload
from server.formats.messages import build_chat_payload, build_qwen_message
from server.retry.http_client import client_session, sync_proxy_env

# Optional override: QWEN_EXT_ACCOUNTS=/path/to/accounts.toml
EXT_ACCOUNTS_TOML = Path(
    os.environ.get(
        "QWEN_EXT_ACCOUNTS",
        str(ROOT / "config" / "upstream" / "qwen" / "ext_accounts.toml"),
    )
)
_PROXY_URL: str = ""


def _load_ext_account(index: int = 0) -> Tuple[str, str]:
    import re

    if not EXT_ACCOUNTS_TOML.is_file():
        raise FileNotFoundError(f"accounts toml not found: {EXT_ACCOUNTS_TOML}")
    text = EXT_ACCOUNTS_TOML.read_text(encoding="utf-8")
    blocks = re.findall(
        r'\[\[accounts\]\]\s*\nusername\s*=\s*"([^"]+)"\s*\npassword\s*=\s*"([^"]+)"',
        text,
    )
    if index >= len(blocks):
        raise IndexError(f"account index {index} out of range ({len(blocks)})")
    return blocks[index]


def _http_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {"ssl": False}
    if _PROXY_URL:
        kw["proxy"] = _PROXY_URL
    return kw


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


async def _login(http: aiohttp.ClientSession, account: Account) -> str:
    async with http.post(
        f"{BASE_URL}/api/v2/auths/signin",
        json={
            "email": account.username,
            "password": hash_password(account.password),
            "remember_me": True,
        },
        headers=build_login_headers(),
        timeout=aiohttp.ClientTimeout(total=45),
        **_http_kwargs(),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"login HTTP {resp.status}: {text[:200]}")
        if text.lstrip().startswith("<!"):
            raise RuntimeError(f"login WAF/html: {text[:120]}")
        data = json.loads(text)
        if not data.get("success"):
            raise RuntimeError(f"login fail: {str(data)[:200]}")
        token = str((data.get("data") or {}).get("token") or "")
        if not token:
            raise RuntimeError("login missing token")
        return token


def _short_baxia() -> Dict[str, str]:
    fp = generate_fingerprint()
    return {
        "bxV": "0.0.3",
        "bxUa": generate_bxua(fp),
        "bxUmidToken": "T2gA" + uuid.uuid4().hex[:40] + "=",
    }


def _short_headers(
    token: str,
    *,
    chat_id: str = "",
    sse: bool = False,
    include_version: bool = False,
) -> Dict[str, str]:
    """短 bx-ua 路径请求头（completions / new-chat）。"""
    from upstream.qwen.auth.crypto import make_request_id, make_timezone

    headers = {
        "Accept": "application/json" if sse else "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Origin": CHAT_ORIGIN,
        "Referer": (
            f"{CHAT_ORIGIN}/c/local"
            if sse
            else (f"{CHAT_ORIGIN}/c/new-chat" if not chat_id else f"{CHAT_ORIGIN}/c/{chat_id}")
        ),
        "source": "web",
        "X-Request-Id": make_request_id(),
        "Timezone": make_timezone(),
        "Sec-Ch-Ua": SEC_CH_UA,
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": SEC_CH_UA_PLATFORM,
        "Cookie": f"token={token}",
    }
    baxia = _short_baxia()
    headers["bx-v"] = baxia["bxV"]
    headers["bx-ua"] = baxia["bxUa"]
    headers["bx-umidtoken"] = baxia["bxUmidToken"]
    if include_version:
        headers["Version"] = APP_VERSION
    if sse:
        headers["X-Accel-Buffering"] = "no"
    return headers


async def _rogator_create_chat(http: aiohttp.ClientSession, token: str, model: str) -> str:
    headers = await build_headers_async(token, include_version=True, api_path=NEW_CHAT_PATH)
    payload = build_new_chat_payload(model)
    async with http.post(
        f"{BASE_URL}{NEW_CHAT_PATH}",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=60),
        **_http_kwargs(),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"create_chat HTTP {resp.status}: {text[:240]}")
        data = json.loads(text)
        if not data.get("success"):
            raise RuntimeError(f"create_chat fail: {str(data)[:240]}")
        chat_id = str((data.get("data") or {}).get("id") or "")
        if not chat_id:
            raise RuntimeError("create_chat no id")
        return chat_id


async def _short_create_chat(http: aiohttp.ClientSession, token: str, model: str) -> str:
    headers = _short_headers(token, include_version=True)
    payload = {
        "chatId": "",
        "models": [model],
        "project_id": "",
        "timestamp": int(time.time() * 1000),
        "chat_type": "t2t",
        "chat_mode": "local",
    }
    async with http.post(
        f"{BASE_URL}{NEW_CHAT_PATH}",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=60),
        **_http_kwargs(),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"create_chat HTTP {resp.status}: {text[:240]}")
        data = json.loads(text)
        if not data.get("success"):
            raise RuntimeError(f"create_chat fail: {str(data)[:240]}")
        chat_id = str((data.get("data") or {}).get("id") or "")
        if not chat_id:
            raise RuntimeError("create_chat no id")
        return chat_id


def _rogator_messages(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def _short_completion_payload(chat_id: str, model: str, prompt: str) -> Dict[str, Any]:
    fid = str(uuid.uuid4())
    child = str(uuid.uuid4())
    ts = int(time.time())
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chatId": chat_id,
        "parentId": "",
        "chat_id": chat_id,
        "chat_mode": "local",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "id": None,
                "fid": fid,
                "parentId": None,
                "childrenIds": [child],
                "role": "user",
                "content": prompt,
                "user_action": "chat",
                "files": [],
                "timestamp": ts,
                "models": [model],
                "model": "",
                "chat_type": "t2t",
                "feature_config": {
                    "thinking_enabled": True,
                    "output_schema": "phase",
                    "research_mode": "normal",
                    "auto_thinking": False,
                    "thinking_mode": "Thinking",
                    "thinking_format": "raw",
                    "auto_search": False,
                },
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }


async def _collect_stream(
    http: aiohttp.ClientSession,
    *,
    mode: str,
    token: str,
    chat_id: str,
    model: str,
    prompt: str,
    out_path: Path,
    max_events: int = 30,
    timeout_s: float = 180.0,
) -> Tuple[str, str, int]:
    if mode == "rogator":
        headers = await build_headers_async(
            token,
            chat_id=chat_id,
            include_sse=True,
            api_path=CHAT_PATH,
        )
        body = build_chat_payload(
            chat_id,
            model,
            build_qwen_message(
                prompt,
                model,
                thinking_enabled=True,
                thinking_mode="Thinking",
            ),
        )
    else:
        headers = _short_headers(token, chat_id=chat_id, sse=True, include_version=True)
        body = _short_completion_payload(chat_id, model, prompt)

    url = f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}"
    raw_parts: List[str] = []
    model_hits = 0
    first_kind = ""
    first_snip = ""
    t0 = time.time()

    async with http.post(
        url,
        json=body,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
        **_http_kwargs(),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            out_path.write_text(text, encoding="utf-8")
            return "http_error", text[:300], resp.status

        async for chunk in resp.content.iter_any():
            if time.time() - t0 > timeout_s:
                break
            piece = chunk.decode("utf-8", errors="replace")
            raw_parts.append(piece)
            for line in piece.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                if "RGV587" in data_str or "FAIL_SYS" in data_str:
                    out_path.write_text("".join(raw_parts), encoding="utf-8")
                    return "sm", data_str[:300], len("".join(raw_parts))
                event = parse_sse_event(data_str)
                if event is None:
                    continue
                if event.get("type") == "error":
                    out_path.write_text("".join(raw_parts), encoding="utf-8")
                    return "error", str(event.get("message", event))[:300], len("".join(raw_parts))
                if event.get("type") in ("answer", "thinking", "thinking_summary"):
                    model_hits += 1
                    if not first_kind:
                        first_kind = str(event.get("type"))
                        first_snip = str(event.get("content") or "")[:120]
                    if model_hits >= max_events:
                        break
            if model_hits >= max_events:
                break

    out_path.write_text("".join(raw_parts), encoding="utf-8")
    if model_hits:
        return first_kind, first_snip, len("".join(raw_parts))
    return "no_model_content", "".join(raw_parts)[:300], len("".join(raw_parts))


async def _run_mode(
    mode: str,
    account: Account,
    prompt: str,
    model: str,
    out_dir: Path,
    *,
    token: str = "",
) -> Dict[str, Any]:
    reset_baxia_runtime()
    t0 = time.time()
    result: Dict[str, Any] = {"mode": mode, "ok": False}
    try:
        async with client_session() as http:
            if token:
                result["login"] = "token_reuse"
            else:
                token = await _login(http, account)
                result["login"] = "ok"
            if mode == "rogator":
                chat_id = await _rogator_create_chat(http, token, model)
                bx = get_baxia_tokens()
                result["bx_v"] = bx.get("bxV", "")
                ua = bx.get("bxUa", "")
                result["bx_ua"] = "231!" if ua.startswith("231!") else f"short({len(ua)})"
            else:
                chat_id = await _short_create_chat(http, token, model)
                bx = _short_baxia()
                result["bx_v"] = bx["bxV"]
                result["bx_ua"] = f"short({len(bx['bxUa'])})"
            result["chat_id"] = chat_id[:12]
            sse_path = out_dir / f"replay-{mode}.sse"
            kind, snippet, nbytes = await _collect_stream(
                http,
                mode=mode,
                token=token,
                chat_id=chat_id,
                model=model,
                prompt=prompt,
                out_path=sse_path,
            )
            result["sse_path"] = str(sse_path)
            result["sse_bytes"] = nbytes
            result["outcome"] = kind
            result["snippet"] = snippet
            result["ok"] = kind in ("answer", "thinking", "thinking_summary")
            if kind == "sm":
                result["ok"] = False
    except Exception as exc:
        result["error"] = str(exc)[:320]
    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-file",
        default=str(ROOT / "logs" / "prompts" / "req-1785757818-d8255a751e4d.txt"),
    )
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--ext-account",
        type=int,
        default=None,
        help="使用外部 accounts toml 第 N 个账号（0-based），省略则用第 0 个",
    )
    parser.add_argument(
        "--proxy",
        default="http://127.0.0.1:7890",
        help="全程 HTTP 代理，传空字符串可禁用",
    )
    parser.add_argument("--pause", type=float, default=12.0)
    parser.add_argument(
        "--use-session",
        action="store_true",
        help="从 persist/qwen/sessions.json 复用 token，跳过 login",
    )
    args = parser.parse_args(argv)

    global _PROXY_URL
    _PROXY_URL = (args.proxy or "").strip()
    if _PROXY_URL:
        os.environ["HTTP_PROXY"] = _PROXY_URL
        os.environ["HTTPS_PROXY"] = _PROXY_URL
        sync_proxy_env()
        print(f"proxy={_PROXY_URL}", flush=True)

    prompt_path = Path(args.prompt_file)
    prompt = _load_prompt(prompt_path)
    print(f"prompt={prompt_path.name} chars={len(prompt)} model={args.model}", flush=True)

    email = args.email.strip()
    password = args.password.strip()
    if args.ext_account is not None:
        email, password = _load_ext_account(args.ext_account)
        print(f"ext账号 index={args.ext_account} email={email[:6]}…", flush=True)
    elif not email:
        email, password = _load_ext_account(0)
        print(f"ext账号 index=0 email={email[:6]}…", flush=True)

    if not password and email:
        import csv

        csv_path = ROOT / "config" / "upstream" / "qwen" / "accounts.csv"
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("email") or "").strip().lower() == email.lower():
                    password = (row.get("password") or "").strip()
                    break
    if not email or not password:
        print("找不到账号密码", flush=True)
        return 2

    account = Account(username=email, password=password)
    token = ""
    if args.use_session:
        sess_path = ROOT / "persist" / "qwen" / "sessions.json"
        data = json.loads(sess_path.read_text(encoding="utf-8"))
        for s in data.get("sessions", []):
            if str(s.get("username", "")).lower() == email.lower() and s.get("is_valid"):
                token = str(s.get("token") or "")
                break
        if not token:
            print("sessions.json 中无有效 token", flush=True)
            return 2
        print("复用 sessions.json token（跳过 login）", flush=True)

    out_dir = ROOT / "logs" / "sse"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in ("rogator", "short_bx"):
        print(f"\n=== {mode.upper()} ===", flush=True)
        r = await _run_mode(mode, account, prompt, args.model, out_dir, token=token)
        results.append(r)
        status = "OK" if r.get("ok") else "FAIL"
        print(
            f"[{status}] outcome={r.get('outcome')} snippet={r.get('snippet','')[:80]!r} "
            f"bx_v={r.get('bx_v')} bx_ua={r.get('bx_ua')} sse_bytes={r.get('sse_bytes')} "
            f"elapsed={r.get('elapsed_ms')}ms",
            flush=True,
        )
        if r.get("error"):
            print(f"  error: {r['error']}", flush=True)
        if r.get("sse_path"):
            print(f"  sse: {r['sse_path']}", flush=True)
        await asyncio.sleep(args.pause)

    print("\n=== 对比 ===", flush=True)
    for r in results:
        model_resp = "有模型响应" if r.get("ok") else "无模型响应"
        print(f"{r['mode']:8} {model_resp:8} outcome={r.get('outcome')} {r.get('snippet','')[:60]!r}", flush=True)
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
