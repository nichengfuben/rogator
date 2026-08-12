#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短 bx-ua vs fireye 账号 × Baxia 策略矩阵 live 探测（只读诊断）。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

import aiohttp

from upstream.qwen.auth.crypto import (
    build_login_headers,
    generate_bxua,
    generate_fingerprint,
    hash_password,
)
from upstream.qwen.auth.baxia_runtime import get_baxia_tokens, reset_baxia_runtime
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

EXT_ACCOUNTS_TOML = Path(
    os.environ.get(
        "QWEN_EXT_ACCOUNTS",
        str(ROOT / "config" / "upstream" / "qwen" / "ext_accounts.toml"),
    )
)
ROGATOR_CSV = ROOT / "config" / "upstream" / "qwen" / "accounts.csv"

EXT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
EXT_SEC_CH_UA = '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"'


@dataclass(frozen=True)
class CaseAccount:
    label: str
    username: str
    password: str
    source: str


@dataclass
class ProbeResult:
    account: str
    mode: str
    stage: str
    ok: bool
    detail: str
    bx_v: str = ""
    bx_ua_kind: str = ""
    elapsed_ms: int = 0


def _mask(email: str) -> str:
    local, _, domain = email.partition("@")
    head = local[:6] if len(local) >= 6 else local
    return f"{head}…@{domain}"


def _load_ext_accounts(limit: int) -> List[CaseAccount]:
    if not EXT_ACCOUNTS_TOML.is_file():
        return []
    text = EXT_ACCOUNTS_TOML.read_text(encoding="utf-8")
    blocks = re.findall(
        r'\[\[accounts\]\]\s*\nusername\s*=\s*"([^"]+)"\s*\npassword\s*=\s*"([^"]+)"',
        text,
    )
    out: List[CaseAccount] = []
    for idx, (user, pwd) in enumerate(blocks[:limit]):
        out.append(CaseAccount(f"ext#{idx + 1}", user, pwd, "ext"))
    return out


def _load_rogator_accounts(keys: List[str]) -> List[CaseAccount]:
    if not ROGATOR_CSV.is_file():
        return []
    rows: Dict[str, Tuple[str, str]] = {}
    with ROGATOR_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip()
            password = (row.get("password") or "").strip()
            if email and password:
                rows[email.lower()] = (email, password)
    out: List[CaseAccount] = []
    for key in keys:
        key_l = key.lower()
        for email, (user, pwd) in rows.items():
            if key_l in email.lower():
                out.append(CaseAccount(f"rogator:{key}", user, pwd, "rogator"))
                break
    return out


def _base_headers(*, user_agent: str, sec_ch_ua: str) -> Dict[str, str]:
    from upstream.qwen.auth.crypto import make_request_id, make_timezone

    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
        "Origin": CHAT_ORIGIN,
        "Referer": f"{CHAT_ORIGIN}/",
        "source": "web",
        "X-Request-Id": make_request_id(),
        "Timezone": make_timezone(),
        "Sec-Ch-Ua": sec_ch_ua,
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": SEC_CH_UA_PLATFORM,
    }


def _short_baxia() -> Dict[str, str]:
    fp = generate_fingerprint()
    return {
        "bxV": "0.0.3",
        "bxUa": generate_bxua(fp),
        "bxUmidToken": "T2gA" + uuid.uuid4().hex[:40] + "=",
        "fingerprint": fp,
    }


def _bx_kind(ua: str) -> str:
    if ua.startswith("231!"):
        return f"231!({len(ua)})"
    if not ua:
        return "empty"
    return f"short({len(ua)})"


async def _login(http: aiohttp.ClientSession, account: CaseAccount) -> str:
    payload = {
        "email": account.username,
        "password": hash_password(account.password),
        "remember_me": True,
    }
    async with http.post(
        f"{BASE_URL}/api/v2/auths/signin",
        json=payload,
        headers=build_login_headers(),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"login HTTP {resp.status}: {text[:200]}")
        data = json.loads(text)
        if not data.get("success"):
            raise RuntimeError(f"login failed: {str(data)[:200]}")
        token = str((data.get("data") or {}).get("token") or "")
        if not token:
            raise RuntimeError("login missing token")
        return token


def _headers_for_mode(
    mode: str,
    token: str,
    *,
    chat_id: str = "",
    include_sse: bool = False,
) -> Dict[str, str]:
    if mode == "short_bx":
        headers = _base_headers(user_agent=EXT_UA, sec_ch_ua=EXT_SEC_CH_UA)
        baxia = _short_baxia()
        headers["bx-v"] = baxia["bxV"]
        headers["bx-ua"] = baxia["bxUa"]
        headers["bx-umidtoken"] = baxia["bxUmidToken"]
        headers["Cookie"] = f"token={token}"
        if include_sse:
            headers["Accept"] = "text/event-stream"
            headers["X-Accel-Buffering"] = "no"
        if chat_id:
            headers["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
        return headers

    use_bearer = mode != "rogator_cookie"
    reset_baxia_runtime()
    if mode == "rogator_short_bx":
        baxia = _short_baxia()
        baxia["bxV"] = "2.5.37"
    else:
        baxia = get_baxia_tokens()

    headers = _base_headers(user_agent=USER_AGENT, sec_ch_ua=SEC_CH_UA)
    if use_bearer:
        headers["Authorization"] = f"Bearer {token}"
    headers["bx-v"] = baxia["bxV"]
    headers["bx-ua"] = baxia["bxUa"]
    headers["bx-umidtoken"] = baxia["bxUmidToken"]
    headers["Version"] = APP_VERSION
    headers["Cookie"] = f"token={token}"
    if include_sse:
        headers["Accept"] = "text/event-stream"
        headers["X-Accel-Buffering"] = "no"
    if chat_id:
        headers["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
    return headers


async def _create_chat(
    http: aiohttp.ClientSession,
    token: str,
    model: str,
    mode: str,
) -> str:
    headers = _headers_for_mode(mode, token)
    if mode == "short_bx":
        payload = {
            "title": "新建对话",
            "models": [model],
            "chat_mode": "local",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000),
            "project_id": "",
        }
    else:
        payload = build_new_chat_payload(model)

    async with http.post(
        f"{BASE_URL}{NEW_CHAT_PATH}",
        json=payload,
        headers=headers,
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=45),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"create_chat HTTP {resp.status}: {text[:240]}")
        data = json.loads(text)
        if not data.get("success"):
            raise RuntimeError(f"create_chat fail: {str(data)[:240]}")
        chat_id = str((data.get("data") or {}).get("id") or "")
        if not chat_id:
            raise RuntimeError(f"create_chat no id: {str(data)[:240]}")
        return chat_id


def _completion_payload(chat_id: str, model: str, message: str) -> Dict[str, Any]:
    fid = str(uuid.uuid4())
    child = str(uuid.uuid4())
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "local",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": fid,
                "parentId": None,
                "childrenIds": [child],
                "role": "user",
                "content": message,
                "user_action": "chat",
                "files": [],
                "timestamp": int(time.time() * 1000),
                "models": [model],
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
            }
        ],
        "timestamp": int(time.time() * 1000),
    }


async def _stream_first_content(
    http: aiohttp.ClientSession,
    token: str,
    chat_id: str,
    model: str,
    mode: str,
) -> Tuple[str, str]:
    headers = _headers_for_mode(mode, token, chat_id=chat_id, include_sse=True)
    payload = _completion_payload(chat_id, model, "回复 OK 两个汉字即可。")
    url = f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}"
    async with http.post(
        url,
        json=payload,
        headers=headers,
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"completion HTTP {resp.status}: {text[:280]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            event = parse_sse_event(data_str)
            if event is None:
                continue
            if event.get("type") == "error":
                raise RuntimeError(str(event.get("message", event))[:280])
            if event.get("type") in ("answer", "thinking"):
                content = str(event.get("content") or "")[:80]
                return event["type"], content
    raise RuntimeError("completion ended without content")


async def _probe_account(account: CaseAccount, mode: str, model: str) -> ProbeResult:
    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as http:
            token = await _login(http, account)
            chat_id = await _create_chat(http, token, model, mode)
            kind, snippet = await _stream_first_content(http, token, chat_id, model, mode)
            headers = _headers_for_mode(mode, token)
            elapsed = int((time.time() - t0) * 1000)
            return ProbeResult(
                account=account.label,
                mode=mode,
                stage="completion",
                ok=True,
                detail=f"{kind}:{snippet!r}",
                bx_v=headers.get("bx-v", ""),
                bx_ua_kind=_bx_kind(headers.get("bx-ua", "")),
                elapsed_ms=elapsed,
            )
    except Exception as exc:
        elapsed = int((time.time() - t0) * 1000)
        msg = str(exc)
        stage = "login"
        if "create_chat" in msg:
            stage = "create_chat"
        elif "completion" in msg or "RGV587" in msg or "SM" in msg:
            stage = "completion"
        return ProbeResult(
            account=account.label,
            mode=mode,
            stage=stage,
            ok=False,
            detail=msg[:320],
            elapsed_ms=elapsed,
        )


MODES = (
    "short_bx",
    "rogator_default",
    "rogator_short_bx",
    "rogator_cookie",
)


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen live matrix: short bx-ua vs fireye")
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--ext-count", type=int, default=1)
    parser.add_argument(
        "--rogator-keys",
        default="psnony,ggobef",
        help="email 子串，逗号分隔",
    )
    args = parser.parse_args(argv)

    accounts: List[CaseAccount] = []
    accounts.extend(_load_ext_accounts(args.ext_count))
    accounts.extend(_load_rogator_accounts([k.strip() for k in args.rogator_keys.split(",") if k.strip()]))
    if not accounts:
        print("未找到任何账号", flush=True)
        return 2

    print(f"模型={args.model}  账号数={len(accounts)}  模式={','.join(MODES)}", flush=True)
    for acc in accounts:
        print(f"  - {acc.label} {_mask(acc.username)} ({acc.source})", flush=True)
    print(flush=True)

    results: List[ProbeResult] = []
    for acc in accounts:
        for mode in MODES:
            res = await _probe_account(acc, mode, args.model)
            results.append(res)
            status = "OK" if res.ok else "FAIL"
            extra = ""
            if res.bx_v:
                extra = f" bx-v={res.bx_v} bx-ua={res.bx_ua_kind}"
            print(
                f"[{status}] {acc.label:16} {mode:22} {res.stage:12} {res.elapsed_ms:5}ms "
                f"{res.detail[:120]}{extra}",
                flush=True,
            )

    print("\n=== 汇总 ===", flush=True)
    by_acc: Dict[str, List[ProbeResult]] = {}
    for r in results:
        by_acc.setdefault(r.account, []).append(r)
    for label, rows in by_acc.items():
        ok_modes = [r.mode for r in rows if r.ok]
        fail_modes = [f"{r.mode}@{r.stage}" for r in rows if not r.ok]
        print(f"{label}: OK={ok_modes or '-'}  FAIL={fail_modes or '-'}", flush=True)

    fails = sum(1 for r in results if not r.ok)
    return 0 if fails < len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
