#!/usr/bin/env python3
"""Cross-account file access experiment v2.

Tests:
1. Direct OSS URL access (GET with/without auth) - the real isolation test
2. Model-based file reading via chat API (control + cross-account)
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

BASE_URL = "https://chat.qwen.ai"
AUTH_BASE_URL = "https://auth.qwen.ai"
CHAT_ORIGIN = "https://chat.qwen.ai"
CHAT_PATH = "/api/v2/chat/completions"
NEW_CHAT_PATH = "/api/v2/chats/new"
STS_TOKEN_PATHS = ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]
PARSE_PATH = "/api/v2/files/parse"
PARSE_STATUS_PATH = "/api/v2/files/parse/status"
BAXIA_VERSION = "0.0.3"
APP_VERSION = "0.2.64"
WEB_VERSION = "0.2.9"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")
SEC_CH_UA = '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"'
SEC_CH_UA_PLATFORM = '"macOS"'
MODEL = "qwen3.5-plus"
PROMPT = "Please repeat the full content of the attached file verbatim, including any secret marker. If you cannot see the attachment, say 'CANNOT_READ_ATTACHMENT'."
SESSIONS_FILE = _root / "persist" / "qwen" / "sessions.json"


def generate_fingerprint() -> str:
    return "^".join([
        uuid.uuid4().hex, "1.0.0", "web", "Chrome", "148.0.0.0",
        "zh-CN", "Asia/Shanghai", "1920x1080", "24", "Win32", "macOS",
        "Apple GPU", "Apple GPU", "desktop", "arena", "stable",
    ])


def _encode_payload(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def generate_bxua(fingerprint: str) -> str:
    return _encode_payload(f"{fingerprint}|{int(time.time() * 1000)}|{BAXIA_VERSION}")


def get_baxia_tokens() -> Dict[str, str]:
    fp = generate_fingerprint()
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    mid = "T2gA" + "".join(secrets.choice(alpha) for _ in range(40))
    return {"bxV": BAXIA_VERSION, "bxUa": generate_bxua(fp), "bxUmidToken": mid, "fingerprint": fp}


def _base_headers() -> Dict[str, str]:
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    if offset is None:
        tz_s = "+0000"
    else:
        raw = int(offset.total_seconds())
        s = "+" if raw >= 0 else "-"
        a = abs(raw)
        tz_s = f"{s}{a // 3600:02d}{(a % 3600) // 60:02d}"
    ts = now.strftime(f"%a %b %d %Y %H:%M:%S GMT{tz_s}")
    return {
        "Accept": "application/json, text/plain, */*", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive", "Content-Type": "application/json", "User-Agent": USER_AGENT,
        "Origin": CHAT_ORIGIN, "Referer": f"{CHAT_ORIGIN}/", "Source": "web",
        "X-Request-Id": str(uuid.uuid4()), "Timezone": ts,
        "Sec-Ch-Ua": SEC_CH_UA, "Sec-Ch-Ua-Mobile": "?0", "Sec-Ch-Ua-Platform": SEC_CH_UA_PLATFORM,
    }


def build_headers(token: str, *, chat_id: str = "", include_sse: bool = False,
                  include_version: bool = True) -> Dict[str, str]:
    h = _base_headers()
    h["Authorization"] = f"Bearer {token}"
    b = get_baxia_tokens()
    h["bx-v"], h["bx-ua"], h["bx-umidtoken"] = b["bxV"], b["bxUa"], b["bxUmidToken"]
    if include_version:
        h["version"] = WEB_VERSION
    if chat_id:
        h["Referer"] = f"{CHAT_ORIGIN}/c/{chat_id}"
    if include_sse:
        h["X-Accel-Buffering"] = "no"
    return h


def build_oss_authorization(method: str, content_type: str, date: str,
                            oss_headers: Dict[str, str], resource: str,
                            access_key_id: str, access_key_secret: str) -> str:
    canonical = "".join(f"{k.lower()}:{v}\n" for k, v in sorted(oss_headers.items()))
    sts = f"{method}\n\n{content_type}\n{date}\n{canonical}{resource}"
    sig = base64.b64encode(hmac.new(access_key_secret.encode(), sts.encode(), hashlib.sha1).digest()).decode()
    return f"OSS {access_key_id}:{sig}"


def load_two_accounts() -> tuple[Dict[str, Any], Dict[str, Any]]:
    data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    valid = [s for s in sessions if s.get("is_valid") and s.get("token")]
    if len(valid) < 2:
        raise RuntimeError(f"Need at least 2 valid accounts, found {len(valid)}")
    return valid[0], valid[1]


async def _get_sts_token(session: aiohttp.ClientSession, token: str,
                         filename: str, filesize: int, filetype: str) -> Dict[str, Any]:
    headers = {"authorization": f"Bearer {token}", "content-type": "application/json;charset=UTF-8",
               "source": "web", "user-agent": USER_AGENT, "origin": BASE_URL,
               "referer": f"{BASE_URL}/", "accept": "application/json"}
    payload = {"filename": filename, "filesize": filesize, "filetype": filetype}
    last_error: Optional[Exception] = None
    for path in STS_TOKEN_PATHS:
        try:
            async with session.post(f"{BASE_URL}{path}", json=payload, headers=headers,
                                    ssl=False, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    last_error = RuntimeError(f"STS HTTP {resp.status}")
                    continue
                data = await resp.json()
                creds = data.get("data", data)
                if all(k in creds for k in ("access_key_id", "access_key_secret", "security_token")):
                    return creds
                last_error = RuntimeError(f"invalid STS: {data}")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"all STS endpoints failed: {last_error}")


def _resolve_oss_hosts(bucket_host: str) -> list[tuple[str, str]]:
    import socket
    def _try_resolve(host: str) -> Optional[str]:
        try:
            addrs = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
            return addrs[0][4][0] if addrs else None
        except OSError:
            return None
    ip = _try_resolve(bucket_host)
    cands: list[tuple[str, str]] = []
    if ip:
        cands.append((ip, bucket_host))
    else:
        print(f"  [upload] DNS resolve failed for {bucket_host}, trying fallbacks...")
        cands.append(("47.113.75.199", bucket_host))
        if "oss-accelerate" in bucket_host:
            fb = bucket_host.replace("oss-accelerate", "oss")
            fb_ip = _try_resolve(fb)
            cands.append((fb_ip or fb, fb if not fb_ip else fb))
    return cands


async def _upload_to_oss(session: aiohttp.ClientSession, file_data: bytes,
                         content_type: str, creds: Dict[str, Any]) -> str:
    file_url = str(creds.get("file_url", ""))
    object_key = str(creds.get("file_path", ""))
    bucket_host = urlparse(file_url).netloc
    host_candidates = _resolve_oss_hosts(bucket_host)
    sec_token = str(creds.get("security_token", ""))
    ak_id = str(creds.get("access_key_id", ""))
    ak_secret = str(creds.get("access_key_secret", ""))
    last_error: Optional[Exception] = None
    for connect_host, header_host in host_candidates:
        resource = f"/{header_host.split('.')[0]}/{object_key}"
        date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        auth = build_oss_authorization("PUT", content_type, date,
                                       {"x-oss-security-token": sec_token}, resource, ak_id, ak_secret)
        headers = {"Host": header_host, "Date": date, "Content-Type": content_type,
                   "Content-Length": str(len(file_data)), "Authorization": auth,
                   "x-oss-security-token": sec_token, "User-Agent": USER_AGENT}
        try:
            async with session.put(f"https://{connect_host}/{object_key}", data=file_data,
                                   headers=headers, ssl=False,
                                   timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status not in (200, 201):
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                return file_url
        except Exception as exc:
            last_error = exc
            print(f"  [upload] OSS {header_host} (via {connect_host}) failed: {exc}")
    raise RuntimeError(f"all OSS hosts failed: {last_error}")


async def upload_file(session: aiohttp.ClientSession, token: str, user_id: str,
                      file_data: bytes, filename: str) -> Dict[str, Any]:
    content_type = "text/plain"
    file_type = "file"
    file_size = len(file_data)
    print(f"  [upload] file: {filename}  size: {file_size} bytes  type: {file_type}")
    creds = await _get_sts_token(session, token, filename, file_size, file_type)
    file_url = await _upload_to_oss(session, file_data, content_type, creds)
    file_id = str(creds.get("file_id", uuid.uuid4()))
    print(f"  [upload] done  url: {file_url[:80]}...")
    print(f"  [upload] file_id: {file_id}")

    # Trigger document parse (required for type=file)
    print(f"  [parse] triggering parse for file_id={file_id}")
    parse_headers = {"authorization": f"Bearer {token}", "content-type": "application/json;charset=UTF-8",
                     "source": "web", "user-agent": USER_AGENT, "origin": BASE_URL,
                     "referer": f"{BASE_URL}/", "accept": "application/json"}
    try:
        async with session.post(f"{BASE_URL}{PARSE_PATH}", json={"file_id": file_id},
                                headers=parse_headers, ssl=False,
                                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            parse_resp = await resp.json()
            print(f"  [parse] trigger response: {json.dumps(parse_resp, ensure_ascii=False)[:200]}")
    except Exception as exc:
        print(f"  [parse] trigger failed: {exc}")

    # Poll parse status
    for attempt in range(30):
        await asyncio.sleep(2)
        try:
            async with session.post(f"{BASE_URL}{PARSE_STATUS_PATH}",
                                    json={"file_id_list": [file_id]},
                                    headers=parse_headers, ssl=False,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                status_data = await resp.json()
                items = (status_data.get("data") or {}).get("items") or status_data.get("data") or []
                if isinstance(items, list) and items:
                    item = items[0]
                    st = item.get("status", "")
                    print(f"  [parse] poll {attempt+1}: status={st}")
                    if st in ("success", "done", "completed"):
                        break
                    if st in ("failed", "error"):
                        print(f"  [parse] FAILED: {item}")
                        break
                else:
                    print(f"  [parse] poll {attempt+1}: {json.dumps(status_data, ensure_ascii=False)[:200]}")
        except Exception as exc:
            print(f"  [parse] poll error: {exc}")

    return {"id": file_id, "name": filename,
            "type": file_type, "size": file_size, "url": file_url,
            "file_type": content_type, "showType": file_type, "file_class": file_type,
            "user_id": user_id, "isQuote": False}


async def create_chat(session: aiohttp.ClientSession, token: str, model: str) -> str:
    payload = {"title": "Cross-account file test", "models": [model], "chat_mode": "local",
               "chat_type": "t2t", "timestamp": int(time.time() * 1000), "project_id": ""}
    async with session.post(f"{BASE_URL}{NEW_CHAT_PATH}", json=payload,
                            headers=build_headers(token, include_version=False),
                            ssl=False, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"create chat HTTP {resp.status}: {(await resp.text())[:300]}")
        data = await resp.json()
        chat_id = str((data.get("data") or {}).get("id", ""))
        if not data.get("success") or not chat_id:
            raise RuntimeError(f"invalid create chat: {data}")
        print(f"  [chat] chat_id: {chat_id}")
        return chat_id


def _build_chat_payload(chat_id: str, model: str, user_message: str,
                        files: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "stream": True, "version": "2.1", "incremental_output": True,
        "chat_id": chat_id, "chat_mode": "local", "model": model, "parent_id": None,
        "messages": [{"fid": str(uuid.uuid4()), "parentId": None,
                      "childrenIds": [str(uuid.uuid4())], "role": "user",
                      "content": user_message, "user_action": "chat", "files": files,
                      "timestamp": int(time.time() * 1000), "models": [model],
                      "chat_type": "t2t", "feature_config": {
                          "thinking_enabled": True, "output_schema": "phase",
                          "research_mode": "normal", "auto_thinking": False,
                          "thinking_mode": "Thinking", "thinking_format": "raw", "auto_search": False},
                      "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t"}],
        "timestamp": int(time.time() * 1000),
    }


async def stream_chat(session: aiohttp.ClientSession, token: str, chat_id: str,
                      model: str, user_message: str,
                      files: List[Dict[str, Any]]) -> list[str]:
    """Send message and collect answer parts from SSE stream."""
    payload = _build_chat_payload(chat_id, model, user_message, files)
    headers = build_headers(token, chat_id=chat_id, include_sse=True)
    answer_parts: list[str] = []
    event_count = 0
    buffer = ""

    async with session.post(f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                            headers=headers, ssl=False,
                            timeout=aiohttp.ClientTimeout(total=120)) as resp:
        if resp.status != 200:
            body = await resp.text()
            print(f"  [chat] ERROR HTTP {resp.status}: {body[:500]}")
            return [f"[ERROR] HTTP {resp.status}: {body[:300]}"]

        async for chunk in resp.content.iter_chunked(4096):
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except Exception:
                    continue
                event_count += 1
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                phase = delta.get("phase", "")
                content = delta.get("content", "")
                role = delta.get("role", "")
                if phase == "answer" and content:
                    answer_parts.append(content)
                    print(content, end="", flush=True)
                elif phase == "think" and content:
                    pass  # skip thinking output
                elif role == "function":
                    fn_content = delta.get("content", "")
                    fn_name = delta.get("name", "")
                    print(f"\n  [WARN] function role detected: name={fn_name} content={fn_content[:100]}")

    print(f"\n  [debug] total SSE events: {event_count}, answer parts: {len(answer_parts)}")
    return answer_parts


async def test_direct_url_access(session: aiohttp.ClientSession, file_url: str,
                                 label: str, token: str = "") -> tuple[bool, str]:
    """Directly GET the OSS file URL to test access control."""
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with session.get(file_url, headers=headers, ssl=False,
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            status = resp.status
            body = await resp.text(errors="replace")
            accessible = status == 200 and len(body) > 10
            snippet = body[:200].replace("\n", "\\n")
            print(f"  [{label}] HTTP {status}  len={len(body)}  accessible={accessible}")
            print(f"  [{label}] body preview: {snippet}")
            return accessible, body
    except Exception as exc:
        print(f"  [{label}] ERROR: {exc}")
        return False, str(exc)


async def main() -> None:
    print("=" * 70)
    print("Cross-Account File Access Experiment v2")
    print("=" * 70)

    acct_a, acct_b = load_two_accounts()
    print(f"\nAccount A (uploader): {acct_a['username']}  user_id={acct_a.get('user_id', '?')}")
    print(f"Account B (reader):   {acct_b['username']}  user_id={acct_b.get('user_id', '?')}")

    secret_marker = f"CROSS_ACCT_SECRET_{uuid.uuid4().hex[:12]}"
    file_content = f"""This is a cross-account file access test file.
Created: {datetime.now().isoformat()}
Secret marker: {secret_marker}
If you can see this text, cross-account file access works.
Please repeat everything above including the secret marker.
"""
    file_bytes = file_content.encode("utf-8")
    filename = f"cross_acct_test_{uuid.uuid4().hex[:8]}.txt"

    async with aiohttp.ClientSession() as session:
        # === Step 1: Account A uploads ===
        print(f"\n{'=' * 70}")
        print("[Step 1] Account A uploads file")
        print(f"{'=' * 70}")
        token_a = acct_a["token"]
        user_id_a = acct_a.get("user_id", "")
        file_obj = await upload_file(session, token_a, user_id_a, file_bytes, filename)
        file_url = file_obj["url"]
        print(f"  file_url: {file_url[:100]}...")
        print(f"  file_id:  {file_obj['id']}")

        # === Step 2: Direct URL access tests ===
        print(f"\n{'=' * 70}")
        print("[Step 2] Direct OSS URL access tests")
        print(f"{'=' * 70}")

        # 2a: No auth
        print("\n  --- 2a: No authentication ---")
        no_auth_ok, no_auth_body = await test_direct_url_access(
            session, file_url, "no-auth")

        # 2b: Account A's token
        print("\n  --- 2b: Account A token ---")
        a_auth_ok, a_auth_body = await test_direct_url_access(
            session, file_url, "acct-a", token_a)

        # 2c: Account B's token
        print("\n  --- 2c: Account B token ---")
        token_b = acct_b["token"]
        b_auth_ok, b_auth_body = await test_direct_url_access(
            session, file_url, "acct-b", token_b)

        # Check if secret marker is in the downloaded content
        a_has_secret = secret_marker in a_auth_body
        b_has_secret = secret_marker in b_auth_body
        no_auth_has_secret = secret_marker in no_auth_body

        # === Step 3: Model-based reading (control) ===
        print(f"\n{'=' * 70}")
        print("[Step 3] Account A reads own file via model (control)")
        print(f"{'=' * 70}")
        chat_id_a = await create_chat(session, token_a, MODEL)
        print(f"  prompt: {PROMPT}")
        print(f"  response: ", end="")
        answer_a = await stream_chat(session, token_a, chat_id_a, MODEL, PROMPT, [file_obj])
        full_answer_a = "".join(answer_a)
        a_model_can_read = secret_marker in full_answer_a
        print(f"  [RESULT] Account A model can read: {a_model_can_read}")

        # === Step 4: Model-based reading (cross-account) ===
        print(f"\n{'=' * 70}")
        print("[Step 4] Account B reads Account A's file via model (experiment)")
        print(f"{'=' * 70}")
        chat_id_b = await create_chat(session, token_b, MODEL)
        print(f"  using Account A's file_obj: id={file_obj['id']}")
        print(f"  prompt: {PROMPT}")
        print(f"  response: ", end="")
        answer_b = await stream_chat(session, token_b, chat_id_b, MODEL, PROMPT, [file_obj])
        full_answer_b = "".join(answer_b)
        b_model_can_read = secret_marker in full_answer_b
        print(f"  [RESULT] Account B model can read: {b_model_can_read}")

        # === Summary ===
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Secret marker: {secret_marker}")
        print()
        print("  Direct URL access:")
        print(f"    No auth:           accessible={no_auth_ok}  has_secret={no_auth_has_secret}")
        print(f"    Account A token:   accessible={a_auth_ok}  has_secret={a_has_secret}")
        print(f"    Account B token:   accessible={b_auth_ok}  has_secret={b_has_secret}")
        print()
        print("  Model-based reading:")
        print(f"    Account A (owner): {a_model_can_read}")
        print(f"    Account B (other): {b_model_can_read}")
        print()

        if b_has_secret:
            print("  [WARN] OSS URL is accessible cross-account! File content leaked.")
        elif no_auth_has_secret:
            print("  [WARN] OSS URL is publicly accessible without any auth!")
        else:
            print("  [OK] Direct URL access is properly isolated.")

        if b_model_can_read:
            print("  [WARN] Model can read cross-account files via chat API!")
        else:
            print("  [OK] Model-based cross-account file reading is blocked.")

        print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
