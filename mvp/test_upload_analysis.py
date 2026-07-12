#!/usr/bin/env python3
"""Test: login -> upload document -> send to model for analysis. Self-contained."""
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
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse
import aiohttp

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from accounts import ACCOUNTS

BASE_URL = "https://chat.qwen.ai"
AUTH_BASE_URL = "https://auth.qwen.ai"
CHAT_ORIGIN = "https://chat.qwen.ai"
CHAT_PATH = "/api/v2/chat/completions"
NEW_CHAT_PATH = "/api/v2/chats/new"
STS_TOKEN_PATHS = ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]
BAXIA_VERSION = "0.0.3"
APP_VERSION = "0.2.64"
WEB_VERSION = "0.2.9"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36")
SEC_CH_UA = '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"'
SEC_CH_UA_PLATFORM = '"macOS"'
MODEL = "qwen3.5-plus"
PROMPT = "请分析以下文档的内容，给出摘要、要��列表和关键发现。如果文档很长，分段总结即可。"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

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

def build_login_headers() -> Dict[str, str]:
    h = _base_headers()
    h["Version"] = APP_VERSION
    h["x-request-origin"] = BASE_URL
    return h

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

def get_credentials() -> tuple[str, str]:
    email = os.environ.get("GENERALUSR", "").strip()
    password = os.environ.get("GENERALPWD", "").strip()
    if email and password:
        return email, password
    if not ACCOUNTS:
        raise SystemExit("no Qwen accounts configured")
    return ACCOUNTS[0].username, ACCOUNTS[0].password

async def login(session: aiohttp.ClientSession, email: str, password: str) -> str:
    payload = {"email": email, "password": hash_password(password), "remember_me": True}
    async with session.post(f"{AUTH_BASE_URL}/api/v2/auths/signin", json=payload,
                            headers=build_login_headers(), ssl=False,
                            timeout=aiohttp.ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"login HTTP {resp.status}: {(await resp.text())[:300]}")
        data = await resp.json()
        token = (data.get("data") or {}).get("access_token", "")
        if not token:
            raise RuntimeError(f"login missing token: {data}")
        return token

async def get_user_id(session: aiohttp.ClientSession, token: str) -> str:
    async with session.get(f"{AUTH_BASE_URL}/api/v2/user",
                           headers=build_headers(token, include_version=False),
                           ssl=False, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status != 200:
            return ""
        data = await resp.json()
        return str((data.get("data") or {}).get("id", ""))

DOCUMENT_URLS = [
    ("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", "War and Peace (Tolstoy)"),
    ("https://www.gutenberg.org/cache/epub/11/pg11.txt", "Alice in Wonderland (Carroll)"),
    ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein (Shelley)"),
]

def download_document(url: str, save_path: str) -> str:
    import urllib.request
    print(f"[doc] downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(save_path, "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
    content = Path(save_path).read_text(encoding="utf-8", errors="replace")
    print(f"[doc] downloaded: {len(content.encode('utf-8')) / 1024:.1f} KB")
    return content

def get_document() -> str:
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    for url, name in DOCUMENT_URLS:
        fn = url.split("/")[-1]
        cp = cache_dir / fn
        fp = cache_dir / f"{fn.replace('.txt', '')}_full.txt"
        for path in [fp, cp]:
            if path.exists() and path.stat().st_size > 10000:
                print(f"[doc] using cached: {name} ({path.stat().st_size / 1024:.1f} KB)")
                return path.read_text(encoding="utf-8", errors="replace")
        try:
            content = download_document(url, str(cp))
            print(f"[doc] got: {name}")
            return content
        except Exception as exc:
            print(f"[doc] failed to download {name}: {exc}")
    raise RuntimeError("无法下载任何测试文档，请检查网络连接")

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
        print(f"[upload] DNS resolve failed for {bucket_host}, trying fallbacks...")
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
        if connect_host != header_host:
            print(f"[upload] using IP {connect_host} with Host: {header_host}")
        try:
            async with session.put(f"https://{connect_host}/{object_key}", data=file_data,
                                   headers=headers, ssl=False,
                                   timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status not in (200, 201):
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                return file_url
        except Exception as exc:
            last_error = exc
            print(f"[upload] OSS {header_host} (via {connect_host}) failed: {exc}")
    raise RuntimeError(f"all OSS hosts failed: {last_error}")

def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {".txt": "text/plain", ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".py": "text/x-python", ".js": "application/javascript"}.get(ext, "application/octet-stream")

def _guess_file_type(mime: str) -> str:
    for p in ("image/", "video/", "audio/"):
        if mime.startswith(p):
            return p.rstrip("/")
    return "file"

async def upload_file(session: aiohttp.ClientSession, token: str, user_id: str,
                      file_data: bytes, filename: str) -> Dict[str, Any]:
    content_type = _guess_mime(filename)
    file_type = _guess_file_type(content_type)
    file_size = len(file_data)
    print(f"[upload] file: {filename}  size: {file_size / 1024 / 1024:.2f} MB  type: {file_type}")
    creds = await _get_sts_token(session, token, filename, file_size, file_type)
    file_url = await _upload_to_oss(session, file_data, content_type, creds)
    print(f"[upload] done url: {file_url[:80]}...")
    return {"id": str(creds.get("file_id", uuid.uuid4())), "name": filename,
            "type": file_type, "size": file_size, "url": file_url,
            "file_type": content_type, "showType": file_type, "file_class": file_type,
            "user_id": user_id, "isQuote": False}

async def create_chat(session: aiohttp.ClientSession, token: str, model: str) -> str:
    payload = {"title": "Doc analysis test", "models": [model], "chat_mode": "local",
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
        print(f"[chat] chat_id: {chat_id}")
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
                      files: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
    payload = _build_chat_payload(chat_id, model, user_message, files)
    headers = build_headers(token, chat_id=chat_id, include_sse=True)
    async with session.post(f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}", json=payload,
                            headers=headers, ssl=False,
                            timeout=aiohttp.ClientTimeout(total=600)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"chat HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
            except Exception:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            phase, content = delta.get("phase", ""), delta.get("content", "")
            if phase == "think" and content:
                yield {"type": "thinking", "content": content}
            elif phase == "answer" and content:
                yield {"type": "answer", "content": content}
    yield {"type": "done"}

async def _collect_response(session: aiohttp.ClientSession, token: str, chat_id: str,
                            model: str, prompt: str,
                            files: List[Dict[str, Any]]) -> tuple[list[str], list[str]]:
    answer_parts: list[str] = []
    think_parts: list[str] = []
    async for event in stream_chat(session, token, chat_id, model, prompt, files):
        if event["type"] == "thinking":
            think_parts.append(event["content"])
        elif event["type"] == "answer":
            answer_parts.append(event["content"])
            print(event["content"], end="", flush=True)
        elif event["type"] == "done":
            break
        elif event["type"] == "error":
            print(f"\n  ERROR: {event['content']}")
            return think_parts, answer_parts
    return think_parts, answer_parts

async def main() -> None:
    print("=" * 60)
    print("Qwen Document Upload & Analysis Test")
    print("=" * 60)
    email, password = get_credentials()
    print(f"\n[1/5] account: {email}")
    async with aiohttp.ClientSession() as session:
        print("\n[2/5] logging in...")
        token = await login(session, email, password)
        user_id = await get_user_id(session, token)
        print(f"  OK  token: {token[:16]}...  user_id: {user_id}")
        print("\n[3/5] downloading real document from internet...")
        doc_bytes = get_document().encode("utf-8")
        doc_filename = f"random_doc_{uuid.uuid4().hex[:8]}.txt"
        print(f"\n[4/5] uploading: {doc_filename}")
        file_obj = await upload_file(session, token, user_id, doc_bytes, doc_filename)
        print(f"\n[5/5] sending to model ({MODEL})...")
        print(f"  prompt: {PROMPT}")
        chat_id = await create_chat(session, token, MODEL)
        think_parts, answer_parts = await _collect_response(
            session, token, chat_id, MODEL, PROMPT, [file_obj])
        print("\n" + "=" * 60)
        print("Done!")
        print(f"  thinking length: {sum(len(p) for p in think_parts)} chars")
        print(f"  answer length:   {sum(len(p) for p in answer_parts)} chars")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
