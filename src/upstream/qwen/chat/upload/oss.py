from __future__ import annotations

"""OSS upload helpers with time synchronization and DNS fallback."""

import base64
import hashlib
import hmac
import logging
import re
import socket
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from server.model.model_thinking import ThinkingRoute
from upstream.qwen.chat.routes import USER_AGENT

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

STS_TOKEN_PATHS = ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]


async def _request_sts_credentials(
    url: str, payload: Dict[str, Any], headers: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    async with aiohttp.ClientSession() as s:
        async with s.post(
            url, json=payload, headers=headers, ssl=False,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            creds = data.get("data", data)
            required = ("access_key_id", "access_key_secret", "security_token")
            if all(k in creds for k in required):
                return creds
            return None


async def get_sts_credentials(
    base_url: str, token: str, filename: str, filesize: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        "Origin": base_url,
        "Referer": f"{base_url}/",
        "Source": "web",
    }
    payload = {"filename": filename, "filesize": str(filesize), "filetype": "file"}
    for path in STS_TOKEN_PATHS:
        try:
            creds = await _request_sts_credentials(f"{base_url}{path}", payload, headers)
            if creds:
                return creds
        except Exception:
            continue
    raise RuntimeError("All STS endpoints failed")


def build_oss_authorization(
    method: str, content_type: str, date: str,
    oss_headers: Dict[str, str], resource: str,
    access_key_id: str, access_key_secret: str,
) -> str:
    canonical_oss_headers = ""
    for key in sorted(oss_headers):
        canonical_oss_headers += f"{key.lower()}:{oss_headers[key]}\n"
    string_to_sign = f"{method}\n\n{content_type}\n{date}\n{canonical_oss_headers}{resource}"
    signature = base64.b64encode(
        hmac.new(
            access_key_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("utf-8")
    return f"OSS {access_key_id}:{signature}"


def build_upload_headers(
    header_host: str, oss_date: str, content_type: str,
    file_size: int, authorization: str, security_token: str,
) -> Dict[str, str]:
    return {
        "Host": header_host,
        "Date": oss_date,
        "Content-Type": content_type,
        "Content-Length": str(file_size),
        "Authorization": authorization,
        "x-oss-security-token": security_token,
        "User-Agent": USER_AGENT,
    }


async def get_oss_time(host: str) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.head(
                f"https://{host}", ssl=False,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                date_header = resp.headers.get("Date")
                if date_header:
                    logger.debug("Got OSS time: %s", date_header)
                    return date_header
    except Exception as e:
        logger.debug("Failed to get OSS time from %s: %s", host, e)
    return None


async def sync_time_with_oss(host: str) -> float:
    oss_date = await get_oss_time(host)
    if oss_date:
        try:
            dt = datetime.strptime(oss_date, "%a, %d %b %Y %H:%M:%S %Z")
            oss_timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
            offset = oss_timestamp - time.time()
            logger.debug("OSS time offset: %.2f seconds", offset)
            return offset
        except Exception as e:
            logger.debug("Failed to parse OSS time: %s", e)
    return 0.0


def _try_resolve(host: str) -> Optional[str]:
    try:
        addrs = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
        return addrs[0][4][0] if addrs else None
    except OSError:
        return None


def resolve_oss_hosts(bucket_host: str) -> list[tuple[str, str]]:
    ip = _try_resolve(bucket_host)
    candidates: list[tuple[str, str]] = []
    if ip:
        candidates.append((ip, bucket_host))
        return candidates
    logger.debug("DNS resolve failed for %s, trying fallbacks...", bucket_host)
    for known_ip in ["47.113.75.199"]:
        candidates.append((known_ip, bucket_host))
    if "oss-accelerate" in bucket_host:
        fallback = bucket_host.replace("oss-accelerate", "oss")
        fb_ip = _try_resolve(fallback)
        if fb_ip:
            candidates.append((fb_ip, fallback))
        else:
            candidates.append((fallback, fallback))
    return candidates


async def try_oss_upload(
    connect_host: str, header_host: str, object_key: str,
    file_data: bytes, content_type: str, oss_date: str,
    creds: Dict[str, Any],
) -> str:
    bucket_name = header_host.split(".")[0]
    resource = f"/{bucket_name}/{object_key}"
    oss_headers = {"x-oss-security-token": str(creds.get("security_token", ""))}
    authorization = build_oss_authorization(
        "PUT", content_type, oss_date, oss_headers, resource,
        str(creds.get("access_key_id", "")),
        str(creds.get("access_key_secret", "")),
    )
    headers = build_upload_headers(
        header_host, oss_date, content_type, len(file_data),
        authorization, str(creds.get("security_token", "")),
    )
    if connect_host != header_host:
        logger.debug("Using IP %s with Host: %s", connect_host, header_host)
    async with aiohttp.ClientSession() as s:
        async with s.put(
            f"https://{connect_host}/{object_key}", data=file_data,
            headers=headers, ssl=False,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
    return str(creds.get("file_url", ""))


async def upload_to_oss(
    file_data: bytes, content_type: str, creds: Dict[str, Any],
) -> str:
    file_url = str(creds.get("file_url", ""))
    object_key = str(creds.get("file_path", ""))
    parsed = urlparse(file_url)
    bucket_host = parsed.netloc

    time_offset = await sync_time_with_oss(bucket_host)
    adjusted_time = time.time() + time_offset
    oss_date = datetime.fromtimestamp(adjusted_time, tz=timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    logger.debug("Using OSS-synchronized date: %s (offset=%.2f)", oss_date, time_offset)

    host_candidates = resolve_oss_hosts(bucket_host)
    last_error: Optional[Exception] = None
    for connect_host, header_host in host_candidates:
        try:
            await try_oss_upload(
                connect_host, header_host, object_key,
                file_data, content_type, oss_date, creds,
            )
            return file_url
        except Exception as exc:
            last_error = exc
            logger.warning("OSS %s (via %s) failed: %s", header_host, connect_host, exc)
            continue
    raise RuntimeError(f"all OSS hosts failed: {last_error}")


# ==== 流式请求的消息准备与文件收集 ====


def _strip_media_for_inject(messages: List[Any]) -> List[Any]:
    """深拷贝 messages 并移除非文本 content parts，供 inject 使用。

    echotools inject 会将 content 序列化为纯文本 prompt；若 content 包含
    image_url / video_url / input_audio 等多模态 part，base64 数据会被当作
    JSON 字符串嵌入 prompt，导致 prompt 膨胀且泄露敏感数据。
    此函数保留纯文本 part，其余 part 丢弃；若剥离后仅剩单个 text part，
    则简化为 str 以匹配 inject 的预期格式。
    """
    import copy

    stripped: List[Any] = []
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(msg)
            continue
        text_parts = [
            p for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        if len(text_parts) == 1:
            new_msg = copy.copy(msg)
            new_msg["content"] = str(text_parts[0].get("text", ""))
            stripped.append(new_msg)
        elif text_parts:
            new_msg = copy.copy(msg)
            new_msg["content"] = "\n".join(
                str(p.get("text", "")) for p in text_parts
            )
            stripped.append(new_msg)
        else:
            new_msg = copy.copy(msg)
            new_msg["content"] = ""
            stripped.append(new_msg)
    return stripped


_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _extract_page_urls(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    urls: List[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(".,;:)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


async def _upload_base64_images(
    client: "QwenClient", session: "QwenSession", image_uris: List[str],
) -> List[Any]:
    files: List[Any] = []
    for uri in image_uris:
        try:
            _, image_obj = await client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    return files


async def _upload_remote_media(
    client: "QwenClient", session: "QwenSession", media_urls: List[str],
) -> List[Any]:
    files: List[Any] = []
    for media_url in media_urls:
        try:
            _, media_obj = await client.upload_file_from_url(session, media_url)
            files.append(media_obj)
        except Exception as e:
            logger.debug("Remote media upload failed: %s", e)
    return files


async def _upload_text_attachment(
    client: "QwenClient",
    session: "QwenSession",
    filename: Optional[str],
    file_bytes: Optional[bytes],
) -> List[Any]:
    if not filename or not file_bytes:
        return []
    try:
        _, file_obj = await client.upload_file(session, file_bytes, filename)
        return [file_obj]
    except Exception as e:
        logger.warning("Upload failed: %s, sending truncated text without attachment", e)
        return []


async def _collect_uploaded_files(
    client: "QwenClient",
    session: "QwenSession",
    messages: List[Any],
    image_uris: List[str],
    media_urls: List[str],
    filename: Optional[str],
    file_bytes: Optional[bytes],
    send_text: str,
) -> List[Any]:
    from upstream.qwen.chat.upload.upstream_api import parse_urls

    files: List[Any] = []
    files.extend(await _upload_base64_images(client, session, image_uris))
    files.extend(await _upload_remote_media(client, session, media_urls))
    files.extend(await _upload_text_attachment(client, session, filename, file_bytes))
    page_urls = _extract_page_urls(send_text)
    if page_urls:
        try:
            files.extend(await parse_urls(client, session, page_urls))
        except Exception as exc:
            logger.debug("parse_urls failed: %s", exc)
    return files


async def prepare_stream(
    state: Any,
    client: "QwenClient",
    session: "QwenSession",
    messages: List[Any],
    model: str,
    tools: Optional[List[Any]],
    req_id: str,
    protocol_options: Optional[Any] = None,
    *,
    prompt_api: str = "openai",
) -> Tuple[List[Any], List[Any], ThinkingRoute]:
    stripped = _strip_media_for_inject(messages)
    injected, full_content, route = prepare_injected_messages(
        state, stripped, tools, req_id, model, protocol_options, prompt_api,
    )
    image_uris = client.extract_base64_images(messages)
    media_urls = client.extract_remote_media_urls(messages)
    final_messages, send_text, filename, file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=True, model=model,
    )
    files = await _collect_uploaded_files(
        client, session, messages, image_uris, media_urls, filename, file_bytes, send_text,
    )
    final_messages[0]["content"] = send_text
    return final_messages, files, route
