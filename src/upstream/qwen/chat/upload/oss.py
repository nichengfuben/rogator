from __future__ import annotations

"""OSS upload helpers with time synchronization and DNS fallback."""

import base64
import hashlib
import hmac
import logging
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from server.formats import DEFAULT_USER_AGENT
from upstream.qwen.auth.http import borrow_http_session, run_with_connection_retry

logger = logging.getLogger("rogator")

STS_TOKEN_PATHS = ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]


async def _request_sts_credentials(
    url: str, payload: Dict[str, Any], headers: Dict[str, str],
    http: aiohttp.ClientSession | None = None,
) -> Optional[Dict[str, Any]]:
    async with borrow_http_session(http) as s:
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
    http: aiohttp.ClientSession | None = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        "Origin": base_url,
        "Referer": f"{base_url}/",
        "Source": "web",
    }
    payload = {"filename": filename, "filesize": filesize, "filetype": "file"}
    for path in STS_TOKEN_PATHS:
        try:
            creds = await _request_sts_credentials(f"{base_url}{path}", payload, headers, http)
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
        "User-Agent": DEFAULT_USER_AGENT,
    }


async def get_oss_time(host: str, http: aiohttp.ClientSession | None = None) -> Optional[str]:
    try:
        async with borrow_http_session(http) as s:
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


async def sync_time_with_oss(host: str, http: aiohttp.ClientSession | None = None) -> float:
    oss_date = await get_oss_time(host, http)
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
    http: aiohttp.ClientSession | None = None,
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
    async with borrow_http_session(http) as s:
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
    http: aiohttp.ClientSession | None = None,
) -> str:
    file_url = str(creds.get("file_url", ""))
    object_key = str(creds.get("file_path", ""))
    parsed = urlparse(file_url)
    bucket_host = parsed.netloc

    time_offset = await sync_time_with_oss(bucket_host, http)
    adjusted_time = time.time() + time_offset
    oss_date = datetime.fromtimestamp(adjusted_time, tz=timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    logger.debug("Using OSS-synchronized date: %s (offset=%.2f)", oss_date, time_offset)

    host_candidates = resolve_oss_hosts(bucket_host)
    last_error: Optional[Exception] = None
    for connect_host, header_host in host_candidates:
        try:
            await run_with_connection_retry(
                f"oss_upload:{header_host}",
                lambda: try_oss_upload(
                    connect_host, header_host, object_key,
                    file_data, content_type, oss_date, creds, http,
                ),
            )
            return file_url
        except Exception as exc:
            last_error = exc
            logger.warning("OSS %s (via %s) failed: %s", header_host, connect_host, exc)
            continue
    raise RuntimeError(f"all OSS hosts failed: {last_error}")
