from __future__ import annotations

"""QwenClient 的文件上传能力：STS 凭证 → OSS PUT → 后端文档解析 + 时间线埋点。"""

import base64
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import aiohttp

from server.formats import TokenExpiredError, UpstreamStsError
from upstream.qwen.auth.crypto import build_headers_async
from upstream.qwen.auth.http import get_qwen_proxy
from upstream.qwen.auth.report import (
    report_file_parse_success,
    report_file_upload_finish,
    report_file_upload_oss_token_time,
    report_file_upload_start,
)
from upstream.qwen.chat.endpoints import BASE_URL
from upstream.qwen.chat.store import QwenSession
from upstream.qwen.chat.upload.files import MediaDownloadMixin
from upstream.qwen.chat.upload.oss import upload_to_oss
from upstream.qwen.chat.upload.parse import wait_file_parsed
from upstream.qwen.chat.upload.storage import (
    apply_parse_status,
    build_file_object,
    get_file_category,
    get_mime_type,
)

logger = logging.getLogger("rogator")

_MAX_FILE_SIZES: Dict[str, int] = {
    "video": 500 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "image": 20 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}


class UploadMixin(MediaDownloadMixin):
    """文件上传管线：STS → OSS PUT → 文档解析 → 时间线埋点。

    通过 ``MediaDownloadMixin`` 组合下载与提取能力，避免 ``files.py``
    和本文件互相引用导致循环 import。
    """

    async def _maybe_parse_document(
        self,
        session: QwenSession,
        file_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        # text/pdf 等 type=file 需 parse；vision/audio 跳过
        if str(file_obj.get("type") or "") != "file":
            return file_obj
        # 纯文本文件无需后端文档解析，LLM 可直接读取内容
        content_type = str(file_obj.get("file_type") or "")
        if content_type == "text/plain":
            return apply_parse_status(file_obj, "success")
        file_id = str(file_obj.get("id") or "")
        if not file_id:
            return file_obj
        ok = await wait_file_parsed(self, session, file_id)
        if not ok:
            logger.warning(
                "Document parse failed or timed out: %s",
                file_obj.get("name", file_id[:8]),
            )
            return apply_parse_status(file_obj, "failed")
        file_obj = apply_parse_status(file_obj, "success")
        await report_file_parse_success(
            self,
            session,
            file_id=file_id,
            filename=str(file_obj.get("name") or ""),
            filesize=int(file_obj.get("size") or 0),
            content_type=str(
                file_obj.get("file_type") or file_obj.get("content_type") or ""
            ),
        )
        return file_obj

    async def _request_sts_token(
        self, path: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BASE_URL}{path}",
                json=payload,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=15),
                proxy=get_qwen_proxy(),
            ) as resp:
                # 401/403 表示会话凭证失效（限流/封禁），透传让调用方终止重试，
                # 而不是当作普通 STS 失败继续尝试备用端点
                if resp.status in (401, 403):
                    raise TokenExpiredError(
                        f"Token expired: HTTP {resp.status} (getstsToken)"
                    )
                if resp.status != 200:
                    return None
                data = await resp.json()
                creds = data.get("data", data)
                if all(
                    k in creds
                    for k in ("access_key_id", "access_key_secret", "security_token")
                ):
                    return creds
                return None

    async def _get_sts_credentials(
        self, session: QwenSession, filename: str, filesize: int, filetype: str
    ) -> Dict[str, Any]:
        headers = await build_headers_async(session.token)
        headers.update(
            {
                "Content-Type": "application/json;charset=UTF-8",
                "Accept": "application/json",
            }
        )
        payload = {"filename": filename, "filesize": str(filesize), "filetype": filetype}
        for path in ["/api/v1/files/getstsToken", "/api/v2/files/getstsToken"]:
            try:
                creds = await self._request_sts_token(path, payload, headers)
                if creds:
                    return creds
            except TokenExpiredError:
                # 凭证失效（如限流）时立即终止，不再尝试备用端点
                raise
            except Exception:
                continue
        raise UpstreamStsError("All STS endpoints failed")

    async def _report_upload_timeline(
        self,
        session: QwenSession,
        *,
        filename: str,
        file_size: int,
        content_type: str,
        t_sts0: float,
        t_sts1: float,
        t_up0: float,
        t_up1: float,
    ) -> None:
        await report_file_upload_oss_token_time(
            self, session, filename=filename, filesize=file_size,
            content_type=content_type, start_ms=t_sts0, end_ms=t_sts1,
        )
        await report_file_upload_start(
            self, session, filename=filename, filesize=file_size,
            content_type=content_type, start_ms=t_up0,
        )
        await report_file_upload_finish(
            self, session, filename=filename, filesize=file_size,
            content_type=content_type, upload_start_ms=t_up0,
            upload_end_ms=t_up1, all_elapsed_ms=int(t_up1 - t_sts0),
        )

    async def upload_file(
        self, session: QwenSession, file_data: bytes, filename: str,
        content_type: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        file_url, file_obj = await self._do_upload_file(
            session, file_data, filename, content_type,
        )
        return file_url, await self._maybe_parse_document(session, file_obj)

    async def _do_upload_file(
        self, session: QwenSession, file_data: bytes, filename: str,
        content_type: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        if not content_type:
            content_type = get_mime_type(filename)
        file_type, _ = get_file_category(content_type)
        file_size = len(file_data)
        limit = _MAX_FILE_SIZES.get(file_type, 20 * 1024 * 1024)
        if file_size > limit:
            raise RuntimeError(f"file too large: {filename} ({file_size} > {limit})")
        page_t0 = time.perf_counter()

        def _perf_ms() -> float:
            return 1_000_000.0 + (time.perf_counter() - page_t0) * 1000.0

        t_sts0 = _perf_ms()
        creds = await self._get_sts_credentials(session, filename, file_size, file_type)
        t_sts1 = _perf_ms()
        t_up0 = _perf_ms()
        file_url = await upload_to_oss(file_data, content_type, creds)
        t_up1 = _perf_ms()
        await self._report_upload_timeline(
            session,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            t_sts0=t_sts0,
            t_sts1=t_sts1,
            t_up0=t_up0,
            t_up1=t_up1,
        )
        file_obj = build_file_object(
            file_id=str(creds.get("file_id", uuid.uuid4())),
            file_url=file_url,
            filename=filename,
            size=file_size,
            content_type=content_type,
            user_id=session.user_id,
        )
        return file_url, file_obj

    async def upload_file_from_base64(
        self, session: QwenSession, data_uri: str
    ) -> Tuple[str, Dict[str, Any]]:

        if not data_uri.startswith("data:") or ";base64," not in data_uri:
            raise RuntimeError("invalid base64 data URI")
        header, encoded = data_uri.split(";base64,", 1)
        mime_type = header.split("data:", 1)[1]
        padding = (-len(encoded)) % 4
        if padding:
            encoded += "=" * padding
        ext = MediaDownloadMixin.ext_from_content_type(mime_type)
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        return await self.upload_file(
            session, base64.b64decode(encoded), filename, content_type=mime_type,
        )

    async def upload_file_from_url(
        self, session: QwenSession, media_url: str
    ) -> Tuple[str, Dict[str, Any]]:

        async with aiohttp.ClientSession() as s:
            async with s.get(
                media_url,
                headers={
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "User-Agent": "qwen-mock",  # 真实 UA 来自 endpoints.USER_AGENT，避免循环 import
                },
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30),
                proxy=get_qwen_proxy(),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"download file failed: HTTP {resp.status}")
                data = await resp.read()
                content_type = resp.headers.get(
                    "Content-Type", "application/octet-stream"
                ).split(";", 1)[0]
        ext = MediaDownloadMixin.ext_from_content_type(content_type)
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        return await self.upload_file(session, data, filename, content_type=content_type)

    async def upload_file_from_path(
        self,
        session: QwenSession,
        file_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        import os
        if not os.path.exists(file_path):
            raise RuntimeError(f"file not found: {file_path}")
        from pathlib import Path
        data = Path(file_path).read_bytes()
        return await self.upload_file(session, data, os.path.basename(file_path))