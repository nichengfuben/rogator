from __future__ import annotations

"""DeepSeek 文件上传与解析轮询（对齐 FE upload_file / fetch_files）。"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from upstream.deepseek.lib.guard.pow import get_pow_response
from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST, MODEL_VISION
from upstream.deepseek.lib.protocol.headers import build_headers

logger = logging.getLogger(__name__)

UPLOAD_PATH: str = "/api/v0/file/upload_file"
FETCH_PATH: str = "/api/v0/file/fetch_files"
POLL_INTERVAL_SEC: float = 3.0
MAX_POLL_ATTEMPTS: int = 60

_PARSE_SUCCESS = frozenset({"SUCCESS"})
_PARSE_ERRORS = frozenset(
    {
        "FAILED",
        "CONTENT_FILTER",
        "CONTENT_TOO_LONG",
        "CANCELLED",
        "CONTENT_EMPTY",
        "_CUSTOM_SYSTEM_ERROR_FAIL",
    }
)


def resolve_model_type(model: str) -> str:

    return "vision" if model == MODEL_VISION else "default"


def is_parse_success_status(status: str) -> bool:
    return status in _PARSE_SUCCESS


def is_parse_error_status(status: str) -> bool:
    return status in _PARSE_ERRORS


def _upload_url() -> str:
    return "https://{host}{path}".format(host=DEFAULT_HOST, path=UPLOAD_PATH)


def _fetch_url() -> str:
    return "https://{host}{path}".format(host=DEFAULT_HOST, path=FETCH_PATH)


def _guess_content_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"


def _biz_ok(data: Dict[str, Any]) -> bool:
    if data.get("code") != 0:
        return False
    inner = data.get("data") or {}
    return inner.get("biz_code", -1) == 0


def _extract_file_record(data: Dict[str, Any]) -> Dict[str, Any]:
    inner = data.get("data") or {}
    biz = inner.get("biz_data") or {}
    if isinstance(biz, dict) and "id" in biz:
        return biz
    if isinstance(biz, dict):
        nested = biz.get("file")
        if isinstance(nested, dict):
            return nested
    return biz if isinstance(biz, dict) else {}


def _build_upload_headers(
    token: str,
    file_bytes: bytes,
    filename: str,
    *,
    hif_leim: str,
    hif_dliq: str,
    pow_resp: str,
    model_type: str,
    thinking_enabled: bool,
) -> Tuple[Dict[str, str], aiohttp.FormData]:
    headers = build_headers(
        token=token,
        hif_leim=hif_leim,
        hif_dliq=hif_dliq,
        pow_response=pow_resp,
    )
    headers.pop("content-type", None)
    headers["x-thinking-enabled"] = "1" if thinking_enabled else "0"
    headers["x-model-type"] = model_type
    headers["x-file-size"] = str(len(file_bytes))

    form = aiohttp.FormData()
    form.add_field(
        "file",
        file_bytes,
        filename=filename,
        content_type=_guess_content_type(filename),
    )
    return headers, form


async def upload_file(
    session: aiohttp.ClientSession,
    token: str,
    file_bytes: bytes,
    filename: str,
    *,
    hif_leim: str = "",
    hif_dliq: str = "",
    pow_solver: Any = None,
    model_type: str = "default",
    thinking_enabled: bool = False,
) -> Dict[str, Any]:

    pow_resp = ""
    if pow_solver is not None and getattr(pow_solver, "available", False):
        pow_resp = await get_pow_response(session, token, pow_solver, UPLOAD_PATH)

    headers, form = _build_upload_headers(
        token,
        file_bytes,
        filename,
        hif_leim=hif_leim,
        hif_dliq=hif_dliq,
        pow_resp=pow_resp,
        model_type=model_type,
        thinking_enabled=thinking_enabled,
    )

    async with session.post(
        _upload_url(),
        data=form,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=600),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            raise RuntimeError("upload_file HTTP {}".format(resp.status))
        data = await resp.json()
        if not _biz_ok(data):
            inner = data.get("data") or {}
            raise RuntimeError(
                "upload_file biz_code={} msg={}".format(
                    inner.get("biz_code"),
                    inner.get("biz_msg"),
                )
            )
        record = _extract_file_record(data)
        if not record.get("id"):
            raise RuntimeError("upload_file missing file id")
        return record


async def fetch_files(
    session: aiohttp.ClientSession,
    token: str,
    file_ids: Sequence[str],
) -> List[Dict[str, Any]]:

    if not file_ids:
        return []
    async with session.get(
        _fetch_url(),
        headers=build_headers(token=token),
        params={"file_ids": ",".join(file_ids)},
        timeout=aiohttp.ClientTimeout(total=60),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            raise RuntimeError("fetch_files HTTP {}".format(resp.status))
        data = await resp.json()
        if not _biz_ok(data):
            inner = data.get("data") or {}
            raise RuntimeError(
                "fetch_files biz_code={} msg={}".format(
                    inner.get("biz_code"),
                    inner.get("biz_msg"),
                )
            )
        inner = data.get("data") or {}
        biz = inner.get("biz_data") or {}
        files = biz.get("files") or []
        return [f for f in files if isinstance(f, dict)]


async def wait_files_ready(
    session: aiohttp.ClientSession,
    token: str,
    file_ids: Sequence[str],
    *,
    poll_interval: float = POLL_INTERVAL_SEC,
    max_attempts: int = MAX_POLL_ATTEMPTS,
) -> List[Dict[str, Any]]:

    pending = set(file_ids)
    latest: Dict[str, Dict[str, Any]] = {}
    for _ in range(max_attempts):
        if not pending:
            break
        records = await fetch_files(session, token, sorted(pending))
        for record in records:
            fid = str(record.get("id") or "")
            if not fid:
                continue
            latest[fid] = record
            status = str(record.get("status") or "")
            if is_parse_success_status(status):
                pending.discard(fid)
            elif is_parse_error_status(status):
                raise RuntimeError(
                    "file {} parse failed: status={}".format(fid, status)
                )
        if pending:
            await asyncio.sleep(poll_interval)

    if pending:
        raise TimeoutError(
            "file parse timeout, pending={}".format(",".join(sorted(pending)))
        )
    return [latest[fid] for fid in file_ids if fid in latest]


async def upload_and_wait(
    session: aiohttp.ClientSession,
    token: str,
    file_bytes: bytes,
    filename: str,
    *,
    hif_managers: Dict[str, Any],
    username: str,
    pow_solver: Any,
    model_type: str = "default",
    thinking_enabled: bool = False,
) -> str:

    hif_leim = ""
    hif_dliq = ""
    mgr = hif_managers.get(username)
    if mgr is not None:
        hif_leim, hif_dliq = await mgr.ensure_valid()

    record = await upload_file(
        session,
        token,
        file_bytes,
        filename,
        hif_leim=hif_leim,
        hif_dliq=hif_dliq,
        pow_solver=pow_solver,
        model_type=model_type,
        thinking_enabled=thinking_enabled,
    )
    file_id = str(record.get("id") or "")
    status = str(record.get("status") or "")
    if is_parse_success_status(status):
        return file_id
    if is_parse_error_status(status):
        raise RuntimeError("upload rejected: status={}".format(status))
    await wait_files_ready(session, token, [file_id])
    return file_id


async def upload_attachments(
    session: aiohttp.ClientSession,
    token: str,
    username: str,
    attachments: Sequence[Tuple[bytes, str]],
    *,
    hif_managers: Dict[str, Any],
    pow_solver: Any,
    model_type: str = "default",
    thinking_enabled: bool = False,
) -> List[str]:
    """批量上传附件，返回 ref_file_ids。"""
    ref_ids: List[str] = []
    for data, name in attachments:
        fid = await upload_and_wait(
            session,
            token,
            data,
            name,
            hif_managers=hif_managers,
            username=username,
            pow_solver=pow_solver,
            model_type=model_type,
            thinking_enabled=thinking_enabled,
        )
        ref_ids.append(fid)
    return ref_ids
