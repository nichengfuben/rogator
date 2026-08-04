from __future__ import annotations

"""文件上传相关 tongyi-sg / aes 上报与启动 beacon。"""

import json
from typing import TYPE_CHECKING, Dict
from urllib.parse import urlencode

from upstream.qwen.auth.report.core import (
    PAGE_HOME,
    SPM_HOME,
    ms_now,
    report_aes_events,
    report_aplus_event,
    silent_request,
    uid,
)
from upstream.qwen.chat.routes import APP_VERSION, CHAT_ORIGIN

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession


def _upload_file_meta(filename: str, filesize: int, content_type: str) -> str:
    return json.dumps(
        {"filename": filename, "filesize": filesize, "filetype": content_type},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _upload_mb_s(filesize: int, elapsed_ms: int) -> float:
    if elapsed_ms <= 0 or filesize <= 0:
        return 0.0
    return round((filesize / (1024 * 1024)) / (elapsed_ms / 1000.0), 3)


def _aes_upload_event(
    session: "QwenSession",
    *,
    p1: str,
    extra: Dict[str, str],
) -> Dict[str, str]:
    ev = {
        "c1": uid(session),
        "c10": APP_VERSION,
        "p1": p1,
        "p4": "OTHER",
        "ts": str(ms_now()),
        "type": "event",
    }
    ev.update(extra)
    return ev


async def report_file_upload_all_time(
    client: "QwenClient",
    session: "QwenSession",
    *,
    filename: str,
    filesize: int,
    content_type: str,
    elapsed_ms: int,
    page_url: str = PAGE_HOME,
) -> None:
    """上传整段耗时：FileUpload-AllTime（tongyi-sg）。"""
    _ = filename, content_type
    mb_s = _upload_mb_s(filesize, elapsed_ms)
    elapsed = str(max(0, int(elapsed_ms)))
    await report_aplus_event(
        client,
        session,
        "FileUpload-AllTime",
        gmkey="self_define",
        extra_gokey={
            "c1": uid(session),
            "c4": "file",
            "c5": str(filesize),
            "c6": elapsed,
            "c8": elapsed,
            "c9": str(mb_s),
        },
        page_url=page_url,
        spm_cnt_base=SPM_HOME,
    )


async def report_file_parse_success(
    client: "QwenClient",
    session: "QwenSession",
    *,
    file_id: str,
    filename: str,
    filesize: int,
    content_type: str,
    page_url: str = PAGE_HOME,
) -> None:
    """文档解析成功：filePaseSuccess（tongyi-sg + aes）。"""
    c7 = json.dumps(
        {
            "name": filename,
            "size": filesize,
            "content_type": content_type,
            "parse_meta": {"parse_status": "success"},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    extra = {
        "c1": uid(session),
        "c4": file_id,
        "c5": filename,
        "c6": content_type,
        "c7": c7,
    }
    await report_aplus_event(
        client,
        session,
        "filePaseSuccess",
        gmkey="self_define",
        extra_gokey=extra,
        page_url=page_url,
        spm_cnt_base=SPM_HOME,
    )
    await report_aes_events(
        client,
        session,
        [{**extra, "c10": APP_VERSION, "p1": "filePaseSuccess", "p4": "OTHER",
          "ts": str(ms_now()), "type": "event"}],
        page_url=page_url,
    )


async def report_file_upload_oss_token_time(
    client: "QwenClient",
    session: "QwenSession",
    *,
    filename: str,
    filesize: int,
    content_type: str,
    start_ms: float,
    end_ms: float,
    page_url: str = PAGE_HOME,
) -> None:
    """STS 耗时：FileUpload-ossTokenTime（aes）。"""
    meta = _upload_file_meta(filename, filesize, content_type)
    await report_aes_events(
        client,
        session,
        [
            _aes_upload_event(
                session,
                p1="FileUpload-ossTokenTime",
                extra={
                    "c4": str(start_ms),
                    "c5": str(end_ms),
                    "c6": str(max(0.0, end_ms - start_ms)),
                    "c8": meta,
                },
            )
        ],
        page_url=page_url,
    )


async def report_file_upload_start(
    client: "QwenClient",
    session: "QwenSession",
    *,
    filename: str,
    filesize: int,
    content_type: str,
    start_ms: float,
    page_url: str = PAGE_HOME,
) -> None:
    """开始 OSS 上传：FileUpload-startUpload（aes）。"""
    meta = _upload_file_meta(filename, filesize, content_type)
    await report_aes_events(
        client,
        session,
        [
            _aes_upload_event(
                session,
                p1="FileUpload-startUpload",
                extra={"c4": str(start_ms), "c8": meta},
            )
        ],
        page_url=page_url,
    )


def _finish_upload_aes_events(
    session: "QwenSession",
    *,
    meta: str,
    upload_start_ms: float,
    upload_end_ms: float,
    filesize: int,
    all_elapsed_ms: int,
) -> list[Dict[str, str]]:
    elapsed_upload = max(0.0, upload_end_ms - upload_start_ms)
    mb_s = _upload_mb_s(filesize, all_elapsed_ms)
    elapsed_all = str(max(0, int(all_elapsed_ms)))
    return [
        _aes_upload_event(
            session,
            p1="FileUpload-finishUpload",
            extra={
                "c4": str(upload_start_ms),
                "c5": str(upload_end_ms),
                "c6": str(elapsed_upload),
                "c7": "start",
                "c8": meta,
            },
        ),
        _aes_upload_event(
            session,
            p1="FileUpload-AllTime",
            extra={
                "c4": "file",
                "c5": str(filesize),
                "c6": elapsed_all,
                "c7": "0",
                "c8": elapsed_all,
                "c9": str(mb_s),
            },
        ),
    ]


async def report_file_upload_finish(
    client: "QwenClient",
    session: "QwenSession",
    *,
    filename: str,
    filesize: int,
    content_type: str,
    upload_start_ms: float,
    upload_end_ms: float,
    all_elapsed_ms: int,
    page_url: str = PAGE_HOME,
) -> None:
    """OSS 完成 + 整段耗时：aes(finish+AllTime) 与 tongyi-sg(AllTime)。"""
    meta = _upload_file_meta(filename, filesize, content_type)
    await report_aes_events(
        client,
        session,
        _finish_upload_aes_events(
            session,
            meta=meta,
            upload_start_ms=upload_start_ms,
            upload_end_ms=upload_end_ms,
            filesize=filesize,
            all_elapsed_ms=all_elapsed_ms,
        ),
        page_url=page_url,
    )
    await report_file_upload_all_time(
        client,
        session,
        filename=filename,
        filesize=filesize,
        content_type=content_type,
        elapsed_ms=all_elapsed_ms,
        page_url=page_url,
    )


async def report_compare_log_arrival(client: "QwenClient") -> None:
    """启动期 compareLogService beacon。"""
    import secrets

    log_id = secrets.token_hex(20)
    ts = ms_now()
    for host, service in (
        ("https://aplus.qwen.ai", "aplus.qwen.ai"),
        ("https://ss.qwen.ai", "ss.qwen.ai"),
    ):
        path = (
            "/service.stability.log_arrival_rate"
            if "aplus" in host
            else "/ss.compare.service"
        )
        body = {
            "gmkey": "OTHER",
            "gokey": urlencode(
                {
                    "logId": log_id,
                    "timestamp": str(ts),
                    "domain": "chat.qwen.ai",
                    "testTag": "compareLogService",
                    "testVersion": "5.0.0",
                    "serviceName": service,
                    "requestType": "sendBeacon",
                }
            ),
        }
        await silent_request(
            client,
            "POST",
            f"{host}{path}?logId={log_id}",
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "Origin": CHAT_ORIGIN,
                "Referer": f"{CHAT_ORIGIN}/",
            },
            json_body=body,
        )
