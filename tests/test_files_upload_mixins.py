"""``UploadMixin._do_upload_file`` 的参数组装最小覆盖。

覆盖三条契约：
1. 超大文件直接抛 ``RuntimeError``，不发起 STS/OSS 调用
2. ``_get_sts_credentials`` 抛 ``UpstreamStsError`` 时原样上抛（无降级）
3. 正常路径：调用 ``upload_to_oss``，落 ``build_file_object``，把 STS ``file_id``
   透传到 file_obj；``_report_upload_timeline`` 至少被调用一次。
"""
from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.formats import UpstreamStsError
from upstream.qwen.chat.store import QwenSession
from upstream.qwen.chat.upload.files_upload import UploadMixin

from core.session.accounts import Account


def _make_session() -> QwenSession:
    return QwenSession(
        account=Account(username="u@test.com", password="pw"),
        token="tok",
        user_id="u",
        login_time=0.0,
    )


def _make_client() -> UploadMixin:
    return UploadMixin()


class _Awaitable:
    def __init__(self, value: Any) -> None:
        self._value = value

    def __await__(self):  # pragma: no cover - delegated
        async def _coro() -> Any:
            return self._value

        return _coro().__await__()


def _async_return(value: Any) -> Any:
    async def _coro() -> Any:
        return value

    return _coro()


@pytest.mark.asyncio
async def test_oversize_file_raises_without_network(monkeypatch) -> None:
    """>20MB image 直接 RuntimeError，不触发 STS/OSS。"""
    client = _make_client()
    sts_mock = AsyncMock(side_effect=AssertionError("STS must not be called"))
    oss_mock = AsyncMock(side_effect=AssertionError("OSS must not be called"))
    monkeypatch.setattr(client, "_get_sts_credentials", sts_mock)
    monkeypatch.setattr(client, "_report_upload_timeline", AsyncMock())
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.upload_to_oss", oss_mock,
    )
    big = b"x" * (21 * 1024 * 1024)
    with pytest.raises(RuntimeError, match="file too large"):
        await client._do_upload_file(_make_session(), big, "big.png", "image/png")
    sts_mock.assert_not_called()
    oss_mock.assert_not_called()


@pytest.mark.asyncio
async def test_sts_failure_propagates(monkeypatch) -> None:
    """STS 失败抛 UpstreamStsError，不被静默吞掉。"""
    client = _make_client()
    monkeypatch.setattr(
        client,
        "_get_sts_credentials",
        AsyncMock(side_effect=UpstreamStsError("All STS endpoints failed")),
    )
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.upload_to_oss",
        AsyncMock(return_value="https://oss.example/never"),
    )
    monkeypatch.setattr(client, "_report_upload_timeline", AsyncMock())
    with pytest.raises(UpstreamStsError):
        await client._do_upload_file(
            _make_session(), b"abc", "tiny.png", "image/png",
        )


@pytest.mark.asyncio
async def test_happy_path_assembles_file_object(monkeypatch) -> None:
    """正常路径：file_id 从 STS 透传到 build_file_object；timeline 埋点被调用。"""
    client = _make_client()
    creds = {
        "access_key_id": "ak",
        "access_key_secret": "sk",
        "security_token": "st",
        "file_id": "fid-from-sts",
    }
    monkeypatch.setattr(
        client, "_get_sts_credentials", AsyncMock(return_value=creds),
    )
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.upload_to_oss",
        AsyncMock(return_value="https://oss.example/returned"),
    )
    timeline_mock = AsyncMock()
    monkeypatch.setattr(client, "_report_upload_timeline", timeline_mock)

    captured: Dict[str, Any] = {}

    def _capture_build(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": kwargs.get("file_id"),
            "url": kwargs.get("file_url"),
            "name": kwargs.get("filename"),
            "size": kwargs.get("size"),
            "file_type": kwargs.get("content_type"),
            "user_id": kwargs.get("user_id"),
        }

    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.build_file_object",
        _capture_build,
    )

    url, obj = await client._do_upload_file(
        _make_session(), b"hello-bytes", "hello.png", "image/png",
    )
    assert url == "https://oss.example/returned"
    assert obj["id"] == "fid-from-sts"
    assert captured["file_id"] == "fid-from-sts"
    assert captured["file_url"] == "https://oss.example/returned"
    assert captured["filename"] == "hello.png"
    assert captured["size"] == len(b"hello-bytes")
    assert captured["content_type"] == "image/png"
    assert timeline_mock.await_count == 1


@pytest.mark.asyncio
async def test_content_type_inferred_from_filename(monkeypatch) -> None:
    """``content_type=None`` 时按 filename 推断 mime，并落到 file_obj。"""
    client = _make_client()
    monkeypatch.setattr(
        client,
        "_get_sts_credentials",
        AsyncMock(return_value={
            "access_key_id": "ak", "access_key_secret": "sk", "security_token": "st",
            "file_id": "fid",
        }),
    )
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.upload_to_oss",
        AsyncMock(return_value="https://oss.example/x"),
    )
    monkeypatch.setattr(client, "_report_upload_timeline", AsyncMock())

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.build_file_object",
        lambda **kw: (captured.setdefault("kwargs", kw) or {"id": kw["file_id"]}),
    )
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.get_mime_type",
        lambda name: ("image/png" if name.endswith(".png") else "application/octet-stream"),
    )
    monkeypatch.setattr(
        "upstream.qwen.chat.upload.files_upload.get_file_category",
        lambda ct: ("image", ""),
    )
    _url, _obj = await client._do_upload_file(
        _make_session(), b"abc", "x.png", None,
    )
    assert captured["kwargs"]["content_type"] == "image/png"