"""Qwen 附件上传失败行为测试：上传失败不再静默降级，而是一路抛到请求者。

覆盖两条已修改的路径：
1. ``files._request_sts_token`` 对 401/403 抛 ``TokenExpiredError``，且
   ``_get_sts_credentials`` 不再尝试备用端点（限流凭证失效立即终止）。
2. ``oss._upload_text_attachment`` 上传失败直接上抛，不再返回 [] 静默降级
   为"截断文本无附件"；``classify_stream_error`` / ``handler_error_response``
   最终把 ``TokenExpiredError`` 映射为 HTTP 429 交给请求者。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from server.formats import TokenExpiredError, UpstreamStsError
from upstream.qwen.chat.upload import files_upload as files_mod
from upstream.qwen.chat.upload.oss import _upload_text_attachment
from upstream.qwen.chat.store import QwenSession

from core.session.accounts import Account


class _FakeResp:
    """最小 aiohttp 响应替身：只有 status 与 json()。"""

    def __init__(self, status: int, payload: Dict[str, Any]) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> Dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeSession:
    """Fake aiohttp.ClientSession：按构造表依次返回响应，并记录调用。"""

    def __init__(self, responses: List[_FakeResp]) -> None:
        self._responses = list(responses)
        self.calls: int = 0

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def post(self, *args: Any, **kwargs: Any) -> _FakeResp:
        self.calls += 1
        return self._responses.pop(0)


def _make_client() -> files_mod.UploadMixin:
    return files_mod.UploadMixin()


def _make_session() -> QwenSession:
    return QwenSession(
        account=Account(username="u@test.com", password="pw"),
        token="tok",
        user_id="u",
        login_time=0.0,
    )


def _fake_ok_payload() -> Dict[str, Any]:
    return {
        "data": {
            "access_key_id": "ak",
            "access_key_secret": "sk",
            "security_token": "st",
        }
    }


@pytest.mark.asyncio
async def test_sts_401_raises_token_expired(monkeypatch) -> None:
    """凭证失效（限流 401）：抛 TokenExpiredError，而不是返回 None。"""
    fake = _FakeSession([_FakeResp(401, {})])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    with pytest.raises(TokenExpiredError, match="Token expired: HTTP 401"):
        await client._request_sts_token("/api/v1/files/getstsToken", {}, {})
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_sts_403_raises_token_expired(monkeypatch) -> None:
    fake = _FakeSession([_FakeResp(403, {})])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    with pytest.raises(TokenExpiredError, match="Token expired: HTTP 403"):
        await client._request_sts_token("/api/v2/files/getstsToken", {}, {})


@pytest.mark.asyncio
async def test_sts_200_returns_creds(monkeypatch) -> None:
    """正常 200 仍返回 STS 凭证。"""
    fake = _FakeSession([_FakeResp(200, _fake_ok_payload())])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    creds = await client._request_sts_token("/api/v1/files/getstsToken", {}, {})
    assert creds == _fake_ok_payload()["data"]
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_sts_200_missing_keys_returns_none(monkeypatch) -> None:
    """200 但缺少必需字段（代理中途篡改等）仍返回 None。"""
    fake = _FakeSession([_FakeResp(200, {"data": {"nothing": "here"}})])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    assert await client._request_sts_token("/api/v1/files/getstsToken", {}, {}) is None


@pytest.mark.asyncio
async def test_get_sts_credentials_bad_status_tries_fallback(monkeypatch) -> None:
    """非 401/403 失败（如 500）会依次尝试备用端点，最后汇总 UpstreamStsError。"""
    fake = _FakeSession([_FakeResp(500, {}), _FakeResp(500, {})])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    with pytest.raises(UpstreamStsError, match="All STS endpoints failed"):
        await client._get_sts_credentials(_make_session(), "a.txt", 3, "file")
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_get_sts_credentials_401_does_not_try_fallback(monkeypatch) -> None:
    """401 凭证失效立即终止：备用端点不应被调用（此前会吞掉继续试）。"""
    fake = _FakeSession([_FakeResp(401, {})])
    monkeypatch.setattr(files_mod.aiohttp, "ClientSession", lambda *a, **k: fake)
    client = _make_client()
    with pytest.raises(TokenExpiredError, match="Token expired: HTTP 401"):
        await client._get_sts_credentials(_make_session(), "a.txt", 3, "file")
    assert fake.calls == 1


# ---- oss._upload_text_attachment：上传失败不得静默降级为"截断文本无附件" ----


class _UploadClient:
    """最小 client 替身：用注入的 async 函数模拟 upload_file 的成败。"""

    def __init__(self, impl) -> None:
        self._impl = impl

    async def upload_file(self, session, file_bytes: bytes, filename: str):
        return await self._impl(session, file_bytes, filename)


@pytest.mark.asyncio
async def test_upload_text_attachment_no_attachment_noop() -> None:
    """无附件（长 prompt 未超限）时返回空列表，不触发上传。"""
    client = _UploadClient(lambda *a: (_ for _ in ()).throw(AssertionError("must not upload")))
    result = await _upload_text_attachment(client, _make_session(), None, None)
    assert result == []


@pytest.mark.asyncio
async def test_upload_text_attachment_ok_returns_file_obj() -> None:
    """正常上传返回 file_obj 列表。"""
    file_obj = {"id": "f1", "name": "remaining.txt"}

    async def _ok(session, file_bytes, filename):
        assert session is not None
        assert file_bytes == b"ABCDE"
        assert filename == "remaining.txt"
        return ("https://oss/x/remaining.txt", file_obj)

    client = _UploadClient(_ok)
    result = await _upload_text_attachment(client, _make_session(), "remaining.txt", b"ABCDE")
    assert result == [file_obj]


@pytest.mark.asyncio
async def test_upload_text_attachment_failure_propagates(monkeypatch) -> None:
    """上传失败（限流凭证失效）必须上抛，绝不静默降级为不带附件的截断文本。

    回归场景：日志中反复出现的 "Upload failed: All STS endpoints failed,
    sending truncated text without attachment"。
    """
    client = _UploadClient(
        lambda *a: (_ for _ in ()).throw(TokenExpiredError("Token expired: HTTP 401 (getstsToken)"))
    )
    with pytest.raises(TokenExpiredError):
        await _upload_text_attachment(client, _make_session(), "remaining.txt", b"ABCDE")


@pytest.mark.asyncio
async def test_upload_text_attachment_generic_failure_propagates() -> None:
    """任意上传异常同样上抛（网络中断等），不留任何吞错路径。"""
    client = _UploadClient(
        lambda *a: (_ for _ in ()).throw(RuntimeError("All STS endpoints failed"))
    )
    with pytest.raises(RuntimeError, match="All STS endpoints failed"):
        await _upload_text_attachment(client, _make_session(), "remaining.txt", b"ABCDE")


# ---- 错误最终映射：TokenExpiredError -> HTTP 429 rate_limited ----


def test_classify_stream_error_token_expired_maps_to_429() -> None:
    from handlers.shared.api_errors import classify_stream_error

    info = classify_stream_error(TokenExpiredError("Token expired: HTTP 401 (getstsToken)"))
    assert info.kind == "rate_limited"
    assert info.code == 429


def test_handler_error_response_token_expired_429_openai() -> None:
    from handlers.shared.api_errors import handler_error_response

    resp = handler_error_response(
        TokenExpiredError("Token expired: HTTP 401 (getstsToken)"), label="test"
    )
    assert resp.status == 429
    body = json.loads(resp.body.decode("utf-8"))
    assert body["error"]["type"] == "rate_limited"
    assert "Token expired" in body["error"]["message"]


def test_handler_error_response_token_expired_429_anthropic() -> None:
    from handlers.shared.api_errors import handler_error_response

    resp = handler_error_response(
        TokenExpiredError("Token expired: HTTP 401 (getstsToken)"),
        label="test",
        protocol="anthropic",
    )
    assert resp.status == 429
    body = json.loads(resp.body.decode("utf-8"))
    assert body["error"]["type"] == "rate_limit_error"