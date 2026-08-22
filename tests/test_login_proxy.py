"""验证 Qwen 登录请求遵循 proxy toggle 状态（login 超时的回归）。"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_login_signin_uses_proxy_when_toggle_enabled(monkeypatch):
    from upstream.qwen import account as acc_mod
    from upstream.qwen.media.proxy_toggle import get_proxy_toggle

    toggle = get_proxy_toggle()
    monkeypatch.setattr(toggle, "_enabled", True)
    monkeypatch.setattr(toggle, "_initialized", True)

    captured: dict = {}

    class _FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"success": True, "data": {"access_token": "tok"}}

    class _FakeHttp:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    class _FakeAccount:
        username = "user@example.com"
        password = "secret"

    async def _no_user_id(*a, **k):
        return ""

    async def _no_warmup(*a, **k):
        return None

    def _no_task(coro, *a, **k):  # 跳过后台 sync task
        coro.close()
        return None

    monkeypatch.setattr(acc_mod, "fetch_user_id", _no_user_id)
    monkeypatch.setattr(acc_mod.asyncio, "create_task", _no_task)
    import upstream.qwen.chat.upload.upstream_api as api_mod

    monkeypatch.setattr(api_mod, "warmup_session", _no_warmup)
    monkeypatch.setattr(api_mod, "check_and_sync_user_settings", _no_warmup)

    session = await acc_mod._qwen_signin_once(None, _FakeHttp(), _FakeAccount())

    assert session is not None
    from server.retry.http_client import active_proxy_url

    assert captured.get("proxy") == active_proxy_url()


@pytest.mark.asyncio
async def test_login_signin_direct_when_toggle_disabled(monkeypatch):
    from upstream.qwen import account as acc_mod
    from upstream.qwen.media.proxy_toggle import get_proxy_toggle

    toggle = get_proxy_toggle()
    monkeypatch.setattr(toggle, "_enabled", False)
    monkeypatch.setattr(toggle, "_initialized", True)

    captured: dict = {}

    class _FakeResp:
        status = 401

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {}

    class _FakeHttp:
        def post(self, url, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    class _FakeAccount:
        username = "user@example.com"
        password = "secret"

    session = await acc_mod._qwen_signin_once(None, _FakeHttp(), _FakeAccount())

    assert session is None
    assert captured.get("proxy") is None
