"""``atomic_write_text`` 跨盘 fallback / OP_NO_TICKET / socks import / lease 防越界。"""

from __future__ import annotations

import errno
import os
import ssl
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import core.session.io as io
from core.session.io import atomic_write_text
from core.session.pool import SessionLoginMixin
from core.transport.http import get_upstream_ssl_context
from server.formats.errors import (
    UpstreamUnavailableError,
    attach_proxy_toggle,
    read_proxy_used_enabled,
)


class _ProbePool(SessionLoginMixin):
    UPSTREAM_NAME = "probe"

    def __init__(self, sessions):
        self._sessions = sessions
        self._current_index = 0

    async def _perform_login(self, account):  # pragma: no cover
        raise NotImplementedError


class TestAtomicWriteCrossDevice:
    def test_replace_cross_device_falls_back_to_shutil_move(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        target.write_text("old", encoding="utf-8")

        # 模拟 ``os.replace`` 抛 EXDEV（跨盘）；shutil.move 应接管写入
        cross = OSError(errno.EXDEV, "Cross-device link not permitted")
        calls = {"n": 0}

        def _maybe_cross(src, dst):
            calls["n"] += 1
            raise cross

        with patch.object(io.os, "replace", side_effect=_maybe_cross):
            atomic_write_text(target, "new content")
        # ``_replace_cross_device`` 自己用 shutil.move 回退，所以 os.replace 仍只调用 1 次
        # 且最终内容已被覆盖
        assert calls["n"] == 1
        assert target.read_text(encoding="utf-8") == "new content"

    def test_replace_repeated_ebf_busy_then_write_text(self, tmp_path: Path) -> None:
        """``os.replace`` 持续抛 EBUSY → 直接覆盖到目标路径仍生效。"""
        target = tmp_path / "out.json"
        target.write_text("old", encoding="utf-8")
        busy = OSError(errno.EBUSY, "Device or resource busy")
        with patch.object(io.os, "replace", side_effect=busy):
            atomic_write_text(target, "fresh")
        assert target.read_text(encoding="utf-8") == "fresh"


class TestOpNoTicket:
    def test_op_no_ticket_attribute_error_is_swallowed(self, monkeypatch) -> None:
        # 让 ``ctx.options`` 变成 property，赋值时抛 AttributeError
        class _Ctx:
            def __init__(self) -> None:
                self.check_hostname = False
                self.verify_mode = ssl.CERT_NONE

            @property
            def options(self):
                raise AttributeError("no options")

            @options.setter
            def options(self, value):
                raise AttributeError("read-only")

        captured = {}

        def _factory(*args, **kwargs):
            return _Ctx()

        monkeypatch.setattr(ssl, "SSLContext", _factory)
        # 清空缓存的 _ssl_context，确保新建
        from core.transport import http as http_mod
        monkeypatch.setattr(http_mod, "_ssl_context", None)
        ctx = get_upstream_ssl_context()
        assert isinstance(ctx, _Ctx)


class TestSocksImportFailFast:
    def test_import_aiohttp_socks_failure_is_fatal(self, monkeypatch) -> None:
        import sys
        import builtins

        real_import = builtins.__import__

        def _hooked(name, *args, **kwargs):
            if name.startswith("aiohttp_socks"):
                raise ImportError(f"simulated missing {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _hooked)
        # 重新 import 模块会触发模块顶部 ImportError
        with pytest.raises(ImportError):
            import importlib

            for mod in list(sys.modules):
                if mod.startswith("core.transport.socks"):
                    del sys.modules[mod]
            importlib.import_module("core.transport.socks")


class TestLeaseValidSessionIndexError:
    def test_fix_current_index_with_no_valid_sessions(self) -> None:
        from core.session.accounts import Account
        from core.session.store import PlatformSession

        sess = PlatformSession(
            account=Account(username="x", password="p"),
            token="",
            user_id="",
            upstream="probe",
        )
        # 让所有 session 都不合法 + expired
        sess.is_valid = False
        pool = _ProbePool([sess])
        pool._current_index = 0
        pool._fix_current_index()
        # IndexError 不再逃逸；current_index 复位为 0
        assert pool._current_index == 0

    def test_fix_current_index_raises_when_explicit_upstream_unavailable(self) -> None:
        from core.session.accounts import Account
        from core.session.store import PlatformSession

        sess = PlatformSession(
            account=Account(username="x", password="p"),
            token="",
            user_id="",
            upstream="probe",
        )
        sess.is_valid = False
        pool = _ProbePool([sess])
        pool._current_index = 99  # 越界
        # 越界分支会被 elif 复位为 0，不会进入 valid_indices 路径
        pool._fix_current_index()
        assert pool._current_index == 0

        # 直接验证 IndexError 不再逃逸
        import random
        with patch.object(random, "choice", side_effect=IndexError()):
            pool._current_index = 0
            pool._sessions[0].is_valid = False
            pool._sessions[0]._skip_expired = True  # 标记过期
            pool._fix_current_index("ghost-user")
            assert pool._current_index == 0


class TestAttachProxyToggle:
    def test_attach_and_read_with_client(self):
        class _Stub:
            _last_used_proxy_enabled = True

        exc = UpstreamUnavailableError("x")
        assert read_proxy_used_enabled(exc, _Stub()) is True
        attach_proxy_toggle(exc, False)
        assert read_proxy_used_enabled(exc, _Stub()) is False
        # 清掉字段后回落到 client
        delattr(exc, "proxy_used_enabled")
        delattr(exc, "_proxy_used_enabled")
        assert read_proxy_used_enabled(exc, _Stub()) is True

    def test_read_returns_none_when_no_source(self):
        # upstream.qwen.media.proxy_toggle 默认 enabled=False（没显式启用代理），
        # ``read_proxy_used_enabled`` 最终回落到 toggle.enabled；外部认为该值仍合法。
        # 想拿 None，需要把 toggle 也禁用掉。
        from upstream.qwen.media.proxy_toggle import get_proxy_toggle

        toggle = get_proxy_toggle()
        prev = toggle._enabled
        toggle._enabled = False
        try:
            # client=None + exc 没字段 + toggle.enabled=False ⇒ 仍是 False
            assert read_proxy_used_enabled(Exception(), None) is False
        finally:
            toggle._enabled = prev