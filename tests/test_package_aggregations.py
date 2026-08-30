"""``core.session`` 与 ``errors`` package 的聚合 import 可用。"""

from __future__ import annotations


def test_core_session_package_reexports() -> None:
    from core.session import (  # noqa: F401
        Account,
        CLEANUP_INTERVAL,
        MUTE_LOGIN_BLOCK_SECONDS,
        PlatformSession,
        clean_expired,
        is_account_mute_blocked,
        load_upstream_sessions,
        mark_invalid,
        mask_username,
        prune_expired_muted_accounts,
        remove_by_username,
        replace_or_append,
        save_upstream_sessions,
        save_upstream_sessions_async,
        valid_session_count,
    )


def test_server_formats_reexports_proxy_helpers() -> None:
    from server.formats import (  # noqa: F401
        BaxiaSmBlockedError,
        DataInspectionFailedError,
        UpstreamConnectionError,
        UpstreamStsError,
        UpstreamTimeoutError,
        UpstreamUnavailableError,
        UpstreamWafBlockedError,
        UpstreamWafBlockedErrorWithProxy,
        attach_proxy_toggle,
        read_proxy_used_enabled,
    )


def test_core_transport_socks_reexports() -> None:
    from core.transport.socks import ProxyConnector, socks_connector  # noqa: F401


def test_core_persist_locks_reexports() -> None:
    from core.persist.locks import (  # noqa: F401
        get_async_lock,
        mark_migrated,
        reset_async_lock,
        save_lock,
        save_lock_async,
    )