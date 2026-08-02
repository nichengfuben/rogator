from __future__ import annotations

"""config / model_thinking / session_retry 单元测试。"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.qwen.account import Account
from dataclasses import replace
from server.config import CONFIG, AppConfig, get_config, load_config, reload_config
from server.config.app_config import LiveConfig, _load_upstream_toml, _loads_toml
from server.config.reload import RESTART_REQUIRED, apply_runtime_config
from server.config.files import (
    PROJECT_ROOT,
    ensure_user_config_file,
    overlay_user_config,
    read_server_version,
    warn_if_config_version_mismatch,
)
from server.model.model_registry import get_model_registry, load_model_registry, reload_model_registry
from server.model.model_thinking import resolve_thinking_route, uses_entml_thinking
from server.formats import UpstreamTimeoutError
from server.retry import parse_rate_limit_block_seconds, run_with_session_retry, stream_with_session_retry
from upstream.qwen.chat.store import QwenSession, save_sessions, load_session_store
from server.formats import TokenExpiredError

_PROJECT_TEMPLATE = PROJECT_ROOT / "template" / "config.toml"


def _write_user_config(tmp: str, user_text: str) -> tuple[Path, Path]:
    root = Path(tmp)
    user_path = root / "config.toml"
    user_path.write_text(user_text, encoding="utf-8")
    return user_path, _PROJECT_TEMPLATE


class TestConfig(unittest.TestCase):
    def test_load_config_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(tmp) / "missing.toml", template_path=_PROJECT_TEMPLATE)

    def test_load_config_template_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path = Path(tmp) / "config.toml"
            user_path.write_text("[server]\nport = 8932\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_config(user_path, template_path=Path(tmp) / "missing-template.toml")

    def test_load_config_invalid_toml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(tmp, "bad[[[")
            with self.assertRaises(ValueError):
                load_config(user_path, template_path=tpl_path)

    def test_load_config_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, '[server]\nprelogin = 5\n[retry]\nmax_retry_on_error = 2\n',
            )
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.prelogin, 5)
            self.assertEqual(cfg.max_retry_on_error, 2)

    def test_load_config_limits_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(tmp, "[server]\nport = 8932\n")
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.qwen_send_max_chars, 1_024_000)
            self.assertEqual(cfg.model_context_length, 256_000)
            self.assertFalse(cfg.send_full_prompt)
            self.assertEqual(cfg.prelogin, 32)
            self.assertEqual(cfg.login_interval, 15.0)

    def test_load_config_login_interval_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, "[server]\nlogin_interval = 5.0\n",
            )
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.login_interval, 5.0)

    def test_load_config_models_refresh_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(tmp, "[server]\nport = 8932\n")
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.models_refresh_interval, 3600.0)

    def test_load_config_upstream_enabled_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(tmp, "[server]\nport = 8932\n")
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.upstream_enabled, ("qwen",))

    def test_load_config_upstream_enabled_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, '[upstream]\nenabled = ["qwen", "deepseek"]\n',
            )
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.upstream_enabled, ("qwen", "deepseek"))

    def test_load_config_send_full_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, "[limits]\nsend_full_prompt = true\n",
            )
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertTrue(cfg.send_full_prompt)

    def test_load_config_shutdown_wait_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(tmp, "[server]\nport = 8932\n")
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.shutdown_wait_active_requests, 3.0)
            self.assertEqual(cfg.shutdown_total_timeout, 8.0)
            self.assertEqual(cfg.shutdown_hard_exit_timeout, 25.0)

    def test_load_config_shutdown_user_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, "[shutdown]\nwait_active_requests = 5.0\n",
            )
            cfg = load_config(user_path, template_path=tpl_path)
            self.assertEqual(cfg.shutdown_wait_active_requests, 5.0)
            self.assertEqual(cfg.shutdown_total_timeout, 8.0)
            self.assertEqual(cfg.shutdown_hard_exit_timeout, 25.0)

    def test_load_upstream_toml_overlay_chain(self) -> None:
        raw = _load_upstream_toml("qwen")
        self.assertEqual(raw.get("limits", {}).get("qwen_send_max_chars"), 1_024_000)
        caps = raw.get("capabilities") or {}
        self.assertTrue(caps.get("chat"))
        self.assertTrue(caps.get("vision"))

    def test_overlay_user_config_keeps_template_sections(self) -> None:
        template = {"server": {"port": 8932, "prelogin": 32}, "limits": {"max_concurrent": 32}}
        user = {"server": {"port": 9000}}
        merged = overlay_user_config(template, user)
        self.assertEqual(merged["server"]["port"], 9000)
        self.assertEqual(merged["server"]["prelogin"], 32)
        self.assertEqual(merged["limits"]["max_concurrent"], 32)

    def test_loads_toml_parses_sections(self) -> None:
        data = _loads_toml('[server]\nport = 9000\nhost = "127.0.0.1"\n')
        self.assertEqual(data["server"]["port"], 9000)
        self.assertEqual(data["server"]["host"], "127.0.0.1")

    def test_read_server_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.toml"
            p.write_text('[server]\nversion = "1.0.0"\n', encoding="utf-8")
            self.assertEqual(read_server_version(p), "1.0.0")
            self.assertIsNone(read_server_version(Path(tmp) / "missing.toml"))

    def test_warn_config_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl_dir = root / "template"
            tpl_dir.mkdir()
            tpl_dir.joinpath("config.toml").write_text(
                '[server]\nversion = "2.0.0"\n', encoding="utf-8",
            )
            user_cfg = root / "config.toml"
            user_cfg.write_text('[server]\nversion = "1.0.0"\n', encoding="utf-8")
            mock_logger = MagicMock()
            with patch("server.config.files.PROJECT_ROOT", root), \
                 patch("server.config.files.TEMPLATE_DIR", tpl_dir), \
                 patch("server.config.files.USER_CONFIG_PATH", user_cfg):
                warn_if_config_version_mismatch(user_cfg, mock_logger)
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            self.assertEqual(args[1], "1.0.0")
            self.assertEqual(args[2], "2.0.0")

    def test_warn_config_version_missing_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl_dir = root / "template"
            tpl_dir.mkdir()
            tpl_dir.joinpath("config.toml").write_text(
                '[server]\nversion = "2.0.0"\n', encoding="utf-8",
            )
            user_cfg = root / "config.toml"
            user_cfg.write_text('[server]\nport = 8932\n', encoding="utf-8")
            mock_logger = MagicMock()
            with patch("server.config.files.PROJECT_ROOT", root), \
                 patch("server.config.files.TEMPLATE_DIR", tpl_dir), \
                 patch("server.config.files.USER_CONFIG_PATH", user_cfg):
                warn_if_config_version_mismatch(user_cfg, mock_logger)
            mock_logger.warning.assert_not_called()

    def test_warn_config_version_match_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tpl_dir = root / "template"
            tpl_dir.mkdir()
            content = '[server]\nversion = "2.0.0"\n'
            tpl_dir.joinpath("config.toml").write_text(content, encoding="utf-8")
            user_cfg = root / "config.toml"
            user_cfg.write_text(content, encoding="utf-8")
            mock_logger = MagicMock()
            with patch("server.config.files.PROJECT_ROOT", root), \
                 patch("server.config.files.TEMPLATE_DIR", tpl_dir), \
                 patch("server.config.files.USER_CONFIG_PATH", user_cfg):
                warn_if_config_version_mismatch(user_cfg, mock_logger)
            mock_logger.warning.assert_not_called()

    def test_ensure_user_config_from_legacy_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            target = config_dir / "config.toml"
            legacy_root = root / "config.toml"
            legacy_root.write_text('[server]\nport = 9001\n', encoding="utf-8")
            tpl = root / "template" / "config.toml"
            tpl.parent.mkdir()
            shutil.copy2(_PROJECT_TEMPLATE, tpl)
            with patch("server.config.files.migrate_config_layout"), \
                 patch("server.config.files.PROJECT_ROOT", root), \
                 patch("server.config.files.USER_CONFIG_DIR", config_dir), \
                 patch("server.config.files.USER_CONFIG_PATH", target), \
                 patch("server.config.files.LEGACY_ROOT_CONFIG", legacy_root):
                path = ensure_user_config_file()
            self.assertEqual(path, target)
            self.assertTrue(path.is_file())
            cfg = load_config(path, template_path=tpl)
            self.assertEqual(cfg.port, 9001)


class TestConfigReload(unittest.TestCase):
    def test_live_config_identity_survives_swap(self) -> None:
        self.assertIsInstance(CONFIG, LiveConfig)
        imported = CONFIG
        old = get_config()
        new = replace(old, max_retry_on_error=old.max_retry_on_error + 1)
        try:
            CONFIG.swap(new)
            self.assertIs(imported, CONFIG)
            self.assertEqual(imported.max_retry_on_error, new.max_retry_on_error)
            self.assertEqual(get_config().max_retry_on_error, new.max_retry_on_error)
        finally:
            CONFIG.swap(old)

    def test_reload_config_applies_hot_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, "[retry]\nmax_retry_on_error = 9\n",
            )
            old = get_config()
            try:
                ok = reload_config(path=user_path, template_path=tpl_path)
                self.assertTrue(ok)
                self.assertEqual(CONFIG.max_retry_on_error, 9)
            finally:
                CONFIG.swap(old)

    def test_reload_config_invalid_keeps_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_path = root / "config.toml"
            user_path.write_text("bad[[[", encoding="utf-8")
            old = get_config()
            ok = reload_config(path=user_path, template_path=_PROJECT_TEMPLATE)
            self.assertFalse(ok)
            self.assertEqual(get_config(), old)

    def test_reload_restart_required_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_path, tpl_path = _write_user_config(
                tmp, "[server]\nport = 19999\n",
            )
            old = get_config()
            mock_logger = MagicMock()
            try:
                with patch("server.config.reload.logger", mock_logger):
                    ok = reload_config(path=user_path, template_path=tpl_path)
                self.assertTrue(ok)
                self.assertEqual(CONFIG.port, 19999)
                self.assertIn("port", RESTART_REQUIRED)
                warn_msgs = " ".join(
                    str(c) for c in mock_logger.warning.call_args_list
                )
                self.assertIn("需重启", warn_msgs)
            finally:
                CONFIG.swap(old)

    def test_apply_runtime_updates_queue_and_splitter(self) -> None:
        import state as state_mod

        old_cfg = get_config()
        old_queue = state_mod.MAX_QUEUE_SIZE
        splitter = state_mod.LongTextSplitter(
            max_chars=100, send_full_prompt=False,
        )
        state = MagicMock()
        state.splitter = splitter
        state._clients = {}
        state.scheduler = MagicMock()
        state.scheduler.wake_for_config = AsyncMock()
        new_cfg = replace(
            old_cfg,
            max_queue_size=old_cfg.max_queue_size + 7,
            qwen_send_max_chars=12345,
            send_full_prompt=True,
            prelogin=old_cfg.prelogin,
        )
        try:
            apply_runtime_config(old_cfg, new_cfg, state)
            self.assertEqual(state_mod.MAX_QUEUE_SIZE, new_cfg.max_queue_size)
            self.assertEqual(splitter.max_chars, 12345)
            self.assertTrue(splitter.send_full_prompt)
        finally:
            state_mod.MAX_QUEUE_SIZE = old_queue


class TestModelThinking(unittest.TestCase):
    def test_registry_entml_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text(
                "qwen3-7-max:qwen3.7-max:true\n"
                "qwen3-8-max-preview:qwen3.8-max-preview:false\n",
                encoding="utf-8",
            )
            reg = load_model_registry(p)
            self.assertTrue(reg.by_internal["qwen3.7-max"].uses_entml)
            self.assertFalse(reg.by_internal["qwen3.8-max-preview"].uses_entml)

    def test_resolve_entml_model(self) -> None:
        route = resolve_thinking_route("qwen3.7-max", "on")
        self.assertTrue(route.use_entml)
        self.assertFalse(route.qwen_native_enabled)

    def test_resolve_deepseek_entml(self) -> None:
        route = resolve_thinking_route("deepseek-v4-flash", "high")
        self.assertTrue(route.use_entml)
        self.assertFalse(route.qwen_native_enabled)

    def test_resolve_native_model_on(self) -> None:
        route = resolve_thinking_route("qwen3.8-max-preview", "on")
        self.assertFalse(route.use_entml)
        self.assertTrue(route.qwen_native_enabled)
        self.assertEqual(route.qwen_native_mode, "Thinking")

    def test_resolve_native_model_off(self) -> None:
        route = resolve_thinking_route("qwen3.8-max-preview", "off")
        self.assertFalse(route.use_entml)
        self.assertTrue(route.qwen_native_enabled)
        self.assertEqual(route.qwen_native_mode, "Thinking")

    def test_resolve_native_auto(self) -> None:
        route = resolve_thinking_route("qwen3.8-max-preview", "auto")
        self.assertTrue(route.qwen_native_enabled)
        self.assertEqual(route.qwen_native_mode, "Thinking")

    def test_resolve_native_none(self) -> None:
        route = resolve_thinking_route("qwen3.8-max-preview", "none")
        self.assertFalse(route.use_entml)
        self.assertTrue(route.qwen_native_enabled)
        self.assertEqual(route.qwen_native_mode, "Thinking")


class TestThinkingLevels(unittest.TestCase):
    def test_build_protocol_options_levels(self) -> None:
        from handlers.openai import _build_protocol_options, protocol_thinking_level
        from echotools.exec.fncall.protocols.entml_think.core import (
            default_max_thinking_length_for_level,
            resolve_thinking_injection,
        )

        cases = {
            "none": (None, None),
            "low": ("low", 12800),
            "medium": ("medium", 25600),
            "high": ("high", 64000),
            "xhigh": ("xhigh", 102400),
            "max": ("max", 134736),
            "auto": ("auto", None),
        }
        for level, (mode, default_max) in cases.items():
            opts = _build_protocol_options({"thinking_level": level}) or {}
            self.assertEqual(protocol_thinking_level(opts), level)
            if level == "none":
                self.assertIsNone(resolve_thinking_injection(opts))
                continue
            resolved = resolve_thinking_injection(opts)
            assert resolved is not None
            inj_mode, max_len = resolved
            self.assertEqual(inj_mode, mode)
            self.assertEqual(max_len, default_max)

    def test_build_protocol_options_reasoning_effort(self) -> None:
        from handlers.openai import _build_protocol_options, protocol_thinking_level

        opts = _build_protocol_options({"reasoning_effort": "high"}) or {}
        self.assertEqual(protocol_thinking_level(opts), "high")
        self.assertEqual(opts.get("thinking_level"), "high")

    def test_build_protocol_options_off_disables_thinking(self) -> None:
        from handlers.openai import _build_protocol_options, protocol_thinking_level
        from echotools.exec.fncall.protocols.entml_think.core import resolve_thinking_injection

        for body in (
            {"reasoning_effort": "off"},
            {"reasoning_effort": "none"},
            {"thinking": False},
            {"thinking": "off"},
            {"thinking_level": "off"},
        ):
            opts = _build_protocol_options(body) or {}
            self.assertEqual(protocol_thinking_level(opts), "none", msg=str(body))
            self.assertEqual(opts.get("thinking_level"), "none", msg=str(body))
            self.assertIsNone(resolve_thinking_injection(opts), msg=str(body))

    def test_models_list_think_efforts(self) -> None:
        from server.model.model_catalog import (
            MODEL_CONTEXT_LENGTH,
            build_openai_model_entry,
            model_supports_thinking,
        )
        from server.model.model_registry import get_model_registry

        registry = get_model_registry()
        entry37 = registry.by_external["qwen3-7-max"]
        entry38 = registry.by_external["qwen3-8-max-preview"]
        self.assertTrue(model_supports_thinking(entry37))
        self.assertTrue(model_supports_thinking(entry38))
        entry = build_openai_model_entry(
            "qwen3-7-max",
            registry_entry=entry37,
        )
        self.assertEqual(entry.get("context_length"), MODEL_CONTEXT_LENGTH)
        te = entry.get("think_efforts") or {}
        self.assertTrue(te.get("support"))
        self.assertNotIn("none", te.get("valid_efforts", []))
        self.assertEqual(te.get("off_effort"), "none")
        self.assertEqual(te.get("default_effort"), "medium")
        entry38_out = build_openai_model_entry(
            "qwen3-8-max-preview",
            registry_entry=entry38,
        )
        self.assertTrue(entry38_out.get("always_thinking"))
        self.assertNotIn("think_efforts", entry38_out)

    def test_inject_renders_thinking_behavior(self) -> None:
        from echotools import get_protocol, inject_fncall
        from handlers.openai import _build_protocol_options, _inject_protocol_options

        opts = _inject_protocol_options(
            _build_protocol_options({"thinking_level": "medium"}), True,
        )
        prompt = inject_fncall(
            [{"role": "user", "content": "hi"}],
            [],
            get_protocol("entml"),
            protocol_options=opts,
        )[0]["content"]
        self.assertIn("<thinking_behavior>", prompt)
        self.assertIn("<entml:thinking_mode>medium</entml:thinking_mode>", prompt)
        self.assertIn("<entml:max_thinking_length>25600</entml:max_thinking_length>", prompt)

    def test_inject_thinking_off_with_history_thinking_forces_no_think(self) -> None:
        from echotools import get_protocol, inject_fncall
        from handlers.openai import _build_protocol_options, _inject_protocol_options

        opts = _inject_protocol_options(
            _build_protocol_options({"thinking_level": "none"}), True,
        )
        msgs = [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "reasoning": "应先查天气。",
                "content": "好的。",
            },
            {"role": "user", "content": "next"},
        ]
        prompt = inject_fncall(
            msgs,
            [],
            get_protocol("entml"),
            protocol_options=opts,
        )[0]["content"]
        self.assertIn("<entml:thinking>", prompt.split("<current_user_message>")[0])
        self.assertNotIn("<entml:thinking_mode>", prompt)
        self.assertIn("<thinking_behavior>", prompt)
        self.assertIn("Do NOT output a <entml:thinking> block", prompt)

    def test_inject_thinking_off_without_history_thinking_omits_behavior(self) -> None:
        from echotools import get_protocol, inject_fncall
        from handlers.openai import _build_protocol_options, _inject_protocol_options

        opts = _inject_protocol_options(
            _build_protocol_options({"thinking_level": "none"}), True,
        )
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        prompt = inject_fncall(
            msgs,
            [],
            get_protocol("entml"),
            protocol_options=opts,
        )[0]["content"]
        self.assertNotIn("<entml:thinking_mode>", prompt)
        self.assertNotIn("<thinking_behavior>", prompt)

    def test_inject_no_tools_thinking_behavior_omits_invoke(self) -> None:
        from echotools import get_protocol, inject_fncall
        from handlers.openai import _build_protocol_options, _inject_protocol_options

        opts = _inject_protocol_options(
            _build_protocol_options({"thinking_level": "medium"}), True,
        )
        prompt = inject_fncall(
            [{"role": "user", "content": "hi"}],
            [],
            get_protocol("entml"),
            protocol_options=opts,
        )[0]["content"]
        self.assertIn("<thinking_behavior>", prompt)
        self.assertNotIn("<entml:invoke>", prompt.split("</thinking_behavior>")[0])


class TestMessageHistory(unittest.TestCase):
    def test_inject_renders_reasoning_in_history(self) -> None:
        from echotools import get_protocol, inject_fncall
        from handlers.openai import _build_protocol_options, _inject_protocol_options

        msgs = [{
            "role": "assistant",
            "reasoning": "step one",
            "content": "answer text",
        }, {"role": "user", "content": "follow up"}]
        opts = _inject_protocol_options(
            _build_protocol_options({"thinking_level": "medium"}), True,
        )
        prompt = inject_fncall(msgs, [], get_protocol("entml"), protocol_options=opts)[0]["content"]
        self.assertIn("<entml:thinking>", prompt)
        self.assertIn("step one", prompt)
        self.assertIn("answer text", prompt)


class TestSessionStoreMeta(unittest.TestCase):
    def test_save_load_current_index(self) -> None:
        from tests.test_session_cleanup import _make_jwt

        with tempfile.TemporaryDirectory() as tmp:
            sessions_file = Path(tmp) / "qwen" / "sessions.json"
            with patch("core.session.store.sessions_file", return_value=sessions_file):
                acc = Account(username="a@test.com", password="pw")
                s = QwenSession(
                    account=acc, token=_make_jwt(time.time() + 3600),
                    user_id="u", login_time=time.time(),
                )
                save_sessions(
                    [s], current_index=0,
                    blocked_accounts={"a@test.com": time.time() + 3600},
                )
                loaded, meta = load_session_store()
                self.assertEqual(len(loaded), 1)
                self.assertEqual(meta.current_index, 0)
                self.assertIn("a@test.com", meta.blocked_accounts)

    def test_atomic_write_falls_back_when_replace_fails(self) -> None:
        from core.session.io import atomic_write_text

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sessions.json"
            payload = '{"count": 1}'
            with patch("core.session.io.os.replace", side_effect=OSError(5, "拒绝访问")):
                atomic_write_text(target, payload)
            self.assertEqual(target.read_text(encoding="utf-8"), payload)


class TestStreamToolJsonSync(unittest.TestCase):
    def test_arguments_json_equal_ignores_whitespace(self) -> None:
        from handlers.anthropic.stream_tools import _arguments_json_equal

        compact = '{"command":"hello"}'
        spaced = '{"command": "hello"}'
        self.assertTrue(_arguments_json_equal(compact, spaced))


class TestSessionRetry(unittest.TestCase):
    def test_parse_rate_limit_hours(self) -> None:
        msg = '{"data": {"code": "RateLimited", "num": 12}}'
        self.assertEqual(parse_rate_limit_block_seconds(msg), 12 * 3600)

    def test_run_with_session_retry_success(self) -> None:
        state = MagicMock()
        state.client.current_session_username = "old@test.com"
        state.client.switch_to_next = AsyncMock(return_value=MagicMock(username="new@test.com"))

        async def _run():
            return "ok"

        import asyncio
        result = asyncio.run(run_with_session_retry("req-1", state, _run))
        self.assertEqual(result, "ok")

    def test_run_with_session_retry_exhausted(self) -> None:
        state = MagicMock()
        state.client.current_session_username = "old@test.com"
        state.client.block_account = MagicMock()
        state.client.switch_to_next = AsyncMock(return_value=None)

        calls = {"n": 0}

        async def _run():
            calls["n"] += 1
            raise TokenExpiredError("Rate limited: num 12")

        import asyncio
        with patch("server.retry.session_retry.CONFIG", replace(get_config(), max_retry_on_error=1)):
            with self.assertRaises(TokenExpiredError):
                asyncio.run(run_with_session_retry("req-2", state, _run))
        self.assertEqual(calls["n"], 1)

    def test_run_with_session_retry_upstream_timeout(self) -> None:
        state = MagicMock()
        calls = {"n": 0}

        async def _run():
            calls["n"] += 1
            if calls["n"] <= 2:
                raise UpstreamTimeoutError("Create chat timed out after 15s")
            return "ok"

        import asyncio
        with patch("server.retry.session_retry.CONFIG", replace(get_config(), max_retry_on_error=3)):
            result = asyncio.run(run_with_session_retry("req-3", state, _run))
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)

    def test_run_with_session_retry_upstream_timeout_exhausted(self) -> None:
        state = MagicMock()
        calls = {"n": 0}

        async def _run():
            calls["n"] += 1
            raise UpstreamTimeoutError("Create chat timed out after 15s")

        import asyncio
        with patch("server.retry.session_retry.CONFIG", replace(get_config(), max_retry_on_error=3)):
            with self.assertRaises(UpstreamTimeoutError):
                asyncio.run(run_with_session_retry("req-4", state, _run))
        self.assertEqual(calls["n"], 4)

    def test_stream_with_session_retry_early_break_closes_inner(self) -> None:
        import asyncio
        from contextlib import aclosing

        state = MagicMock()
        closed = {"n": 0}

        async def make_stream():
            try:
                for i in range(10):
                    yield {"type": "answer", "content": str(i)}
                    await asyncio.sleep(0)
            finally:
                closed["n"] += 1

        async def _run():
            async with aclosing(stream_with_session_retry("req-5", state, make_stream)) as events:
                async for event in events:
                    if event["content"] == "2":
                        break

        asyncio.run(_run())
        self.assertEqual(closed["n"], 1)


if __name__ == "__main__":
    unittest.main()
