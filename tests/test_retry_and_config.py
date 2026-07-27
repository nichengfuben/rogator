from __future__ import annotations

"""config / model_thinking / session_retry 单元测试。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from accounts import Account
from server.config import AppConfig, _loads_toml, load_config
from server.model_thinking import load_model_entml_map, resolve_qwen_thinking, uses_entml_thinking
from server.session_retry import parse_rate_limit_block_seconds, run_with_session_retry
from server.session_store import QwenSession, save_sessions, load_session_store
from server.formats import TokenExpiredError


class TestConfig(unittest.TestCase):
    def test_load_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp) / "missing.toml")
            self.assertEqual(cfg.prelogin, 3)
            self.assertEqual(cfg.max_retry_on_error, 3)

    def test_load_config_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.toml"
            p.write_text('[server]\nprelogin = 5\n[retry]\nmax_retry_on_error = 2\n', encoding="utf-8")
            cfg = load_config(p)
            self.assertEqual(cfg.prelogin, 5)
            self.assertEqual(cfg.max_retry_on_error, 2)

    def test_loads_toml_parses_sections(self) -> None:
        data = _loads_toml('[server]\nport = 9000\nhost = "127.0.0.1"\n')
        self.assertEqual(data["server"]["port"], 9000)
        self.assertEqual(data["server"]["host"], "127.0.0.1")


class TestModelThinking(unittest.TestCase):
    def test_entml_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "map.jsonl"
            p.write_text("qwen3.7-max:true\nqwen3.8-max-preview:false\n", encoding="utf-8")
            m = load_model_entml_map(p)
            self.assertTrue(m["qwen3.7-max"])
            self.assertFalse(m["qwen3.8-max-preview"])

    def test_resolve_entml_model(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.7-max", "on")
        self.assertFalse(enabled)
        self.assertEqual(mode, "Fast")
        self.assertTrue(use_entml)

    def test_resolve_native_model_on(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.8-max-preview", "on")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")
        self.assertFalse(use_entml)

    def test_resolve_native_model_off(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.8-max-preview", "off")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")
        self.assertFalse(use_entml)

    def test_resolve_native_auto(self) -> None:
        enabled, mode, _ = resolve_qwen_thinking("qwen3.8-max-preview", "auto")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")

    def test_resolve_native_none(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.8-max-preview", "none")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")
        self.assertFalse(use_entml)


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
        from server.model_catalog import build_openai_model_entry, model_supports_thinking

        self.assertTrue(model_supports_thinking("qwen3.7-max"))
        self.assertTrue(model_supports_thinking("qwen3.8-max-preview"))
        entry = build_openai_model_entry("qwen3.7-max")
        te = entry.get("think_efforts") or {}
        self.assertTrue(te.get("support"))
        self.assertNotIn("none", te.get("valid_efforts", []))
        self.assertEqual(te.get("off_effort"), "none")
        self.assertEqual(te.get("default_effort"), "medium")
        entry38 = build_openai_model_entry("qwen3.8-max-preview")
        self.assertTrue(entry38.get("always_thinking"))
        self.assertNotIn("think_efforts", entry38)

    def test_inject_renders_thinking_behavior(self) -> None:
        from echotools.fncall import get_protocol, inject_fncall
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


class TestMessageHistory(unittest.TestCase):
    def test_inject_renders_reasoning_in_history(self) -> None:
        from echotools.fncall import get_protocol, inject_fncall
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
            sessions_file = Path(tmp) / "sessions.json"
            with patch("server.session_store.SESSIONS_FILE", str(sessions_file)):
                acc = Account(username="a@test.com", password="pw")
                s = QwenSession(
                    account=acc, token=_make_jwt(time.time() + 3600),
                    user_id="u", login_time=time.time(),
                )
                save_sessions(
                    [s], current_index=0, account_index=1,
                    blocked_accounts={"a@test.com": time.time() + 3600},
                )
                loaded, meta = load_session_store()
                self.assertEqual(len(loaded), 1)
                self.assertEqual(meta.current_index, 0)
                self.assertEqual(meta.account_index, 1)
                self.assertIn("a@test.com", meta.blocked_accounts)

    def test_migrate_legacy_sessions_file(self) -> None:
        from tests.test_session_cleanup import _make_jwt

        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "qwen" / "sessions.json"
            target = Path(tmp) / "sessions.json"
            legacy.parent.mkdir(parents=True)
            acc = Account(username="legacy@test.com", password="pw")
            s = QwenSession(
                account=acc, token=_make_jwt(time.time() + 3600),
                user_id="u1", login_time=time.time(),
            )
            legacy.write_text(
                json.dumps({
                    "sessions": [s.to_dict()],
                    "current_index": 2,
                    "account_index": 3,
                    "blocked_accounts": {},
                }),
                encoding="utf-8",
            )
            with patch("server.session_store.SESSIONS_FILE", str(target)), \
                 patch("server.session_store.LEGACY_SESSIONS_FILE", str(legacy)):
                loaded, meta = load_session_store()
                self.assertTrue(target.exists())
                self.assertFalse(legacy.exists())
                self.assertEqual(len(loaded), 1)
                self.assertEqual(meta.current_index, 2)
                self.assertEqual(meta.account_index, 3)


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
        with patch("server.session_retry.CONFIG", AppConfig(max_retry_on_error=1)):
            with self.assertRaises(TokenExpiredError):
                asyncio.run(run_with_session_retry("req-2", state, _run))
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
