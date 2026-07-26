from __future__ import annotations

"""config / model_thinking / message_history / session_retry 单元测试。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from accounts import Account
from server.config import AppConfig, load_config
from server.message_history import embed_reasoning_in_messages, merge_anthropic_assistant_blocks
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
        self.assertEqual(mode, "NoThinking")
        self.assertTrue(use_entml)

    def test_resolve_native_model_on(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.8-max-preview", "on")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")
        self.assertFalse(use_entml)

    def test_resolve_native_model_off(self) -> None:
        enabled, mode, use_entml = resolve_qwen_thinking("qwen3.8-max-preview", "off")
        self.assertFalse(enabled)
        self.assertEqual(mode, "NoThinking")
        self.assertFalse(use_entml)

    def test_resolve_native_auto(self) -> None:
        enabled, mode, _ = resolve_qwen_thinking("qwen3.8-max-preview", "auto")
        self.assertTrue(enabled)
        self.assertEqual(mode, "Thinking")


class TestMessageHistory(unittest.TestCase):
    def test_embed_openai_reasoning(self) -> None:
        msgs = [{
            "role": "assistant",
            "content": "answer text",
            "reasoning": "step one",
        }]
        out = embed_reasoning_in_messages(msgs)
        content = out[0]["content"]
        self.assertIn("<entml:thinking>", content)
        self.assertIn("step one", content)
        self.assertIn("answer text", content)
        self.assertNotIn("reasoning", out[0])

    def test_merge_anthropic_thinking(self) -> None:
        merged = merge_anthropic_assistant_blocks([
            {"type": "thinking", "thinking": "plan A"},
            {"type": "text", "text": "hello"},
        ])
        self.assertIn("<entml:thinking>", merged)
        self.assertIn("plan A", merged)
        self.assertIn("hello", merged)


class TestSessionStoreMeta(unittest.TestCase):
    def test_save_load_current_index(self) -> None:
        from tests.test_session_cleanup import _make_jwt

        with tempfile.TemporaryDirectory() as tmp:
            sessions_file = Path(tmp) / "sessions.json"
            with patch("server.session_store.SESSIONS_FILE", str(sessions_file)), \
                 patch("server.session_store.DATA_DIR", str(tmp)):
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
