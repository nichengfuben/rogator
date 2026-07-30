from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upstream.cursor.auth.store import auth_path, get_access_token, write_auth
from upstream.cursor.client import _merge_model_inventory, _model_ids_from_config
from upstream.cursor.models.identity import (
    external_id_for,
    meta_for_model,
    parse_cursor_model_id,
)
from upstream.cursor.models.store import load_merged, read_cache, write_cache
from upstream.cursor.chat.convert import messages_to_cursor_history, split_prompt_and_history
from upstream.cursor.auth.token_pool import KeyPool, is_limit_reached, parse_usage


class TestCursorAuthStore(unittest.TestCase):
    def test_write_and_read_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(write_auth("tok123", "ref456", root=root))
            self.assertEqual(get_access_token(root=root), "tok123")
            text = auth_path(root).read_text(encoding="utf-8")
            self.assertIn("accessToken", text)
            self.assertIn("ref456", text)


class TestCursorModelsStore(unittest.TestCase):
    def test_read_write_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cache(
                ["composer-2.5-fast", "composer-2.5"],
                {"composer-2.5-fast": {"id": "composer-2.5-fast", "owned_by": "cursor"}},
                root=root,
            )
            ids, meta, updated_at = read_cache(root=root)
            self.assertEqual(ids, ["composer-2.5-fast", "composer-2.5"])
            self.assertIn("composer-2.5-fast", meta)
            self.assertGreater(updated_at, 0)

    def test_load_merged_prefers_config_then_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cache(["remote-model"], {}, root=root)
            models, _, _ = load_merged(["composer-2.5-fast"], root=root)
            self.assertEqual(models[0], "composer-2.5-fast")
            self.assertIn("remote-model", models)


class TestCursorModelIdentity(unittest.TestCase):
    def test_parse_effort_suffix(self) -> None:
        identity = parse_cursor_model_id("claude-opus-5-high")
        assert identity is not None
        self.assertEqual(identity.base_name, "claude-opus-5")
        self.assertEqual(identity.effort, "high")
        self.assertFalse(identity.thinking)

    def test_parse_thinking_and_fast(self) -> None:
        identity = parse_cursor_model_id("claude-opus-4-7-high-fast")
        assert identity is not None
        self.assertEqual(identity.effort, "high")
        self.assertTrue(identity.fast)

    def test_meta_includes_effort(self) -> None:
        meta = meta_for_model("claude-opus-5-low")
        self.assertEqual(meta["cursor_effort"], "low")
        self.assertEqual(meta["think_efforts"]["default_effort"], "low")

    def test_rejects_dirty_model_id(self) -> None:
        self.assertIsNone(parse_cursor_model_id("{'5-turbo': 'composer-2'}"))

    def test_external_id_replaces_dots(self) -> None:
        self.assertEqual(external_id_for("composer-2.5-fast"), "composer-2-5-fast")


class TestCursorModelInventory(unittest.TestCase):
    def test_merge_api_and_mapping(self) -> None:
        ids, meta = _merge_model_inventory([
            {"modelId": "composer-2.5-fast", "displayName": "Fast"},
            {"modelId": "composer-2.5", "displayName": "Standard"},
            {"modelId": "{'5-turbo': 'composer-2'}"},
        ])
        self.assertIn("composer-2.5-fast", ids)
        self.assertIn("composer-2.5", ids)
        self.assertNotIn("{'5-turbo': 'composer-2'}", ids)
        self.assertIn("display_name", meta["composer-2.5-fast"])
        self.assertEqual(meta["composer-2.5-fast"]["display_name"], "Fast")

    def test_config_fallback_ids(self) -> None:
        with patch("upstream.cursor.client.load_cursor_upstream_config", return_value={
            "models": {"default": "composer-2.5-fast", "mapping": {"gpt-4": "composer-2.5-fast"}, "fallback": ["legacy-model"]},
            "cursor": {},
        }):
            ids = _model_ids_from_config()
        self.assertIn("gpt-4", ids)
        self.assertIn("legacy-model", ids)


class TestCursorConfig(unittest.TestCase):
    def test_default_starcursor_section(self) -> None:
        with patch("server.config.app_config._load_upstream_toml", return_value={}):
            from upstream.cursor.setup.config import load_cursor_upstream_config
            cfg = load_cursor_upstream_config()
        sc = cfg["starcursor"]
        self.assertIn("base_url", sc)
        self.assertIn("poll_interval", sc)
        self.assertEqual(sc["poll_interval"], 30)


class TestCursorConverter(unittest.TestCase):
    def test_split_prompt_and_history(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, "again")
        self.assertEqual(len(history), 2)
        cursor_hist = messages_to_cursor_history(messages[:-1])
        self.assertEqual(history, cursor_hist)


class TestCursorTokenHelpers(unittest.TestCase):
    def test_key_pool_rotates(self) -> None:
        pool = KeyPool(["k1", "k2"], threshold=80, refresh_interval=60)
        first = pool.current
        assert first is not None
        pool.switch_next()
        second = pool.current
        assert second is not None
        self.assertNotEqual(first.key, second.key)

    def test_usage_limit(self) -> None:
        u = parse_usage({
            "individualUsage": {"plan": {
                "autoPercentUsed": 96,
                "apiPercentUsed": 94,
                "breakdown": {"total": 100},
            }},
        })
        self.assertTrue(is_limit_reached(u, threshold=95.0))


class TestCursorClientStartup(unittest.IsolatedAsyncioTestCase):
    async def test_startup_without_keys_warns(self) -> None:
        from upstream.cursor.client import CursorClient

        with patch("upstream.cursor.client.starcursor_config", return_value={"api_keys": [], "poll_interval": 30}):
            with patch("upstream.cursor.client.get_access_token", return_value=None):
                with patch("upstream.cursor.client.CursorClient.fetch_models") as fetch_mock:
                    client = CursorClient(None)
                    await client.startup()
        fetch_mock.assert_not_called()
        self.assertTrue(client._startup_done)


class TestDeepSeekModelsCache(unittest.TestCase):
    def test_load_cache_reads_updated_at(self) -> None:
        from upstream.deepseek.client import DeepSeekClient

        with patch("upstream.deepseek.client.read_models_cache", return_value=(["deepseek-v4-pro"], 1234567890.0)):
            client = DeepSeekClient(None)
            models = client.load_models_cache()
        self.assertEqual(models, ["deepseek-v4-pro"])
        self.assertEqual(client._models_fetch_time, 1234567890.0)


class TestCursorExecTools(unittest.TestCase):
    def test_git_diff_request(self) -> None:
        from upstream.cursor.stream.exec import execute_tool

        results = execute_tool({
            "id": 1,
            "gitDiffRequest": {"files": [], "baseBranch": "HEAD"},
        })
        self.assertEqual(len(results), 1)
        self.assertIn("gitDiffResponse", results[0])

    def test_execute_hook_args(self) -> None:
        from upstream.cursor.stream.exec import execute_tool

        results = execute_tool({
            "id": 2,
            "executeHookArgs": {"request": {"preToolUse": {"case": "preToolUse"}}},
        })
        self.assertEqual(len(results), 1)
        hook = results[0]["executeHookResult"]["response"]
        self.assertIn("preToolUse", hook)
        self.assertEqual(hook["preToolUse"]["permission"], "allow")


class _FakeH2Conn:
    def __init__(self, windows: list[int]) -> None:
        self._windows = list(windows)
        self.sent_chunks: list[bytes] = []
        self.recv_calls = 0

    def local_flow_control_window(self, _stream_id: int) -> int:
        return self._windows[0] if self._windows else 65536

    def send_data(self, _stream_id: int, piece: bytes, end_stream: bool = False) -> None:
        self.sent_chunks.append(piece)
        if self._windows:
            self._windows[0] -= len(piece)

    def receive_data(self, _chunk: bytes) -> list[object]:
        self.recv_calls += 1
        if self._windows and self._windows[0] <= 0:
            self._windows.pop(0)
        return []

    def data_to_send(self) -> bytes:
        return b"ack"

    def get_next_available_stream_id(self) -> int:
        return 1

    def send_headers(self, *_args, **_kwargs) -> None:
        return None


class _FakeSock:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.recv_count = 0

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, _size: int) -> bytes:
        self.recv_count += 1
        return b"window-update"


class TestCursorStreamFlowControl(unittest.TestCase):
    def test_safe_send_data_splits_large_payload(self) -> None:
        from upstream.cursor.stream.proto import safe_send_data

        conn = _FakeH2Conn([32768, 65536])
        sock = _FakeSock()
        payload = b"x" * 71552

        safe_send_data(conn, sock, 1, payload)

        self.assertEqual(sum(len(piece) for piece in conn.sent_chunks), len(payload))
        self.assertGreater(len(conn.sent_chunks), 1)
        self.assertLessEqual(max(len(piece) for piece in conn.sent_chunks), 16384)
        self.assertEqual(conn.recv_calls, 1)

    def test_safe_send_data_with_lock_does_not_reenter(self) -> None:
        """回归：持锁发送时不可再二次 acquire 非重入 Lock（kv 回复会死锁）。"""
        import threading

        from upstream.cursor.stream.proto import safe_send_data

        conn = _FakeH2Conn([65536])
        sock = _FakeSock()
        lock = threading.Lock()
        payload = b"kv-reply"

        done = threading.Event()

        def _run() -> None:
            safe_send_data(conn, sock, 1, payload, sock_lock=lock)
            done.set()

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        self.assertTrue(done.wait(2.0), "safe_send_data deadlocked on sock_lock")
        self.assertEqual(b"".join(conn.sent_chunks), payload)

    def test_worker_uses_safe_send_data_for_run_request(self) -> None:
        from upstream.cursor.stream.worker import _run_agent_stream

        q = object()
        token = {"accessToken": "tok", "machineId": "mid", "macMachineId": "mmid"}
        conn = _FakeH2Conn([65536, 65536])
        sock = _FakeSock()
        calls: list[int] = []

        def _fake_safe_send(_conn, _sock, _stream_id: int, data: bytes, *, sock_lock=None) -> None:
            calls.append(len(data))

        with patch("upstream.cursor.stream.worker.agent_config", return_value={}):
            with patch("upstream.cursor.stream.worker.agent_host", return_value="example.com"):
                with patch("upstream.cursor.stream.worker._open_h2_socket", return_value=(sock, conn)):
                    with patch("upstream.cursor.stream.worker.safe_send_data", side_effect=_fake_safe_send):
                        with patch("upstream.cursor.stream.worker.run_agent_loop") as loop_mock:
                            _run_agent_stream(
                                q, token, "hello", "composer-2.5-fast", None, None, None, "X:/ws",
                                files=[{"path": "big.txt", "content": "x" * 70000}],
                            )
        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0], 65536)
        loop_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
