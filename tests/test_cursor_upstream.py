from __future__ import annotations

import json
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
    def test_default_token_service_section(self) -> None:
        with patch("server.config.app_config._load_upstream_toml", return_value={}):
            from upstream.cursor.setup.config import load_cursor_upstream_config
            cfg = load_cursor_upstream_config()
        sc = cfg["token_service"]
        self.assertIn("base_url", sc)
        self.assertIn("poll_interval", sc)
        self.assertEqual(sc["poll_interval"], 30)

    def test_legacy_starcursor_section_merged(self) -> None:
        raw = {"starcursor": {"api_keys": ["k1"], "poll_interval": 12}}
        with patch("server.config.app_config._load_upstream_toml", return_value=raw):
            from upstream.cursor.setup.config import load_cursor_upstream_config, token_service_config
            cfg = load_cursor_upstream_config()
            self.assertEqual(cfg["token_service"]["poll_interval"], 12)
            self.assertEqual(token_service_config()["api_keys"], ["k1"])



class TestCursorConverter(unittest.TestCase):
    def test_split_prompt_and_history(self) -> None:
        from upstream.cursor.chat.convert import messages_to_cursor_history, split_prompt_and_history

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

    def test_system_goes_to_custom_prompt_not_history(self) -> None:
        from upstream.cursor.chat.convert import (
            IMPORTANT_MCP_TOOLS_ONLY,
            IMPORTANT_NO_TOOLS,
            build_custom_system_prompt,
            openai_tools_to_mcp,
            rewrite_tool_call_for_openai,
            split_prompt_and_history,
        )

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hi"},
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, "hi")
        self.assertEqual(history, [])
        with_tools = build_custom_system_prompt(
            messages,
            [{"type": "function", "function": {"name": "mcp__echo", "parameters": {}}}],
        )
        self.assertTrue(with_tools.startswith(IMPORTANT_MCP_TOOLS_ONLY))
        self.assertIn("Be concise.", with_tools)
        no_tools = build_custom_system_prompt(messages, None)
        self.assertTrue(no_tools.startswith(IMPORTANT_NO_TOOLS))

        mcp = openai_tools_to_mcp([
            {"type": "function", "function": {"name": "mcp__echo", "parameters": {}}},
        ])
        self.assertEqual(mcp[0]["name"], "mcp__echo")
        self.assertEqual(mcp[0]["toolName"], "mcp__echo")
        self.assertNotIn("providerIdentifier", mcp[0])

        mcp2 = openai_tools_to_mcp([
            {
                "type": "function",
                "function": {
                    "name": "mcp__fs__read_file",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            },
        ])
        self.assertEqual(mcp2[0]["name"], "mcp__fs__read_file")
        self.assertEqual(mcp2[0]["providerIdentifier"], "fs")
        self.assertEqual(mcp2[0]["toolName"], "read_file")

        allowed = {"mcp__echo"}
        self.assertIsNone(rewrite_tool_call_for_openai(
            {"id": "1", "function": {"name": "", "arguments": "{}"}},
            allowed_originals=allowed,
        ))
        self.assertIsNone(rewrite_tool_call_for_openai(
            {"id": "1", "function": {"name": "shell", "arguments": "{}"}},
            allowed_originals=allowed,
        ))
        kept = rewrite_tool_call_for_openai(
            {"id": "1", "function": {"name": "mcp__echo", "arguments": "{}"}},
            allowed_originals=allowed,
        )
        assert kept is not None
        self.assertEqual(kept["function"]["name"], "mcp__echo")

    def test_tool_results_go_into_conversation_history(self) -> None:
        from upstream.cursor.chat.convert import (
            _TOOL_CONTINUE_PROMPT,
            split_prompt_and_history,
        )
        from upstream.cursor.stream.worker import _build_run_request

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mcp__Glob",
                        "arguments": '{"glob_pattern":"*.py"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "mcp__Glob",
                "content": "a.py\nb.py",
            },
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, _TOOL_CONTINUE_PROMPT)
        self.assertNotIn("<tool_result", prompt)
        self.assertNotIn("a.py\nb.py", prompt)
        self.assertEqual(len(history), 3)
        self.assertIn("user", history[0])
        self.assertEqual(
            history[1]["assistant"]["content"][0]["toolCall"]["toolName"],
            "mcp__Glob",
        )
        tool_msg = history[2]["tool"]
        self.assertEqual(tool_msg["toolCallId"], "call_1")
        self.assertEqual(tool_msg["toolName"], "mcp__Glob")
        self.assertEqual(tool_msg["content"][0]["text"]["text"], "a.py\nb.py")

        # 空结果也要进 history
        empty_msgs = messages[:-1] + [{
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "mcp__Glob",
            "content": "",
        }]
        _, hist2 = split_prompt_and_history(empty_msgs)
        self.assertEqual(hist2[-1]["tool"]["content"][0]["text"]["text"], "")

        # name 缺失时从 assistant.tool_calls 反查
        nameless = messages[:-1] + [{
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "only-id",
        }]
        p3, h3 = split_prompt_and_history(nameless)
        self.assertEqual(p3, _TOOL_CONTINUE_PROMPT)
        self.assertEqual(h3[-1]["tool"]["toolName"], "mcp__Glob")

        # 多工具：text 仍短，history 含全部 tool
        multi = [
            {"role": "user", "content": "do both"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "Read", "arguments": "{}"}},
                    {"id": "c2", "function": {"name": "Grep", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file-a"},
            {"role": "tool", "tool_call_id": "c2", "content": "hit-b"},
        ]
        pm, hm = split_prompt_and_history(multi)
        self.assertEqual(pm, _TOOL_CONTINUE_PROMPT)
        self.assertNotIn("file-a", pm)
        tools = [m["tool"] for m in hm if "tool" in m]
        self.assertEqual([t["toolName"] for t in tools], ["Read", "Grep"])
        self.assertEqual([t["content"][0]["text"]["text"] for t in tools], ["file-a", "hit-b"])

        payload = _build_run_request(
            prompt=prompt,
            model="composer-2.5-fast",
            conv_id="c1",
            msg_id="m1",
            group_id="g1",
            workspace="X:/ws",
            mcp_tools=[{"name": "mcp__Glob", "toolName": "mcp__Glob", "description": "", "inputSchemaJson": "{}"}],
            conversation_history=history,
        )
        run = payload["runRequest"]
        self.assertEqual(
            run["action"]["userMessageAction"]["conversationHistory"]["messages"],
            history,
        )
        self.assertEqual(run["action"]["userMessageAction"]["userMessage"]["text"], prompt)
        self.assertIn("mcpTools", run)
        self.assertNotIn("customSystemPrompt", run)

    def test_tool_filter_headers(self) -> None:
        from upstream.cursor.stream.exec.tool_filter import (
            HEADER_ALLOWED_TOOLS,
            HEADER_EXCLUDE_TOOLS,
            MCP_ONLY_ALLOWED_TOOLS,
            tool_filter_for_openai,
        )
        from upstream.cursor.stream.worker import _agent_headers

        allowed, exclude = tool_filter_for_openai(True)
        self.assertEqual(allowed, list(MCP_ONLY_ALLOWED_TOOLS))
        self.assertIsNone(exclude)
        headers = _agent_headers(
            "agentn.example",
            {"accessToken": "t"},
            "cli-test",
            "UTC",
            "s1",
            "r1",
            allowed_tools=allowed,
            exclude_tools=exclude,
        )
        self.assertIn((HEADER_ALLOWED_TOOLS, "mcp_tool_call"), headers)

        allowed2, exclude2 = tool_filter_for_openai(False)
        self.assertIsNone(allowed2)
        assert exclude2 is not None
        self.assertGreater(len(exclude2), 40)
        headers2 = _agent_headers(
            "agentn.example",
            {"accessToken": "t"},
            "cli-test",
            "UTC",
            "s1",
            "r1",
            allowed_tools=allowed2,
            exclude_tools=exclude2,
        )
        exclude_hdr = dict(headers2).get(HEADER_EXCLUDE_TOOLS, "")
        self.assertIn("shell_tool_call", exclude_hdr)
        self.assertIn("mcp_tool_call", exclude_hdr)

    def test_mcp_args_proto_unwrap(self) -> None:
        from upstream.cursor.stream.handlers import _mcp_args_to_json

        raw = {
            "path": {"stringValue": "/tmp/a"},
            "n": {"numberValue": 3},
            "flag": {"boolValue": True},
        }
        self.assertEqual(
            json.loads(_mcp_args_to_json(raw)),
            {"path": "/tmp/a", "n": 3, "flag": True},
        )
        plain = {"path": "/tmp/b", "nested": {"x": 1}}
        self.assertEqual(json.loads(_mcp_args_to_json(plain)), plain)

    def test_prepend_system_to_prompt(self) -> None:
        from upstream.cursor.chat.convert import (
            IMPORTANT_NO_TOOLS,
            build_custom_system_prompt,
            prepend_system_to_prompt,
        )

        sys_text = build_custom_system_prompt(
            [{"role": "system", "content": "Be brief."}],
            None,
        )
        out = prepend_system_to_prompt(sys_text, "hello")
        self.assertTrue(out.startswith("<system>\n"))
        self.assertIn(IMPORTANT_NO_TOOLS, out)
        self.assertIn("Be brief.", out)
        self.assertTrue(out.endswith("hello"))
        self.assertNotIn("customSystemPrompt", out)

    def test_build_cursor_turn_keeps_user_query_primary(self) -> None:
        from upstream.cursor.chat.convert import build_cursor_turn
        from upstream.cursor.stream.worker import _build_run_request

        messages = [
            {"role": "system", "content": "You are Kimi Code CLI. " + ("x" * 2000)},
            {"role": "user", "content": "试试agentswam工具是否可用"},
        ]
        tools = [{"type": "function", "function": {"name": "Shell", "parameters": {}}}]
        send, hist, prepend = build_cursor_turn(messages, tools)
        # 对齐逆向：明文 UserMessage，不加 <user_query>；system 走 prepend
        self.assertEqual(send, "试试agentswam工具是否可用")
        self.assertNotIn("<user_query>", send)
        self.assertNotIn("You are Kimi Code CLI", send)
        self.assertEqual(hist, [])
        self.assertEqual(len(prepend), 1)
        self.assertIn("You are Kimi Code CLI", prepend[0]["text"])
        self.assertIn("tool list", prepend[0]["text"])
        self.assertNotIn("<system>", prepend[0]["text"])
        self.assertNotIn('names start with "mcp__"', prepend[0]["text"])

        payload = _build_run_request(
            prompt=send,
            model="composer-2.5-fast",
            conv_id="c1",
            msg_id="m1",
            group_id="g1",
            workspace="X:/ws",
            mcp_tools=None,
            conversation_history=hist or None,
            prepend_user_messages=prepend,
        )
        ua = payload["runRequest"]["action"]["userMessageAction"]
        self.assertEqual(ua["userMessage"]["text"], send)
        self.assertEqual(ua["prependUserMessages"], prepend)
        self.assertNotIn("conversationHistory", ua)
        self.assertNotIn("customSystemPrompt", payload["runRequest"])

    def test_whitespace_user_falls_back_to_prior(self) -> None:
        from upstream.cursor.chat.convert import build_cursor_turn, split_prompt_and_history

        messages = [
            {"role": "user", "content": "真实问题"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "   \n\t  "},
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, "真实问题")
        self.assertEqual(len(history), 1)
        self.assertIn("assistant", history[0])

        send, hist, prepend = build_cursor_turn(messages, None)
        self.assertEqual(send, "真实问题")
        self.assertTrue(prepend)

    def test_user_after_tool_keeps_results_in_history(self) -> None:
        from upstream.cursor.chat.convert import split_prompt_and_history

        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "mcp__echo", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "mcp__echo", "content": "pong"},
            {"role": "user", "content": "thanks"},
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, "thanks")
        self.assertEqual(history[-1]["tool"]["content"][0]["text"]["text"], "pong")


class TestCursorTokenHelpers(unittest.TestCase):
    def test_key_pool_rotates(self) -> None:
        pool = KeyPool(["k1", "k2"], threshold=80, refresh_interval=60)
        first = pool.current
        assert first is not None
        pool.switch_next()
        second = pool.current
        assert second is not None
        self.assertNotEqual(first.key, second.key)

    def test_should_switch_uses_daily_limit_percent(self) -> None:
        pool = KeyPool(["k1"], threshold=80, refresh_interval=60)
        s = pool.current
        assert s is not None
        s.daily_used = 90
        s.daily_limit = None
        self.assertFalse(pool.should_switch(s))
        s.daily_limit = 100
        self.assertTrue(pool.should_switch(s))
        s.daily_used = 79
        self.assertFalse(pool.should_switch(s))

    def test_single_key_without_limit_still_usable(self) -> None:
        from upstream.cursor.auth.token_service import CursorTokenService

        svc = CursorTokenService.__new__(CursorTokenService)
        svc._pool = KeyPool(["only-key"], threshold=80, refresh_interval=60)
        s = svc._pool.current
        assert s is not None
        s.daily_used = 999
        s.daily_limit = None
        self.assertIs(svc._fallback_active_key(), s)
        self.assertFalse(svc._pool.should_switch(s))

    def test_usage_limit(self) -> None:
        u = parse_usage({
            "individualUsage": {"plan": {
                "autoPercentUsed": 96,
                "apiPercentUsed": 94,
                "breakdown": {"total": 100},
            }},
        })
        self.assertTrue(is_limit_reached(u, threshold=95.0))


class TestCursorStreamErrors(unittest.TestCase):
    def test_rate_limit_maps_to_token_expired(self) -> None:
        from server.formats import TokenExpiredError
        from upstream.cursor.chat.openai import _raise_cursor_stream_error

        with self.assertRaises(TokenExpiredError):
            _raise_cursor_stream_error(
                "ERROR_RATE_LIMITED_CHANGEABLE: Get Cursor Pro for more Agent usage"
            )

    def test_other_error_maps_to_unavailable(self) -> None:
        from server.formats import UpstreamUnavailableError
        from upstream.cursor.chat.openai import _raise_cursor_stream_error

        with self.assertRaises(UpstreamUnavailableError):
            _raise_cursor_stream_error("boom internal")


class TestCursorClientStartup(unittest.IsolatedAsyncioTestCase):
    async def test_startup_without_keys_warns(self) -> None:
        from upstream.cursor.client import CursorClient

        with patch("upstream.cursor.client.token_service_config", return_value={"api_keys": [], "poll_interval": 30}):
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
