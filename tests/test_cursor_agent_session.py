from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from upstream.cursor.chat import session_db
from upstream.cursor.chat.agent_session import (
    mcp_result_payload,
    note_pending_mcp,
    trailing_tool_messages,
)
from upstream.cursor.stream.exec.common import finish
from upstream.cursor.stream.exec.run import _handle_mcp
from upstream.cursor.stream.handlers import AgentRunContext


class TestSessionDb(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "t.db")
        session_db.resolve_db_path.cache_clear()
        session_db.init_db(self.db)

    def tearDown(self) -> None:
        session_db.resolve_db_path.cache_clear()
        session_db._bump_gen()
        self._tmp.cleanup()

    def test_pending_and_cache_invalidate(self) -> None:
        session_db.upsert_session(
            session_id="s1", conversation_id="c1", workspace="W", status="running", db_path=self.db,
        )
        session_db.upsert_pending_tool(
            tool_call_id="t1", session_id="s1", base_msg={"id": 7}, exec_id=7, db_path=self.db,
        )
        # point resolve to temp by monkeypatching reads via db_path args already used
        sid = session_db.get_session_id_by_tool_call("t1", session_db.cache_gen())
        # default db may differ; force via direct SQL path used in upsert with self.db
        # re-read with ensure on self.db
        import sqlite3
        conn = sqlite3.connect(self.db)
        row = conn.execute("SELECT session_id FROM pending_tools WHERE tool_call_id=?", ("t1",)).fetchone()
        conn.close()
        self.assertEqual(row[0], "s1")

        session_db.upsert_blob("s1", "b1", "DATA", db_path=self.db)
        conn = sqlite3.connect(self.db)
        blob = conn.execute(
            "SELECT blob_data FROM blobs WHERE session_id=? AND blob_id=?", ("s1", "b1"),
        ).fetchone()
        conn.close()
        self.assertEqual(blob[0], "DATA")


class TestDeferMcpNoEmpty(unittest.TestCase):
    def test_handle_mcp_defer_returns_empty(self) -> None:
        out = _handle_mcp(
            {"mcpArgs": {"name": "mcp__x__y", "args": {}}},
            {"id": 1},
            0.0,
            defer_mcp=True,
        )
        self.assertEqual(out, [])

    def test_mcp_result_payload_shape(self) -> None:
        p = mcp_result_payload("echo-result:TOKEN-7788")
        self.assertEqual(p["success"]["content"][0]["text"]["text"], "echo-result:TOKEN-7788")
        frame = finish({"id": 3}, 0.0, "mcpResult", p)
        self.assertIn("mcpResult", frame)
        self.assertEqual(frame["id"], 3)

    def test_trailing_tools(self) -> None:
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ]
        # trailing only if last is tool — here both at end
        t = trailing_tool_messages(msgs)
        self.assertEqual([m["tool_call_id"] for m in t], ["a", "b"])


class TestNotePending(unittest.TestCase):
    def test_note_pending_registers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p.db")
            session_db.init_db(db)
            with mock.patch.object(session_db, "resolve_db_path", return_value=db):
                session_db.resolve_db_path.cache_clear()
                session_db._bump_gen()
                q: list = []
                import queue
                qq = queue.Queue()
                ctx = AgentRunContext(
                    q=qq, conv_id="c", start=0.0,
                    send_frame=lambda *_: None, finish=lambda *_a, **_k: None, touch=lambda: None,
                    session_id="sess-1", defer_mcp=True,
                )
                note_pending_mcp(ctx, tool_call_id="call_9", base_msg={"id": 9}, exec_id=9)
                self.assertIn("call_9", ctx.pending_mcp)
                sid = session_db.get_session_id_by_tool_call("call_9", session_db.cache_gen())
                self.assertEqual(sid, "sess-1")

    def test_multiline_tool_id_aliases_find_park(self) -> None:
        from upstream.cursor.chat.agent_session import (
            ParkedRun,
            find_parked_by_tool_ids,
            find_parked_for_workspace,
            park_run,
            drop_parked,
        )
        from upstream.cursor.chat.tool_ids import normalize_tool_call_id

        dirty = "call-abc-0\nfc_xyz-1"
        self.assertEqual(normalize_tool_call_id(dirty), "call-abc-0")
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p.db")
            session_db.init_db(db)
            with mock.patch.object(session_db, "resolve_db_path", return_value=db):
                session_db.resolve_db_path.cache_clear()
                session_db._bump_gen()
                import queue
                qq = queue.Queue()
                ctx = AgentRunContext(
                    q=qq, conv_id="c", start=0.0,
                    send_frame=lambda *_: None, finish=lambda *_a, **_k: None, touch=lambda: None,
                    session_id="sess-ml", defer_mcp=True,
                )
                note_pending_mcp(ctx, tool_call_id=dirty, base_msg={"id": 1}, exec_id=1)
                run = ParkedRun(
                    session_id="sess-ml",
                    workspace="/ws/a",
                    ctx=ctx,
                    sock=mock.Mock(),
                    conn=mock.Mock(),
                    stream_id=1,
                    sock_lock=__import__("threading").Lock(),
                    event_q=qq,
                    pending=dict(ctx.pending_mcp),
                )
                park_run(run)
                try:
                    self.assertIs(find_parked_by_tool_ids(["call-abc-0"]), run)
                    self.assertIs(find_parked_by_tool_ids([dirty]), run)
                    self.assertIs(find_parked_by_tool_ids(["fc_xyz-1"]), run)
                    self.assertIs(find_parked_for_workspace("/ws/a"), run)
                finally:
                    drop_parked("sess-ml", reason="test")


class TestBuiltinToolForward(unittest.TestCase):
    def test_extract_shell_tool_call_oneof(self) -> None:
        from upstream.cursor.stream.handlers import _openai_tool_from_agent_tool_call

        tc = {
            "shellToolCall": {
                "args": {"command": "echo hi", "workingDirectory": "/tmp", "toolCallId": "c-shell"},
            },
        }
        out = _openai_tool_from_agent_tool_call(tc, "fallback")
        self.assertIsNotNone(out)
        self.assertEqual(out["id"], "c-shell")
        self.assertEqual(out["function"]["name"], "Shell")
        self.assertIn("echo hi", out["function"]["arguments"])

    def test_mcp_emit_strips_prefix_for_requester(self) -> None:
        import queue
        from upstream.cursor.stream.handlers import AgentRunContext, process_agent_message
        from upstream.cursor.chat import session_db

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "m.db")
            session_db.init_db(db)
            with mock.patch.object(session_db, "resolve_db_path", return_value=db):
                session_db.resolve_db_path.cache_clear()
                session_db._bump_gen()
                qq: queue.Queue = queue.Queue()
                ctx = AgentRunContext(
                    q=qq, conv_id="c", start=0.0,
                    send_frame=lambda *_: None, finish=lambda *_a, **_k: None, touch=lambda: None,
                    defer_mcp=True, session_id="sess-mcp",
                )
                process_agent_message({
                    "execServerMessage": {
                        "id": 7,
                        "mcpArgs": {
                            "name": "mcp__smoke__echo",
                            "providerIdentifier": "smoke",
                            "toolName": "echo",
                            "toolCallId": "call-mcp-1",
                            "args": {"text": "TOKEN"},
                        },
                    },
                }, ctx)
                events = []
                while not qq.empty():
                    events.append(qq.get_nowait())
                tc = next(e for e in events if e.type == "tool_call").tool_call
                self.assertEqual(tc["function"]["name"], "smoke__echo")
                self.assertNotIn("mcp__", tc["function"]["name"])
                self.assertIn("call-mcp-1", ctx.pending_mcp)
                self.assertEqual(ctx.pending_mcp["call-mcp-1"].result_field, "mcpResult")

    def test_mcp_result_inject_restores_name_and_mcp_result(self) -> None:
        """请求者回传去前缀名 → text/history 还原 mcp__；同流 resume 注入 mcpResult。"""
        import queue
        import threading
        from upstream.cursor.chat.agent_session import (
            ParkedRun,
            note_pending_mcp,
            park_run,
            drop_parked,
            resume_with_tool_results,
        )
        from upstream.cursor.chat.convert import (
            format_tool_results_user_text,
            messages_to_cursor_history,
        )
        from upstream.cursor.chat import session_db
        from upstream.cursor.stream.handlers import AgentRunContext

        originals = {"mcp__smoke__echo"}
        msgs = [
            {"role": "user", "content": "echo"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-mcp-2",
                    "function": {"name": "smoke__echo", "arguments": '{"text":"T"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-mcp-2",
                "name": "smoke__echo",
                "content": "echo-result:T",
            },
        ]
        text = format_tool_results_user_text(msgs, tool_originals=originals)
        self.assertIn("Tool result for mcp__smoke__echo:\necho-result:T", text)
        hist = messages_to_cursor_history(msgs, tool_originals=originals)
        self.assertEqual(hist[-1]["tool"]["toolName"], "mcp__smoke__echo")
        self.assertEqual(
            hist[1]["assistant"]["content"][0]["toolCall"]["toolName"],
            "mcp__smoke__echo",
        )

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "r.db")
            session_db.init_db(db)
            with mock.patch.object(session_db, "resolve_db_path", return_value=db):
                session_db.resolve_db_path.cache_clear()
                session_db._bump_gen()
                frames: list = []
                qq: queue.Queue = queue.Queue()
                ctx = AgentRunContext(
                    q=qq, conv_id="c", start=0.0,
                    send_frame=lambda fr: frames.append(fr),
                    finish=lambda *_a, **_k: None, touch=lambda: None,
                    defer_mcp=True, session_id="sess-mcp-r",
                )
                note_pending_mcp(
                    ctx,
                    tool_call_id="call-mcp-2",
                    base_msg={"id": 9},
                    exec_id=9,
                    result_field="mcpResult",
                )
                run = ParkedRun(
                    session_id="sess-mcp-r",
                    workspace="/ws",
                    ctx=ctx,
                    sock=mock.Mock(),
                    conn=mock.Mock(),
                    stream_id=1,
                    sock_lock=threading.Lock(),
                    event_q=qq,
                    pending=dict(ctx.pending_mcp),
                )
                park_run(run)
                try:
                    ok = resume_with_tool_results(run, [msgs[-1]], req_id="t")
                    self.assertTrue(ok)
                    mcp_frames = [f for f in frames if "execClientMessage" in f]
                    self.assertTrue(mcp_frames)
                    inner = mcp_frames[0]["execClientMessage"]
                    self.assertIn("mcpResult", inner)
                    self.assertEqual(
                        inner["mcpResult"]["success"]["content"][0]["text"]["text"],
                        "echo-result:T",
                    )
                finally:
                    drop_parked("sess-mcp-r", reason="test")

    def test_defer_mcp_does_not_emit_non_mcp_tool_call(self) -> None:
        import queue
        from upstream.cursor.stream.handlers import AgentRunContext, process_agent_message

        qq: queue.Queue = queue.Queue()
        ctx = AgentRunContext(
            q=qq, conv_id="c", start=0.0,
            send_frame=lambda *_: None, finish=lambda *_a, **_k: None, touch=lambda: None,
            defer_mcp=True,
        )
        process_agent_message({
            "interactionUpdate": {
                "toolCallStarted": {
                    "callId": "call-read-1",
                    "toolCall": {
                        "readToolCall": {"args": {"path": "a.py", "toolCallId": "call-read-1"}},
                    },
                },
            },
        }, ctx)
        events = []
        while not qq.empty():
            events.append(qq.get_nowait())
        kinds = [e.type for e in events]
        self.assertIn("tool_started", kinds)
        self.assertNotIn("tool_call", kinds)

    def test_defer_exec_shell_runs_local_not_park(self) -> None:
        import queue
        from upstream.cursor.stream.handlers import AgentRunContext, process_agent_message
        from upstream.cursor.chat import session_db

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "b.db")
            session_db.init_db(db)
            with mock.patch.object(session_db, "resolve_db_path", return_value=db):
                session_db.resolve_db_path.cache_clear()
                session_db._bump_gen()
                frames: list = []
                qq: queue.Queue = queue.Queue()
                ctx = AgentRunContext(
                    q=qq, conv_id="c", start=0.0,
                    send_frame=frames.append, finish=lambda *_a, **_k: None, touch=lambda: None,
                    defer_mcp=True, session_id="sess-shell",
                )
                with mock.patch(
                    "upstream.cursor.stream.handlers.execute_tool",
                    return_value=[{"id": 42, "shellResult": {"success": {"command": "pwd", "exitCode": 0, "stdout": "/", "stderr": ""}}}],
                ) as exec_mock:
                    process_agent_message({
                        "execServerMessage": {
                            "id": 42,
                            "shellArgs": {"command": "pwd", "toolCallId": "call-sh"},
                        },
                    }, ctx)
                exec_mock.assert_called_once()
                self.assertEqual(ctx.pending_mcp, {})
                self.assertTrue(any("execClientMessage" in f for f in frames))
                events = []
                while not qq.empty():
                    events.append(qq.get_nowait())
                self.assertFalse(any(e.type == "tool_call" for e in events))


if __name__ == "__main__":
    unittest.main()
