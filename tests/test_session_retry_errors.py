from __future__ import annotations

"""session_retry 层对 STS/ChatNotFound 的捕获与 proxy_toggle 翻转触发测试。"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from server.formats import UpstreamChatNotFoundError, UpstreamConnectionError, UpstreamStsError
from server.retry.session_retry import (
    _dispatch_session_retry,
    _trigger_proxy_toggle_on_block,
    is_retryable_error,
)


def _mock_state(shutting_down: bool = False):
    state = MagicMock()
    state.is_shutting_down = shutting_down
    return state


class TestStsRetryable(unittest.TestCase):
    def test_sts_error_is_retryable_via_connection_error(self) -> None:
        self.assertTrue(is_retryable_error(UpstreamStsError("All STS endpoints failed")))

    def test_sts_error_inherits_connection_error(self) -> None:
        err = UpstreamStsError("All STS endpoints failed")
        self.assertIsInstance(err, UpstreamConnectionError)
        self.assertEqual(err.status, 502)
        self.assertEqual(err.error_type, "upstream_sts_error")


class TestTriggerProxyToggleExpanded(unittest.IsolatedAsyncioTestCase):
    """验证 _trigger_proxy_toggle_on_block 新增类型也会调用 on_sm_block。"""

    async def _run_toggle(self, exc, *, enabled_in_manager: bool) -> None:
        mock_toggle = AsyncMock()
        mock_toggle.enabled = enabled_in_manager
        client = MagicMock()
        client._last_used_proxy_enabled = enabled_in_manager
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            await _trigger_proxy_toggle_on_block("req-x", exc, client)
        mock_toggle.on_sm_block.assert_awaited_once_with("req-x", enabled_in_manager)

    async def test_sts_error_triggers_toggle(self) -> None:
        await self._run_toggle(UpstreamStsError("All STS endpoints failed"), enabled_in_manager=True)

    async def test_generic_connection_error_triggers_toggle(self) -> None:
        await self._run_toggle(UpstreamConnectionError("upstream connect fail"), enabled_in_manager=False)

    async def test_chat_not_found_triggers_toggle(self) -> None:
        await self._run_toggle(UpstreamChatNotFoundError("Qwen chat not found"), enabled_in_manager=True)

    async def test_unrelated_type_skips_toggle(self) -> None:
        from server.formats import PayloadTooLargeError

        mock_toggle = AsyncMock()
        client = MagicMock()
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            await _trigger_proxy_toggle_on_block("req-x", PayloadTooLargeError("413"), client)
        mock_toggle.on_sm_block.assert_not_awaited()

    async def test_data_inspection_failed_skips_toggle(self) -> None:
        from server.formats import DataInspectionFailedError

        mock_toggle = AsyncMock()
        client = MagicMock()
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            await _trigger_proxy_toggle_on_block(
                "req-x",
                DataInspectionFailedError(
                    "内容安全警告：输入数据可能包含不适当的内容！",
                    code="data_inspection_failed",
                    stage="input",
                ),
                client,
            )
        mock_toggle.on_sm_block.assert_not_awaited()


class TestDataInspectionNotRetryable(unittest.TestCase):
    def test_data_inspection_failed_is_not_retryable(self) -> None:
        from server.formats import DataInspectionFailedError

        exc = DataInspectionFailedError("内容安全警告", code="data_inspection_failed")
        self.assertFalse(is_retryable_error(exc))


class TestDataInspectionErrorMapping(unittest.TestCase):
    def test_error_mapped_to_400_for_requester(self) -> None:
        from handlers.shared.api_errors import (
            classify_stream_error,
            handler_error_response,
        )
        from server.formats import DataInspectionFailedError

        exc = DataInspectionFailedError(
            "内容安全警告：输入数据可能包含不适当的内容！",
            code="data_inspection_failed",
            stage="input",
        )
        info = classify_stream_error(exc)
        assert info.kind == "invalid_request_error"
        assert info.code == 400

        resp = handler_error_response(exc, label="test")
        assert resp.status == 400
        body = json.loads(resp.body.decode("utf-8"))
        assert "内容安全警告" in body["error"]["message"]


class TestDataInspectionDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_raises_without_retry_or_toggle(self) -> None:
        from server.formats import DataInspectionFailedError

        mock_toggle = AsyncMock()
        state = _mock_state()
        client = MagicMock()
        exc = DataInspectionFailedError("内容安全警告", code="data_inspection_failed")
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            with self.assertRaises(DataInspectionFailedError):
                await _dispatch_session_retry(
                    "req-di", state, exc, retries=0, limit=3, client=client,
                )
        mock_toggle.on_sm_block.assert_not_awaited()
        client.switch_to_next.assert_not_called()


class TestDispatchSessionRetryStS(unittest.IsolatedAsyncioTestCase):
    """验证 STS 失败经 _dispatch_session_retry 走连接错误分支、先翻转 toggle 再重试。"""

    async def _dispatch(self, exc, *, retries: int = 0, limit: int = 3):
        mock_toggle = AsyncMock()
        state = _mock_state()
        client = MagicMock()
        client._last_used_proxy_enabled = True
        client.reset_http_transport = AsyncMock()
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            new_retries = await _dispatch_session_retry(
                "req-sts", state, exc, retries=retries, limit=limit, client=client,
            )
        return new_retries, mock_toggle, client

    async def test_sts_dispatch_flips_toggle_and_increments_retries(self) -> None:
        exc = UpstreamStsError("All STS endpoints failed")
        new_retries, mock_toggle, client = await self._dispatch(exc)
        self.assertEqual(new_retries, 1)
        mock_toggle.on_sm_block.assert_awaited_once_with("req-sts", True)
        client.reset_http_transport.assert_awaited_once()

    async def test_sts_dispatch_exhausted_raises_after_toggle(self) -> None:
        exc = UpstreamStsError("All STS endpoints failed")
        with self.assertRaises(UpstreamStsError):
            await self._dispatch(exc, retries=3, limit=3)


class TestChatNotFoundToggleBeforeSwitch(unittest.IsolatedAsyncioTestCase):
    """验证 CHAT_NOT_FOUND 路径在换号前先触发 proxy_toggle 翻转。"""

    async def _dispatch(self, *, retries: int = 0, limit: int = 3, has_switch: bool = True):
        mock_toggle = AsyncMock()
        state = _mock_state()
        client = MagicMock()
        client._last_used_proxy_enabled = True
        client.current_session_username = "acc_a@example.com"
        client.mark_invalid_current = MagicMock()
        if has_switch:
            new_session = MagicMock()
            new_session.username = "acc_b@example.com"
            client.switch_to_next = AsyncMock(return_value=new_session)
        else:
            client.switch_to_next = AsyncMock(return_value=None)
        exc = UpstreamChatNotFoundError("Qwen chat not found")
        with patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle
        ):
            new_retries = await _dispatch_session_retry(
                "req-cnf", state, exc, retries=retries, limit=limit, client=client,
            )
        return new_retries, mock_toggle, client

    async def test_chat_not_found_flips_toggle_before_switch(self) -> None:
        new_retries, mock_toggle, client = await self._dispatch()
        self.assertEqual(new_retries, 1)
        mock_toggle.on_sm_block.assert_awaited_once_with("req-cnf", True)
        # mark_invalid 必须在 switch 之前调用，保证不会租回同一失效会话
        client.mark_invalid_current.assert_called_once()
        client.switch_to_next.assert_awaited_once_with(exclude_username="acc_a@example.com")

    async def test_chat_not_found_exhausted_still_flips_then_raises(self) -> None:
        with self.assertRaises(UpstreamChatNotFoundError):
            await self._dispatch(retries=3, limit=3)


if __name__ == "__main__":
    unittest.main()
