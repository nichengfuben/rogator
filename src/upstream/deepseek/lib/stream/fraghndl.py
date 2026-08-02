from __future__ import annotations

"""DeepSeek 流式解析——fragment / 批量操作处理混入类。

从 ``streamparser.py`` 拆分而来，承载 ``_proc`` 及其各分支处理方法，
减少单文件行数并将 ``_proc`` 拆分为若干条职责单一的私有方法。
本模块只提供 Mixin，供 ``StreamParser`` 组合使用，不单独实例化。
"""

from typing import Any, Dict, List, Optional


class _FragmentHandlerMixin:
    """封装 ``_proc`` 系列方法的混入类，供 ``StreamParser`` 继承。

    依赖宿主类提供以下属性/方法：
    ``_search``、``_msg_id``、``_parent_id``、``_status``、``_should_continue``、
    ``_is_think``、``_think_started``、``_tok_usage``、``_first_frag``、
    ``_skip_first_response_frag``、``_content``、``_think``、``_inc``、
    ``_cur_event``、``_proc_cite``。
    """

    def _proc_close_event(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理 ``close`` 事件携带的数据（点击行为 / 自动续写标记）。

        Args:
            data: JSON 数据字典。

        Returns:
            需要立即返回给上层的结果字典，或 None（继续正常处理）。
        """
        click_behavior = data.get("click_behavior")
        auto_resume = data.get("auto_resume")
        if isinstance(click_behavior, str):
            self._close_click_behavior = click_behavior
        if isinstance(auto_resume, bool):
            self._close_auto_resume = auto_resume
        if self._close_click_behavior == "retry":
            self._should_continue = True
            self._cur_event = None
            return {
                "type": "status",
                "status": self._status,
                "needs_continue": True,
            }
        if self._close_auto_resume is not None and self._status != "FINISHED":
            self._should_continue = True
            self._cur_event = None
            return {
                "type": "status",
                "status": self._status,
                "needs_continue": True,
            }
        self._cur_event = None
        return None

    def _extract_message_ids(self, data: Dict[str, Any]) -> None:
        """从数据中提取响应/请求消息 ID。

        Args:
            data: JSON 数据字典。
        """
        if "response_message_id" in data:
            self._msg_id = data["response_message_id"]
        if "request_message_id" in data:
            self._parent_id = data["request_message_id"]

    def _proc_response_init(self, v: Any) -> Optional[Dict[str, Any]]:
        """处理初始化响应体（包含完整 response 对象）。

        Args:
            v: data 中的 ``v`` 字段。

        Returns:
            首个产生输出的 fragment 处理结果，或 None。
        """
        if not (isinstance(v, dict) and "response" in v):
            return None
        rd = v["response"]
        if "message_id" in rd:
            self._msg_id = rd["message_id"]
        if "status" in rd:
            self._status = rd["status"]
            if self._status in ("INCOMPLETE", "TIMEOUT"):
                self._should_continue = True
        if "accumulated_token_usage" in rd:
            self._tok_usage = rd["accumulated_token_usage"]
        for frag in rd.get("fragments", []):
            ft = frag.get("type")
            if ft == "SEARCH":
                self._extract_search(frag.get("results", []))
            else:
                # 处理 THINK 或 RESPONSE fragment，返回产生的 chunk
                res = self._handle_frag(frag)
                if res is not None:
                    return res
        return None

    def _proc_status(self, p: str, v: Any) -> Optional[Dict[str, Any]]:
        """处理 ``response/status`` 状态更新。

        Args:
            p: data 中的 ``p`` 字段。
            v: data 中的 ``v`` 字段。

        Returns:
            状态事件结果字典，或 None。
        """
        if not (p == "response/status" and v):
            return None
        self._status = str(v)
        if v == "FINISHED":
            self._should_continue = False
            if self._is_think:
                self._is_think = False
            return {"type": "status", "status": "FINISHED"}
        if v in ("INCOMPLETE", "TIMEOUT"):
            self._should_continue = True
            return {
                "type": "status",
                "status": str(v),
                "needs_continue": True,
            }
        return None

    def _proc_batch(self, p: str, o: Any, v: Any) -> Optional[Dict[str, Any]]:
        """处理 ``response`` + ``BATCH`` 批量操作。

        遍历所有操作，确保 references 等副作用操作不被跳过，
        同时将多个 content/thinking 增量合并为单个 chunk 返回。

        Args:
            p: data 中的 ``p`` 字段。
            o: data 中的 ``o`` 字段。
            v: data 中的 ``v`` 字段。

        Returns:
            合并后的增量结果字典，或 None。
        """
        if not (p == "response" and o == "BATCH" and isinstance(v, list)):
            return None
        merged_content: str = ""
        merged_thinking: str = ""
        has_content = False
        has_thinking = False
        needs_continue_result: Optional[Dict[str, Any]] = None
        for op in v:
            if not isinstance(op, dict):
                continue
            result = self._proc_batch_op(op)
            if result is None:
                continue
            rt = result.get("type")
            if rt == "content":
                merged_content += result.get("content", "")
                has_content = True
            elif rt == "thinking":
                merged_thinking += result.get("content", "")
                has_thinking = True
            elif rt == "status" and result.get("needs_continue"):
                needs_continue_result = result
        # 优先返回续写标记
        if needs_continue_result is not None:
            return needs_continue_result
        # 思考增量优先于正文增量
        if has_thinking and merged_thinking:
            return {"type": "thinking", "content": merged_thinking}
        if has_content and merged_content:
            return {"type": "content", "content": merged_content}
        return None

    def _proc_batch_op(self, op: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        op_p = op.get("p", "")
        op_v = op.get("v")
        if op_p == "accumulated_token_usage":
            self._tok_usage = op_v or 0
            return None
        if op_p == "quasi_status" and op_v in ("INCOMPLETE", "TIMEOUT"):
            self._status = str(op_v)
            self._should_continue = True
            return {
                "type": "status",
                "status": str(op_v),
                "needs_continue": True,
            }
        if op_p in ("fragments", "response/fragments") and op.get("o") == "APPEND":
            return self._proc_batch_fragments_append(op_v)
        if op_p == "content" and op.get("o") == "APPEND":
            content_str = str(op_v) if op_v else ""
            if content_str:
                return self._handle_chunk(content_str)
            return None
        if op_p == "references" and isinstance(op_v, list):
            self._extract_references(op_v)
        return None

    def _proc_batch_fragments_append(self, op_v: Any) -> Optional[Dict[str, Any]]:
        if not (isinstance(op_v, list) and op_v and isinstance(op_v[0], dict)):
            return None
        return self._handle_frag(op_v[0])

    def _proc_fragments_append(
        self, p: str, o: Any, v: Any
    ) -> Optional[Dict[str, Any]]:
        """处理 ``fragments`` / ``response/fragments`` + ``APPEND`` 操作。"""
        if o != "APPEND" or not isinstance(v, list) or not v:
            return None
        if p not in ("fragments", "response/fragments"):
            return None
        if isinstance(v[0], dict):
            return self._handle_frag(v[0])
        return None

    def _proc_content_delta(self, p: str, v: Any) -> Optional[Dict[str, Any]]:
        """处理正文增量主路径与搜索结果/思考结束标记。

        Args:
            p: data 中的 ``p`` 字段。
            v: data 中的 ``v`` 字段。

        Returns:
            内容增量处理结果字典，或 None。
        """
        # 内容增量（主路径）
        if p == "response/fragments/-1/content" and v is not None:
            return self._handle_chunk(str(v))

        # 搜索结果
        if p == "response/fragments/-1/results" and v:
            self._extract_search(v)

        # 思考结束标记
        if p == "response/fragments/-1/elapsed_secs":
            if self._is_think:
                self._is_think = False
        return None

    def _proc_fallback(self, data: Dict[str, Any], v: Any) -> Optional[Dict[str, Any]]:
        """兜底处理：裸值推送（无 ``p``/``o`` 字段的简单值更新）。

        仅处理标量值（str/int/float/bool），跳过 list/dict 等复合类型，
        避免将未识别的 BATCH 操作列表字符串化后泄漏到输出流。

        Args:
            data: JSON 数据字典。
            v: data 中的 ``v`` 字段。

        Returns:
            兜底 chunk 处理结果字典，或 None。
        """
        if (
            v is not None
            and not isinstance(v, (dict, list))
            and "p" not in data
            and "o" not in data
        ):
            sv = str(v)
            if sv and sv not in ("FINISHED", "INCOMPLETE"):
                return self._handle_chunk(sv)
        return None

    def _proc(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理解析后的 JSON 数据对象。

        Args:
            data: JSON 数据字典。

        Returns:
            处理结果字典或 None。
        """
        if self._cur_event == "close":
            result = self._proc_close_event(data)
            if result is not None:
                return result

        self._extract_message_ids(data)

        v = data.get("v")
        res = self._proc_response_init(v)
        if res is not None:
            return res

        p = data.get("p", "")
        o = data.get("o")

        res = self._proc_status(p, v)
        if res is not None:
            return res

        res = self._proc_batch(p, o, v)
        if res is not None:
            return res

        res = self._proc_fragments_append(p, o, v)
        if res is not None:
            return res

        res = self._proc_content_delta(p, v)
        if res is not None:
            return res

        return self._proc_fallback(data, v)
