from __future__ import annotations

"""Qwen 上报对外入口（chat / upload / 公共辅助）。"""

from upstream.qwen.auth.report.chat import (
    report_after_chat_created,
    report_chat_generation,
    report_clk_generate_mode,
    report_completions_request_id,
    report_create_chat_sequence,
    report_generation_create_return,
    report_page_view,
    report_streaming_statistics,
    report_user_status,
)
from upstream.qwen.auth.report.core import (
    aem_page_id as _aem_page_id,
    base_typarms as _base_typarms,
    silent_request as _silent_request,
    spm_for_path as _spm_for_path,
)
from upstream.qwen.auth.report.upload import (
    report_compare_log_arrival,
    report_file_parse_success,
    report_file_upload_all_time,
    report_file_upload_finish,
    report_file_upload_oss_token_time,
    report_file_upload_start,
)

__all__ = [
    "_aem_page_id",
    "_base_typarms",
    "_silent_request",
    "_spm_for_path",
    "report_after_chat_created",
    "report_chat_generation",
    "report_clk_generate_mode",
    "report_compare_log_arrival",
    "report_completions_request_id",
    "report_create_chat_sequence",
    "report_file_parse_success",
    "report_file_upload_all_time",
    "report_file_upload_finish",
    "report_file_upload_oss_token_time",
    "report_file_upload_start",
    "report_generation_create_return",
    "report_page_view",
    "report_streaming_statistics",
    "report_user_status",
]
