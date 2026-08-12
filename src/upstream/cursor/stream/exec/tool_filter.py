"""Cursor Agent builtin ToolCall oneof names + allow/exclude header helpers.

--allowed-tools / --exclude-tools 取值必须是 agent.v1.ToolCall 的 tool oneof 字段名；
对应上游 allow/exclude 工具列表 HTTP 头（逗号分隔）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# proto field names (snake_case)，与 exclude-tools.ts 校验集合一致
CURSOR_BUILTIN_TOOL_ONEOFS: Tuple[str, ...] = (
    "shell_tool_call",
    "delete_tool_call",
    "glob_tool_call",
    "grep_tool_call",
    "read_tool_call",
    "update_todos_tool_call",
    "read_todos_tool_call",
    "edit_tool_call",
    "ls_tool_call",
    "read_lints_tool_call",
    "mcp_tool_call",
    "sem_search_tool_call",
    "create_plan_tool_call",
    "web_search_tool_call",
    "task_tool_call",
    "list_mcp_resources_tool_call",
    "read_mcp_resource_tool_call",
    "apply_agent_diff_tool_call",
    "ask_question_tool_call",
    "fetch_tool_call",
    "switch_mode_tool_call",
    "generate_image_tool_call",
    "record_screen_tool_call",
    "computer_use_tool_call",
    "write_shell_stdin_tool_call",
    "reflect_tool_call",
    "setup_vm_environment_tool_call",
    "await_tool_call",
    "start_grind_execution_tool_call",
    "start_grind_planning_tool_call",
    "web_fetch_tool_call",
    "report_bugfix_results_tool_call",
    "ai_attribution_tool_call",
    "pr_management_tool_call",
    "mcp_auth_tool_call",
    "blame_by_file_path_tool_call",
    "get_mcp_tools_tool_call",
    "report_bug_tool_call",
    "set_active_branch_tool_call",
    "communicate_update_tool_call",
    "send_final_summary_tool_call",
    "update_pr_code_tour_tool_call",
    "replace_env_tool_call",
    "edit_pr_labels_tool_call",
    "record_ci_investigation_findings_tool_call",
    "send_message_tool_call",
    "fetch_cloud_agent_data_tool_call",
    "send_to_user_tool_call",
    "pi_read_tool_call",
    "pi_bash_tool_call",
    "pi_edit_tool_call",
    "pi_write_tool_call",
    "pi_grep_tool_call",
    "pi_find_tool_call",
    "pi_ls_tool_call",
    "connect_scm_tool_call",
    "search_conversations_tool_call",
    "truncated_tool_call",
)

MCP_ONLY_ALLOWED_TOOLS: Tuple[str, ...] = ("mcp_tool_call",)

HEADER_ALLOWED_TOOLS = "x-cursor-agent-allowed-tools"
HEADER_EXCLUDE_TOOLS = "x-cursor-agent-exclude-tools"


def tool_filter_for_openai(has_tools: bool) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """OpenAI 代理语义：有 tools 时仅放行 mcp_tool_call；无 tools 时排除全部 builtin。"""
    if has_tools:
        return list(MCP_ONLY_ALLOWED_TOOLS), None
    return None, list(CURSOR_BUILTIN_TOOL_ONEOFS)


def apply_tool_filter_headers(
    headers: List[Tuple[str, str]],
    *,
    allowed_tools: Optional[Sequence[str]] = None,
    exclude_tools: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    out = list(headers)
    if allowed_tools is not None:
        out.append((HEADER_ALLOWED_TOOLS, ",".join(allowed_tools)))
    if exclude_tools:
        out.append((HEADER_EXCLUDE_TOOLS, ",".join(exclude_tools)))
    return out
