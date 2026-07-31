from __future__ import annotations

from typing import Any, Dict, Tuple

from upstream.cursor.stream.exec.common import finish

# tool_type -> (result_field, payload)
_STUBS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "conversationSearchArgs": ("conversationSearchResult", {"results": [], "totalCount": 0}),
    "recordScreenArgs": ("recordScreenResult", {"failure": {"error": "Screen recording not supported in headless mode"}}),
    "computerUseArgs": ("computerUseResult", {"error": {"error": "Computer use not supported"}}),
    "writeShellStdinArgs": ("writeShellStdinResult", {"error": {"error": "No active shell session"}}),
    "mcpStateExecArgs": ("mcpStateExecResult", {"success": {}}),
    "smartModeClassifierArgs": ("smartModeClassifierResult", {"approved": True}),
    "shellAllowlistPrecheckArgs": ("shellAllowlistPrecheckResult", {"allowlisted": True}),
    "webFetchAllowlistPrecheckArgs": ("webFetchAllowlistPrecheckResult", {"allowlisted": True}),
    "mcpAllowlistPrecheckArgs": ("mcpAllowlistPrecheckResult", {"allowlisted": True}),
    "canvasDiagnosticsArgs": ("canvasDiagnosticsResult", {"success": {"path": ""}}),
    "backgroundShellSpawnArgs": ("backgroundShellSpawnResult", {"error": {"command": "", "workingDirectory": "", "error": "Background shell not supported"}}),
    "forceBackgroundShellArgs": ("forceBackgroundShellResult", {"error": {"error": "Background shell not supported"}}),
    "subagentArgs": ("subagentResult", {"error": {"error": "Subagents not supported"}}),
    "subagentAwaitArgs": ("subagentAwaitResult", {"notFound": {"agentId": ""}}),
    "forceBackgroundSubagentArgs": ("forceBackgroundSubagentResult", {"error": {"error": "Background subagents not supported"}}),
    "agentStoreConflictArgs": ("agentStoreConflictResult", {"success": {}}),
    "listMcpResourcesExecArgs": ("listMcpResourcesExecResult", {"resources": []}),
    "readMcpResourceExecArgs": ("readMcpResourceExecResult", {"notFound": {"uri": ""}}),
    "generateImageArgs": ("generateImageResult", {"error": {"error": "Image generation not supported"}}),
    "readTodosArgs": ("readTodosResult", {"success": {"todos": []}}),
    "updateTodosArgs": ("updateTodosResult", {"success": {}}),
    "sendFinalSummaryArgs": ("sendFinalSummaryResult", {"success": {}}),
    "communicateUpdateArgs": ("communicateUpdateResult", {"success": {}}),
    "reflectArgs": ("reflectResult", {"success": {}}),
    "taskArgs": ("taskResult", {"success": {}}),
    "sendMessageArgs": ("sendMessageResult", {"success": {}}),
    "sendToUserArgs": ("sendToUserResult", {"success": {}}),
    "reportBugArgs": ("reportBugResult", {"success": {}}),
    "reportBugfixResultsArgs": ("reportBugfixResultsResult", {"success": {}}),
    "setActiveBranchArgs": ("setActiveBranchResult", {"success": {}}),
    "startGrindPlanningArgs": ("startGrindPlanningResult", {"error": {"error": "Grind planning not supported"}}),
    "startGrindExecutionArgs": ("startGrindExecutionResult", {"error": {"error": "Grind execution not supported"}}),
    "recordCiInvestigationFindingsArgs": ("recordCiInvestigationFindingsResult", {"success": {}}),
    "aiAttributionArgs": ("aiAttributionResult", {"success": {}}),
    "applyAgentDiffArgs": ("applyAgentDiffResult", {"error": {"error": "Agent diff not supported"}}),
    "semSearchToolArgs": ("semSearchToolResult", {"success": {"results": []}}),
    "readLintsToolArgs": ("readLintsToolResult", {"success": {"lints": []}}),
    "replaceEnvArgs": ("replaceEnvResult", {"success": {}}),
    "getMcpToolsArgs": ("getMcpToolsResult", {"success": {"tools": []}}),
    "editPrLabelsArgs": ("editPrLabelsResult", {"error": {"error": "PR label editing not supported"}}),
    "updatePrCodeTourArgs": ("updatePrCodeTourResult", {"error": {"error": "PR code tour not supported"}}),
    "prManagementArgs": ("prManagementResult", {"error": {"error": "PR management not supported"}}),
    "connectScmArgs": ("connectScmResult", {"error": {"error": "SCM connection not supported"}}),
    "mcpAuthArgs": ("mcpAuthResult", {"success": {}}),
    "switchModeArgs": ("switchModeResult", {"success": {}}),
    "createPlanArgs": ("createPlanResult", {"success": {}}),
    "setupVmEnvironmentArgs": ("setupVmEnvironmentResult", {"success": {}}),
    "fetchCloudAgentDataArgs": ("fetchCloudAgentDataResult", {"error": {"error": "Cloud agent data not available"}}),
    "awaitArgs": ("awaitResult", {"taskStillRunning": {"taskId": ""}}),
}


def stub_tool(msg: dict, base: dict, start: float, tool: str) -> dict:
    field, payload = _STUBS.get(tool, ("shellResult", {"stdout": "", "stderr": f"Unsupported: {tool}", "exitCode": -1}))
    out_payload = dict(payload)
    if tool == "canvasDiagnosticsArgs":
        path = (msg.get("canvasDiagnosticsArgs") or {}).get("path", "")
        out_payload = {"success": {"path": path}}
    elif tool == "subagentAwaitArgs":
        agent_id = (msg.get("subagentAwaitArgs") or {}).get("agentId", "")
        out_payload = {"notFound": {"agentId": agent_id}}
    elif tool == "readMcpResourceExecArgs":
        uri = (msg.get("readMcpResourceExecArgs") or {}).get("uri", "")
        out_payload = {"notFound": {"uri": uri}}
    elif tool == "awaitArgs":
        task_id = (msg.get("awaitArgs") or {}).get("taskId", "")
        out_payload = {"taskStillRunning": {"taskId": task_id}}
    elif tool == "askQuestionArgs":
        field = "askQuestionResult"
        question = (msg.get("askQuestionArgs") or {}).get("question", "")
        out_payload = {"success": {"answer": f"Auto-answer for: {question}"}}
        return finish(base, start, field, out_payload)
    return finish(base, start, field, out_payload)


def is_stub_tool(tool: str) -> bool:
    return tool in _STUBS or tool == "askQuestionArgs"
