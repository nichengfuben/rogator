#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回放 logs/responses 经 Rogator 非流式/流式解析路径，检查异常与 batch/stream 不一致。

用法:
  python scripts/replay_response_parse.py
  python scripts/replay_response_parse.py --responses-dir logs/responses
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
for path in (ROOT / "src", ROOT, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import path_setup  # noqa: F401

from echotools import get_protocol  # noqa: E402
from echotools.exec.fncall.protocols.entml_think.parse import split_entml_thinking  # noqa: E402
from handlers.openai import _parse_tool_calls  # noqa: E402
from rogator_entml_harness import (  # noqa: E402
    CHUNK_SIZES,
    batch_parse,
    merged_stream_json_per_invoke,
    simulate_anthropic_wire_json,
    simulate_openai_wire_json,
)
from state import AppState  # noqa: E402

TOOL_HEADER_RE = re.compile(
    r"^### ([A-Za-z][A-Za-z0-9]*)\s*\r?\n\r?\nDescription:\s*\r?\n",
    re.MULTILINE,
)
JSON_FENCE_RE = re.compile(r"```json\s*\r?\n(.*?)\r?\n```", re.DOTALL)
INVOKE_NAME_RE = re.compile(r'<entml:invoke(?:\s+name="([^"]+)"|\s+>\s*\r?\n\s*<entml:name>([^<]+)</entml:name>)')


@dataclass
class Issue:
    req_id: str
    severity: str  # error | warn
    category: str
    detail: str


@dataclass
class FileReport:
    req_id: str
    chars: int
    tool_count: int
    invoke_count: int
    batch_calls: int = 0
    issues: List[Issue] = field(default_factory=list)


def _make_app_state() -> AppState:
    state = AppState.__new__(AppState)
    state.protocol = get_protocol("entml")
    return state


def extract_tools_from_prompt(prompt_text: str) -> List[Dict[str, Any]]:
    """从 inject 后 prompt 提取 OpenAI tools（### Name + Description + json schema）。"""
    tools: List[Dict[str, Any]] = []
    headers = list(TOOL_HEADER_RE.finditer(prompt_text))
    for idx, match in enumerate(headers):
        name = match.group(1)
        start = match.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(prompt_text)
        section = prompt_text[start:end]
        json_match = JSON_FENCE_RE.search(section)
        if not json_match:
            continue
        try:
            schema = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(schema, dict):
            continue
        desc_match = re.match(r"(.*?)(?:\n\n#|\n\n```)", section, re.DOTALL)
        description = (desc_match.group(1).strip() if desc_match else "")[:2000]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        })
    return tools


def extract_invoke_names(text: str) -> List[str]:
    names: List[str] = []
    for m in INVOKE_NAME_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if name and name not in names:
            names.append(name)
    return names


def stub_tools(names: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@lru_cache(maxsize=64)
def _load_prompt_tools(prompt_path: str) -> Tuple[List[Dict[str, Any]], int]:
    text = Path(prompt_path).read_text(encoding="utf-8")
    return extract_tools_from_prompt(text), len(text)


def resolve_tools(req_id: str, text: str, prompts_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    prompt_path = prompts_dir / f"{req_id}.txt"
    if prompt_path.is_file():
        prompt_tools, _ = _load_prompt_tools(str(prompt_path))
        if prompt_tools:
            return prompt_tools, "prompt"
    invoke_names = extract_invoke_names(text)
    if invoke_names:
        return stub_tools(invoke_names), "invoke_stub"
    return [], "none"


def check_incomplete_markup(text: str) -> List[str]:
    problems: List[str] = []
    if "<entml:invoke" in text and "</entml:invoke>" not in text:
        problems.append("未闭合的 <entml:invoke>")
    if text.count("<entml:thinking>") != text.count("</entml:thinking>"):
        problems.append("thinking 标签未配对")
    if text.count("<entml:parameter") > text.count("</entml:parameter>"):
        problems.append("parameter 标签未配对")
    return problems


def run_non_stream(state: AppState, text: str, tools: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], str, str]:
    clean, calls = _parse_tool_calls(state, text, tools)
    display, thinking = split_entml_thinking(clean)
    return clean, calls, display, thinking


def compare_stream_to_batch(
    req_id: str,
    text: str,
    tools: List[Dict[str, Any]],
    batch_calls: List[Dict[str, Any]],
) -> List[Issue]:
    issues: List[Issue] = []
    if not tools:
        return issues

    batch_args = [c["function"]["arguments"] for c in batch_calls]

    for chunk in CHUNK_SIZES:
        try:
            merged = merged_stream_json_per_invoke(text, tools, chunk)
        except Exception as exc:
            issues.append(Issue(req_id, "error", "stream_merged", f"chunk={chunk}: {exc}"))
            continue
        if len(merged) != len(batch_args):
            issues.append(Issue(
                req_id, "error", "stream_batch_count",
                f"chunk={chunk}: stream={len(merged)} batch={len(batch_args)}",
            ))
            continue
        for i, (stream_raw, batch_raw) in enumerate(zip(merged, batch_args)):
            try:
                stream_obj = json.loads(stream_raw)
                batch_obj = json.loads(batch_raw)
            except json.JSONDecodeError as exc:
                issues.append(Issue(
                    req_id, "error", "json_decode",
                    f"chunk={chunk} invoke#{i}: {exc}",
                ))
                continue
            if stream_obj != batch_obj:
                issues.append(Issue(
                    req_id, "error", "stream_batch_args",
                    f"chunk={chunk} invoke#{i}: stream!=batch",
                ))

    for chunk in (1, 17, 64):
        try:
            ant_wire, ant_batch = simulate_anthropic_wire_json(text, tools, chunk)
            oai_by_idx, oai_batch = simulate_openai_wire_json(text, tools, chunk)
        except Exception as exc:
            issues.append(Issue(req_id, "error", "wire_sim", f"chunk={chunk}: {exc}"))
            continue
        if len(ant_wire) != len(batch_args):
            issues.append(Issue(
                req_id, "warn", "anthropic_wire_count",
                f"chunk={chunk}: wire={len(ant_wire)} batch={len(batch_args)}",
            ))
        if len(oai_by_idx) != len(batch_args):
            issues.append(Issue(
                req_id, "warn", "openai_wire_count",
                f"chunk={chunk}: wire={len(oai_by_idx)} batch={len(batch_args)}",
            ))

    return issues


def analyze_file(path: Path, prompts_dir: Path, state: AppState) -> FileReport:
    req_id = path.stem
    text = path.read_text(encoding="utf-8")
    tools, tool_source = resolve_tools(req_id, text, prompts_dir)
    invoke_names = extract_invoke_names(text)
    report = FileReport(
        req_id=req_id,
        chars=len(text),
        tool_count=len(tools),
        invoke_count=len(invoke_names),
    )

    if not text.strip():
        report.issues.append(Issue(req_id, "warn", "empty", "响应文件为空"))
        return report

    for problem in check_incomplete_markup(text):
        report.issues.append(Issue(req_id, "warn", "incomplete_markup", problem))

    if invoke_names and not tools:
        report.issues.append(Issue(req_id, "error", "no_tools", "有 invoke 但无法解析 tools"))

    try:
        clean, batch_calls, display, thinking = run_non_stream(state, text, tools)
        report.batch_calls = len(batch_calls)
    except Exception as exc:
        report.issues.append(Issue(
            req_id, "error", "non_stream",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ))
        return report

    for i, call in enumerate(batch_calls):
        args = call.get("function", {}).get("arguments", "")
        if not args or args.strip() in ("{}", "[]"):
            report.issues.append(Issue(req_id, "warn", "empty_args", f"invoke#{i} arguments 为空或 {{}}"))
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as exc:
            report.issues.append(Issue(req_id, "error", "batch_json", f"invoke#{i}: {exc}"))

    if "<entml:invoke" in text and len(batch_calls) < len(invoke_names):
        report.issues.append(Issue(
            req_id, "error", "missing_calls",
            f"invoke 标签 {len(invoke_names)} 个，batch 解析 {len(batch_calls)} 个",
        ))

    leak_markers = ("<entml:invoke", "<entml:parameter", "</entml:invoke>")
    for marker in leak_markers:
        if marker in display:
            report.issues.append(Issue(req_id, "error", "entml_leak", f"可见文本含 {marker!r}"))

    report.issues.extend(compare_stream_to_batch(req_id, text, tools, batch_calls))
    if tool_source == "invoke_stub" and tools:
        report.issues.append(Issue(
            req_id, "warn", "tool_source",
            f"未找到 prompt，使用 invoke 名 stub tools（{len(tools)} 个）",
        ))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="回放 logs/responses 解析测试")
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=ROOT / "logs" / "responses",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=ROOT / "logs" / "prompts",
    )
    args = parser.parse_args()

    if not args.responses_dir.is_dir():
        print(f"目录不存在: {args.responses_dir}", file=sys.stderr)
        return 2

    files = sorted(args.responses_dir.glob("*.txt"))
    if not files:
        print(f"无响应文件: {args.responses_dir}")
        return 0

    state = _make_app_state()
    reports = [analyze_file(path, args.prompts_dir, state) for path in files]

    errors = [i for r in reports for i in r.issues if i.severity == "error"]
    warns = [i for r in reports for i in r.issues if i.severity == "warn"]

    print(f"扫描 {len(files)} 个响应文件 @ {args.responses_dir}")
    print("-" * 72)
    for r in reports:
        status = "OK"
        if any(i.severity == "error" for i in r.issues):
            status = "FAIL"
        elif r.issues:
            status = "WARN"
        print(
            f"[{status}] {r.req_id}  chars={r.chars}  tools={r.tool_count}  "
            f"invokes={r.invoke_count}  batch_calls={r.batch_calls}  issues={len(r.issues)}",
        )
        for issue in r.issues:
            print(f"    [{issue.severity.upper()}][{issue.category}] {issue.detail}")

    print("-" * 72)
    print(f"合计: {len(errors)} error, {len(warns)} warn, {len(files)} files")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
