#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按分支落盘 Rogator prompt（与 handlers/openai._prepare_stream 同逻辑）。

每个分支目录只写：
  attachment.txt  — 附件完整内容（无则空文件）
  send_text.txt   — 最终发往 Qwen 的完整 prompt

用法：
  python scripts/preview_prompt_cases.py
  python scripts/preview_prompt_cases.py --dump-dir ./prompt_case_dumps
  python scripts/preview_prompt_cases.py --max-chars 256000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echotools.fncall import get_protocol, inject_fncall

from handlers import extract_system_for_inject
from handlers.openai import (
    _build_protocol_options,
    _inject_protocol_options,
    convert_tools_to_openai,
    protocol_thinking_level,
)
from server.model_thinking import resolve_qwen_thinking
from state import LongTextSplitter

Branch = Literal[
    "under_limit",
    "over_limit_tools_upload_ok",
    "over_limit_no_tools_upload_ok",
    "over_limit_upload_fail",
]

DEFAULT_DUMP_DIR = ROOT / "scripts" / "prompt_case_dumps"


@dataclass
class PromptCase:
    name: str
    description: str
    history_chars: int
    tools_count: int
    upload_ok: bool
    include_tools: bool


def demo_tools(count: int = 2) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    base = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "todo_write",
                "description": "写入待办事项",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["content"],
                },
            },
        },
    ]
    if count <= len(base):
        return base[:count]
    return base + [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "示例工具 " + "x" * 120,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"p_{j}": {"type": "string", "description": "参数 " + "y" * 60}
                        for j in range(4)
                    },
                    "required": ["p_0"],
                },
            },
        }
        for i in range(len(base), count)
    ]


def demo_messages(history_chars: int) -> List[Dict[str, Any]]:
    filler = "H" * max(0, history_chars)
    return [
        {"role": "system", "content": "你是编程助手，回答简洁。"},
        {"role": "user", "content": "帮我查一下北京天气"},
        {
            "role": "assistant",
            "reasoning": "应先调用 get_weather。",
            "content": "我来查北京天气。",
            "tool_calls": [{
                "id": "call_weather_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "北京"}, ensure_ascii=False),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_001",
            "content": json.dumps({"city": "北京", "temp": 28, "condition": "晴"}, ensure_ascii=False),
        },
        {"role": "assistant", "content": "北京 28°C，晴。"},
        {"role": "user", "content": f"历史上下文填充段（{history_chars} chars）：\n{filler}" if filler else "短历史。"},
        {"role": "assistant", "content": "收到，已阅读上述上下文。"},
        {"role": "user", "content": "【当前轮】请总结并继续执行下一步。"},
    ]


def build_full_content(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    model: str,
) -> str:
    user_system_prompt, prepared = extract_system_for_inject(messages)
    openai_tools = convert_tools_to_openai(tools)
    protocol_options = _build_protocol_options({"thinking_level": "medium"})
    _, _, use_entml = resolve_qwen_thinking(model, protocol_thinking_level(protocol_options))
    inject_options = _inject_protocol_options(protocol_options, use_entml)
    injected = inject_fncall(
        prepared, openai_tools, get_protocol("entml"), lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=inject_options,
    )
    return injected[0]["content"]


def apply_prepare_stream_logic(
    full_content: str,
    *,
    max_chars: int,
    upload_ok: bool,
    has_tools: bool,
) -> Tuple[str, Branch, str]:
    """inject 后超限：尾部 max_chars → send；剩余前缀 → 附件；上传失败则附件为空。"""
    splitter = LongTextSplitter(max_chars=max_chars)
    send_text, filename, file_bytes = splitter.split(full_content)
    remaining = file_bytes.decode("utf-8") if file_bytes else ""

    if not filename or not file_bytes:
        return send_text, "under_limit", ""

    if not upload_ok:
        return send_text, "over_limit_upload_fail", ""

    if has_tools:
        return send_text, "over_limit_tools_upload_ok", remaining
    return send_text, "over_limit_no_tools_upload_ok", remaining


def default_cases(max_chars: int) -> List[PromptCase]:
    over = max_chars + 50_000
    huge = max_chars * 8
    return [
        PromptCase("1_under_limit_short", "短对话 + tools，未超限", 2_000, 2, True, True),
        PromptCase("2_under_limit_no_tools", "短对话、无 tools，未超限", 2_000, 0, True, False),
        PromptCase("3_over_limit_tools_upload_ok", f"历史 {over:,} + tools，上传成功", over, 5, True, True),
        PromptCase("4_over_limit_no_tools_upload_ok", f"历史 {over:,}、无 tools，上传成功", over, 0, True, False),
        PromptCase("5_over_limit_upload_fail", f"历史 {over:,} + tools，上传失败", over, 5, False, True),
        PromptCase("6_over_limit_tools_upload_ok_huge", f"历史 {huge:,} + 30 tools", huge, 30, True, True),
    ]


def dump_case(
    case: PromptCase,
    *,
    max_chars: int,
    model: str,
    dump_dir: Path,
) -> None:
    tools = demo_tools(case.tools_count) if case.include_tools else []
    full_content = build_full_content(demo_messages(case.history_chars), tools, model=model)
    send_text, branch, attachment = apply_prepare_stream_logic(
        full_content,
        max_chars=max_chars,
        upload_ok=case.upload_ok,
        has_tools=case.include_tools,
    )

    case_dir = dump_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "attachment.txt").write_text(attachment, encoding="utf-8")
    (case_dir / "send_text.txt").write_text(send_text, encoding="utf-8")

    print(
        f"{case.name}: {branch}  "
        f"send={len(send_text):,}  attachment={len(attachment):,}  "
        f"-> {case_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=256_000)
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--dump-dir", type=str, default=str(DEFAULT_DUMP_DIR))
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  max_chars={args.max_chars:,}")
    print(f"dump_dir={dump_dir.resolve()}\n")
    for case in default_cases(args.max_chars):
        dump_case(case, max_chars=args.max_chars, model=args.model, dump_dir=dump_dir)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
