#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建完整 entml 提示词（与 handlers/openai.py inject 路径一致）。

用法（在 rogator 仓库根目录）:
  python scripts/dump_full_prompt.py
  python scripts/dump_full_prompt.py --thinking-level high
  python scripts/dump_full_prompt.py --format anthropic
  python scripts/dump_full_prompt.py --input request.json --out full_prompt.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echotools.exec.fncall.protocols.entml_think.core import (
    default_max_thinking_length_for_level,
)

from scripts.build_prompt_preview import (
    build_from_anthropic,
    build_prompt,
    demo_anthropic_body,
    load_request,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "description": "温度单位 c/f"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询城市当前时间",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "联网搜索",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

MESSAGES = [
    {"role": "system", "content": "你是旅行助手，优先用工具查事实。"},
    {"role": "user", "content": "帮我规划明天去杭州，先查天气。"},
    {
        "role": "assistant",
        "content": "我先查一下杭州天气。",
        "reasoning": "用户计划杭州行程，应先调用 get_weather 获取实时天气。",
        "tool_calls": [{
            "id": "call_w1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"杭州","unit":"c"}',
            },
        }],
    },
    {"role": "tool", "tool_call_id": "call_w1", "content": "晴，26°C，东南风 2 级，湿度 55%。"},
    {
        "role": "assistant",
        "content": "天气不错。再确认一下当地时间，并搜一下西湖周边热门景点。",
        "reasoning": "天气适宜出行，需并行查询本地时间与西湖热门景点。",
        "tool_calls": [
            {"id": "call_t1", "type": "function", "function": {"name": "get_time", "arguments": '{"city":"杭州"}'}},
            {"id": "call_s1", "type": "function", "function": {"name": "search_web", "arguments": '{"query":"杭州西湖 周边热门景点"}'}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_t1", "content": "2026-07-26 14:30 CST"},
    {"role": "tool", "tool_call_id": "call_s1", "content": "断桥残雪、雷峰塔、苏堤、灵隐寺。"},
    {
        "role": "assistant",
        "content": "杭州明天晴 26°C，本地时间下午。可去断桥、雷峰塔、苏堤。需要我再细化行程吗？",
        "reasoning": "综合天气、时间与景点列表，给出简要推荐并询问是否细化。",
    },
    {"role": "user", "content": "给出上午/下午各一个景点，并行调用工具获取一下时间"},
]


def _demo_anthropic_travel_body() -> Dict[str, Any]:
    body = demo_anthropic_body()
    body["system"] = "你是旅行助手，优先用工具查事实。"
    body["output_config"] = {"effort": "high"}
    return body


def _history_slice(prompt: str) -> str:
    start = prompt.find("<entml:conversation_history>")
    end = prompt.find("</entml:conversation_history>")
    if start >= 0 and end >= 0:
        return prompt[start:end]
    return prompt


def run_checks(result: Dict[str, Any]) -> Dict[str, bool]:
    prompt = result["prompt"]
    level = result["thinking_level_request"]
    history = _history_slice(prompt)
    user_sys = result.get("user_system_prompt") or ""
    current_block = ""
    if "<current_user_message>" in prompt:
        current_block = prompt.split("<current_user_message>", 1)[1].split(
            "</current_user_message>", 1
        )[0]

    checks: Dict[str, bool] = {
        "user_system_prompt_block": (
            "<user_system_prompt>" in prompt
            and user_sys in prompt
            and user_sys not in current_block
        ),
        "history_block": "<entml:conversation_history>" in prompt,
        "current_user_message": "<current_user_message>" in prompt,
        "thinking_behavior_block": "<thinking_behavior>" in prompt,
        "bare_invoke_instruction": '<entml:invoke name="' in prompt,
        "no_function_calls_wrapper_instr": "<entml:function_calls>" not in prompt,
        "weather_call_line": "[get_weather: 杭州 | c]" in history,
        "time_call_line": "[get_time: 杭州]" in history,
        "search_call_line": "[search_web: 杭州西湖 周边热门景点]" in history,
        "multi_tool_blocks": history.count("<tool>") >= 1,
        "important_invoke_reminder": (
            "IMPORTANT: Completed tool turns in conversation history" in prompt
        ),
        "tools_section": "get_weather" in prompt,
        "thinking_mode_tag": f"<entml:thinking_mode>{level}</entml:thinking_mode>" in prompt,
    }

    if level in ("low", "medium", "high", "xhigh", "max"):
        expected_max = default_max_thinking_length_for_level(level)
        checks["default_max_for_level"] = (
            expected_max is not None
            and f"<entml:max_thinking_length>{expected_max}</entml:max_thinking_length>"
            in prompt
        )
    elif level == "auto":
        checks["no_default_max_length"] = "<entml:max_thinking_length>" not in prompt
    elif level == "none":
        checks["no_thinking_mode_tag"] = "<entml:thinking_mode>" not in prompt

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump full entml prompt")
    parser.add_argument("--thinking-level", default="high")
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "full_prompt_dump.txt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.input:
        data = load_request(args.input)
        if args.format == "anthropic" or (
            "messages" in data
            and any(isinstance(m.get("content"), list) for m in data.get("messages", []))
        ):
            result = build_from_anthropic(data)
        else:
            result = build_prompt(
                data.get("messages", []),
                data.get("tools", []),
                data.get("model", args.model),
                thinking_level=args.thinking_level,
            )
    elif args.format == "anthropic":
        result = build_from_anthropic(_demo_anthropic_travel_body())
    else:
        result = build_prompt(MESSAGES, TOOLS, args.model, thinking_level=args.thinking_level)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result["prompt"], encoding="utf-8")
    checks = run_checks(result)

    if args.json:
        print(json.dumps({**result, "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if all(checks.values()) else 1

    sep = "=" * 72
    print(sep)
    print("FULL PROMPT DUMP")
    print(f"thinking_level        : {result['thinking_level_request']}")
    print(f"user_system_prompt    : {result.get('user_system_prompt')!r}")
    print(f"prompt length         : {len(result['prompt'])} chars")
    print(sep)
    print(result["prompt"])
    print(sep)
    print(f"wrote: {args.out}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(checks.values())
    print("RESULT:", "ALL PASS" if all_ok else "HAS FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
