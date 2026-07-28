#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建完整 entml 提示词（与 handlers/openai.py inject 路径一致）。

用法（在 rogator 仓库根目录）:
  python scripts/dump_full_prompt.py
  python scripts/dump_full_prompt.py --thinking-level high
  python scripts/dump_full_prompt.py --format anthropic
  python scripts/dump_full_prompt.py --demo-loop-warning
  python scripts/dump_full_prompt.py --demo-markup-warning
  python scripts/dump_full_prompt.py --input request.json --out full_prompt.txt

校验项对齐 echotools 2.3.63+ 历史工具行 ``{Name: json}`` / 简单标量 ``{Name: value}``，
以及 ``<loop_warning>`` / ``<history_markup_warning>``（需对应 --demo-* 开关）。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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

MESSAGES: List[Dict[str, Any]] = [
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

_REPEAT_WEATHER_ASSISTANT: Dict[str, Any] = {
    "role": "assistant",
    "content": "再查一次天气确认。",
    "tool_calls": [{
        "id": "call_repeat",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city":"杭州","unit":"c"}',
        },
    }],
}


def _demo_anthropic_travel_body() -> Dict[str, Any]:
    body = demo_anthropic_body()
    body["system"] = "你是旅行助手，优先用工具查事实。"
    body["output_config"] = {"effort": "high"}
    return body


def _prepare_messages(
    *,
    demo_loop_warning: bool,
    demo_markup_warning: bool,
) -> List[Dict[str, Any]]:
    msgs = copy.deepcopy(MESSAGES)
    insert_at = len(msgs) - 1
    if demo_loop_warning:
        for i in range(3):
            tc = copy.deepcopy(_REPEAT_WEATHER_ASSISTANT)
            tc["tool_calls"][0]["id"] = f"call_repeat_{i}"
            msgs.insert(insert_at, {"role": "tool", "tool_call_id": tc["tool_calls"][0]["id"], "content": "晴"})
            msgs.insert(insert_at, tc)
    if demo_markup_warning:
        msgs.insert(
            insert_at,
            {
                "role": "assistant",
                "content": (
                    "（错误示例，勿模仿）\n"
                    "<tool>\n"
                    '{Read: {"path": "x.py"}}\n'
                    "</tool>\n"
                ),
            },
        )
    return msgs


def _history_slice(prompt: str) -> str:
    start = prompt.find("<entml:conversation_history>")
    end = prompt.find("</entml:conversation_history>")
    if start >= 0 and end >= 0:
        return prompt[start:end]
    return prompt


def run_checks(
    result: Dict[str, Any],
    *,
    expect_loop_warning: bool = False,
    expect_markup_warning: bool = False,
) -> Dict[str, bool]:
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
        "history_clarify_preamble": (
            "All tool invocations and their results shown here have already been executed"
            in prompt
        ),
        "current_user_message": "<current_user_message>" in prompt,
        "thinking_behavior_block": "<thinking_behavior>" in prompt,
        "bare_invoke_instruction": '<entml:invoke name="' in prompt,
        "no_function_calls_wrapper_instr": "<entml:function_calls>" not in prompt,
        "history_reasoning_blocks": history.count("<entml:thinking>") >= 2,
        "weather_call_brace_json": '{get_weather: {"city": "杭州", "unit": "c"}}' in history,
        "time_call_brace_scalar": "{get_time: 杭州}" in history,
        "search_call_brace_scalar": "{search_web: 杭州西湖 周边热门景点}" in history,
        "multi_tool_blocks": history.count("<tool>") >= 1,
        "parallel_tools_one_block": (
            "{get_time: 杭州}" in history and "{search_web: 杭州西湖 周边热门景点}" in history
        ),
        "important_invoke_reminder": (
            "IMPORTANT: Completed tool turns in conversation history" in prompt
        ),
        "tools_section": "get_weather" in prompt,
        "thinking_mode_tag": f"<entml:thinking_mode>{level}</entml:thinking_mode>" in prompt,
        "thinking_after_current_user": (
            prompt.find("<current_user_message>") >= 0
            and prompt.find("<entml:thinking_mode>") > prompt.find("<current_user_message>")
        ),
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

    checks["loop_warning_block"] = (
        "<loop_warning>" in prompt if expect_loop_warning else "<loop_warning>" not in prompt
    )
    checks["history_markup_warning_block"] = (
        "<history_markup_warning>" in prompt
        if expect_markup_warning
        else "<history_markup_warning>" not in prompt
    )

    if expect_loop_warning:
        checks["loop_warning_before_current_user"] = (
            prompt.find("<loop_warning>") < prompt.find("<current_user_message>")
        )
    if expect_markup_warning:
        checks["markup_warning_before_current_user"] = (
            prompt.find("<history_markup_warning>") < prompt.find("<current_user_message>")
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump full entml prompt")
    parser.add_argument("--thinking-level", default="high")
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "full_prompt_dump.txt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--demo-loop-warning",
        action="store_true",
        help="在历史末尾插入 3 次相同 tool_calls，触发 <loop_warning>",
    )
    parser.add_argument(
        "--demo-markup-warning",
        action="store_true",
        help="插入含块级 <tool> 的 assistant 消息，触发 <history_markup_warning>",
    )
    args = parser.parse_args()

    expect_loop = args.demo_loop_warning
    expect_markup = args.demo_markup_warning

    if args.input:
        data = load_request(args.input)
        if args.format == "anthropic" or (
            "messages" in data
            and any(isinstance(m.get("content"), list) for m in data.get("messages", []))
        ):
            result = build_from_anthropic(data)
            expect_loop = expect_markup = False
        else:
            result = build_prompt(
                data.get("messages", []),
                data.get("tools", []),
                data.get("model", args.model),
                thinking_level=args.thinking_level,
            )
            expect_loop = expect_markup = False
    elif args.format == "anthropic":
        result = build_from_anthropic(_demo_anthropic_travel_body())
        expect_loop = expect_markup = False
    else:
        messages = _prepare_messages(
            demo_loop_warning=args.demo_loop_warning,
            demo_markup_warning=args.demo_markup_warning,
        )
        result = build_prompt(messages, TOOLS, args.model, thinking_level=args.thinking_level)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result["prompt"], encoding="utf-8")
    checks = run_checks(
        result,
        expect_loop_warning=expect_loop,
        expect_markup_warning=expect_markup,
    )

    if args.json:
        print(json.dumps({**result, "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if all(checks.values()) else 1

    sep = "=" * 72
    print(sep)
    print("FULL PROMPT DUMP")
    print(f"thinking_level        : {result['thinking_level_request']}")
    print(f"user_system_prompt    : {result.get('user_system_prompt')!r}")
    print(f"prompt length         : {len(result['prompt'])} chars")
    if expect_loop:
        print("demo                  : loop_warning")
    if expect_markup:
        print("demo                  : history_markup_warning")
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
