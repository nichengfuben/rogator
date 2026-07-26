#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建并预览发往 Qwen 上游的 prompt（与 handlers/openai.py 同路径）。

包含：
  - 过去会话历史（user / assistant / tool）
  - 模型历史思考（reasoning / thinking → <entml:thinking>）
  - 工具定义（OpenAI tools 或 Anthropic tools）
  - 新用户输入

用法：
  python scripts/build_prompt_preview.py
  python scripts/build_prompt_preview.py --model qwen3.8-max-preview --thinking on
  python scripts/build_prompt_preview.py --input sample_request.json
  python scripts/build_prompt_preview.py --format anthropic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echotools.fncall import get_protocol, inject_fncall

from handlers import fold_system_into_user
from handlers.anthro import _normalize_anthropic_messages, _normalize_anthropic_tools
from handlers.openai import _build_protocol_options, convert_tools_to_openai
from server.formats import build_qwen_message
from server.message_history import embed_reasoning_in_messages
from server.model_thinking import resolve_qwen_thinking


def demo_openai_messages() -> List[Dict[str, Any]]:
    """示例：多轮历史 + reasoning + tool call/result + 新用户消息。"""
    return [
        {"role": "system", "content": "你是编程助手，回答简洁。"},
        {"role": "user", "content": "帮我查一下北京天气"},
        {
            "role": "assistant",
            "reasoning": "用户要查天气，应先调用 get_weather 工具获取实时数据。",
            "content": "我来查一下北京当前的天气。",
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
        {
            "role": "assistant",
            "reasoning": "工具返回 28°C 晴，可以直接总结给用户。",
            "content": "北京现在 28°C，晴。",
        },
        {"role": "user", "content": "那上海呢？顺便把 todo 里加一条：提醒我带伞。"},
    ]


def demo_openai_tools() -> List[Dict[str, Any]]:
    return [
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


def demo_anthropic_body() -> Dict[str, Any]:
    return {
        "model": "qwen3.7-max",
        "system": "你是编程助手，回答简洁。",
        "thinking": {"type": "enabled"},
        "tools": [
            {
                "name": "get_weather",
                "description": "查询指定城市当前天气",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "todo_write",
                "description": "写入待办",
                "input_schema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "帮我查一下北京天气"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "用户要查天气，应调用 get_weather。"},
                    {"type": "text", "text": "我来查一下。"},
                    {
                        "type": "tool_use",
                        "id": "toolu_weather01",
                        "name": "get_weather",
                        "input": {"city": "北京"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_weather01",
                    "content": '{"city":"北京","temp":28,"condition":"晴"}',
                }],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "收到 28°C 晴，可以回复用户。"},
                    {"type": "text", "text": "北京现在 28°C，晴。"},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "那上海呢？顺便 todo 加一条：提醒我带伞。"}],
            },
        ],
    }


def build_prompt(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str,
    thinking_mode: Optional[str] = "on",
) -> Dict[str, Any]:
    """与 openai_chat_handler 相同的 prompt 构建链路。"""
    body: Dict[str, Any] = {"thinking": thinking_mode} if thinking_mode else {}
    protocol_options = _build_protocol_options(body)
    qwen_enabled, qwen_mode, use_entml = resolve_qwen_thinking(
        model, (protocol_options or {}).get("thinking_mode"),
    )
    entml_options = protocol_options if use_entml else None

    prepared = embed_reasoning_in_messages(messages)
    prepared = fold_system_into_user(prepared)
    openai_tools = convert_tools_to_openai(tools)
    protocol = get_protocol("entml")
    injected = inject_fncall(
        prepared, openai_tools, protocol, lang="zh", protocol_options=entml_options,
    )
    prompt = injected[0]["content"]
    qwen_msg = build_qwen_message(
        prompt, model,
        thinking_enabled=qwen_enabled,
        thinking_mode=qwen_mode,
    )
    return {
        "model": model,
        "thinking_mode_request": (protocol_options or {}).get("thinking_mode", "off"),
        "use_entml_thinking": use_entml,
        "qwen_thinking_enabled": qwen_enabled,
        "qwen_thinking_mode": qwen_mode,
        "tool_count": len(openai_tools),
        "prompt": prompt,
        "qwen_feature_config": qwen_msg["feature_config"],
    }


def build_from_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    model = body.get("model", "qwen3.7-max")
    raw_messages = body.get("messages", [])
    system = body.get("system")
    tools = _normalize_anthropic_tools(body.get("tools") or [])
    messages = _normalize_anthropic_messages(raw_messages)
    if system:
        sys_text = system if isinstance(system, str) else json.dumps(system, ensure_ascii=False)
        messages = [{"role": "system", "content": sys_text}, *messages]
    protocol_options = _build_protocol_options(body)
    thinking = (protocol_options or {}).get("thinking_mode", "off")
    return build_prompt(messages, tools, model, thinking_mode=thinking)


def load_request(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON 根节点必须是 object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="预览 Qwen 上游 prompt 构建结果")
    parser.add_argument("--input", type=Path, help="请求 JSON 文件（OpenAI chat 或 Anthropic messages 格式）")
    parser.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--thinking", default="on", help="on|off|auto")
    parser.add_argument("--output", type=Path, help="写入完整 prompt 到文件")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非纯文本")
    args = parser.parse_args()

    if args.input:
        data = load_request(args.input)
        if args.format == "anthropic" or "messages" in data and any(
            isinstance(m.get("content"), list) for m in data.get("messages", [])
        ):
            result = build_from_anthropic(data)
        else:
            messages = data.get("messages", [])
            tools = data.get("tools", [])
            model = data.get("model", args.model)
            body_thinking = data.get("thinking")
            if body_thinking is not None:
                result = build_prompt(messages, tools, model, thinking_mode=None)
                # 用请求体自带的 thinking 解析
                opts = _build_protocol_options(data)
                mode = (opts or {}).get("thinking_mode", "off")
                result = build_prompt(messages, tools, model, thinking_mode=mode)
            else:
                result = build_prompt(messages, tools, model, thinking_mode=args.thinking)
    elif args.format == "anthropic":
        result = build_from_anthropic(demo_anthropic_body())
    else:
        result = build_prompt(
            demo_openai_messages(), demo_openai_tools(), args.model, args.thinking,
        )

    if args.output:
        args.output.write_text(result["prompt"], encoding="utf-8")
        print(f"prompt 已写入 {args.output} ({len(result['prompt'])} chars)")

    if args.json:
        out = dict(result)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    sep = "=" * 72
    print(sep)
    print("Prompt 构建预览")
    print(sep)
    print(f"model                 : {result['model']}")
    print(f"request thinking_mode : {result['thinking_mode_request']}")
    print(f"use_entml_thinking    : {result['use_entml_thinking']}")
    print(f"qwen_thinking_enabled : {result['qwen_thinking_enabled']}")
    print(f"qwen_thinking_mode    : {result['qwen_thinking_mode']}")
    print(f"tools                 : {result['tool_count']}")
    print(f"prompt length         : {len(result['prompt'])} chars")
    print(sep)
    print(result["prompt"])
    print(sep)
    print("Qwen feature_config:")
    print(json.dumps(result["qwen_feature_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
