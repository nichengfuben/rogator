#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按分支落盘 / 实网验证 Rogator prompt。

超限：全文尾部 max_chars → send_text，前缀 → attachment。
真实 prompt 文件按原文截断（不再二次 inject）。

用法：
  python scripts/preview_prompt_cases.py --case 7_real_prompt_file --max-chars 131072
  python scripts/preview_prompt_cases.py --live --prompt-file logs/prompts/req-xxx.txt --max-chars 131072
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
import path_setup  # noqa: F401

from echotools import get_protocol

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from handlers.openai.protocol import _build_protocol_options
from server.config import CONFIG
from state import LongTextSplitter

Branch = Literal[
    "under_limit",
    "over_limit_tools_upload_ok",
    "over_limit_no_tools_upload_ok",
    "over_limit_upload_fail",
]

DEFAULT_DUMP_DIR = ROOT / "scripts" / "prompt_case_dumps"
DEFAULT_PROMPT = ROOT / "logs" / "prompts" / "req-1785757818-d8255a751e4d.txt"


# 对话里只放短填充；超限靠 inject 后再 pad，避免 "H"*max_chars 直接 MemoryError
_MAX_MSG_FILLER = 8_192
_PAD_CHUNK = 256_000


@dataclass
class PromptCase:
    name: str
    description: str
    history_chars: int
    tools_count: int
    upload_ok: bool
    include_tools: bool
    prompt_file: Optional[Path] = None
    # inject 后若不足则补齐到该长度（用于超限分支，可为 None）
    min_full_chars: Optional[int] = None


def _preview_state(max_chars: int, *, send_full_prompt: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        protocol=get_protocol("entml"),
        splitter=LongTextSplitter(
            max_chars=max_chars,
            send_full_prompt=send_full_prompt,
        ),
    )


def _pad_to_length(text: str, target: int) -> str:
    """按块追加，避免一次构造超大字面量。"""
    need = target - len(text)
    if need <= 0:
        return text
    parts = [text]
    while need > 0:
        n = min(need, _PAD_CHUNK)
        parts.append("P" * n)
        need -= n
    return "".join(parts)


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
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
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
                        f"p_{j}": {
                            "type": "string",
                            "description": "参数 " + "y" * 60,
                        }
                        for j in range(4)
                    },
                    "required": ["p_0"],
                },
            },
        }
        for i in range(len(base), count)
    ]


def demo_messages(history_chars: int) -> List[Dict[str, Any]]:
    n = max(0, min(int(history_chars), _MAX_MSG_FILLER))
    filler = ("H" * n) if n else ""
    return [
        {"role": "system", "content": "你是编程助手，回答简洁。"},
        {"role": "user", "content": "帮我查一下北京天气"},
        {
            "role": "assistant",
            "reasoning": "应先调用 get_weather。",
            "content": "我来查北京天气。",
            "tool_calls": [
                {
                    "id": "call_weather_001",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "北京"}, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_001",
            "content": json.dumps(
                {"city": "北京", "temp": 28, "condition": "晴"},
                ensure_ascii=False,
            ),
        },
        {"role": "assistant", "content": "北京 28°C，晴。"},
        {
            "role": "user",
            "content": (
                f"历史上下文填充段（目标 {history_chars} chars，消息内 {n}）：\n{filler}"
                if filler
                else "短历史。"
            ),
        },
        {"role": "assistant", "content": "收到，已阅读上述上下文。"},
        {"role": "user", "content": "【当前轮】请总结并继续执行下一步。"},
    ]


def resolve_branch(
    *,
    filename: Optional[str],
    file_bytes: Optional[bytes],
    upload_ok: bool,
    has_tools: bool,
) -> Branch:
    if not filename or not file_bytes:
        return "under_limit"
    if not upload_ok:
        return "over_limit_upload_fail"
    if has_tools:
        return "over_limit_tools_upload_ok"
    return "over_limit_no_tools_upload_ok"


def prepare_case(
    case: PromptCase,
    *,
    max_chars: int,
    model: str,
    thinking_level: str,
    send_full_prompt: bool,
) -> Dict[str, Any]:
    """对齐 _prepare_stream：合成 case 走 inject；真实 prompt 文件视为已 inject 全文只做 split。"""
    from server.model.model_thinking import resolve_thinking_route
    from handlers.openai.protocol import protocol_thinking_level

    state = _preview_state(max_chars, send_full_prompt=send_full_prompt)
    protocol_options = _build_protocol_options({"thinking_level": thinking_level})
    route = resolve_thinking_route(model, protocol_thinking_level(protocol_options))

    if case.prompt_file is not None:
        # logs/prompts 落盘已是 inject 后全文，禁止再次 inject
        full_content = case.prompt_file.read_text(encoding="utf-8")
        tools: List[Dict[str, Any]] = (
            demo_tools(case.tools_count) if case.include_tools else []
        )
        injected: List[Dict[str, Any]] = [{"role": "user", "content": full_content}]
    else:
        messages = demo_messages(case.history_chars)
        tools = demo_tools(case.tools_count) if case.include_tools else []
        injected, full_content, route = prepare_injected_messages(
            state,
            messages,
            tools,
            req_id=f"preview-{case.name}",
            model=model,
            protocol_options=protocol_options,
            prompt_api="openai",
        )

    if case.min_full_chars is not None and len(full_content) < case.min_full_chars:
        full_content = _pad_to_length(full_content, int(case.min_full_chars))
        injected = [{**injected[0], "content": full_content}, *injected[1:]]

    _final, send_text, filename, file_bytes = apply_prompt_budget(
        state,
        injected,
        full_content,
        use_file_split=True,
    )
    branch = resolve_branch(
        filename=filename,
        file_bytes=file_bytes,
        upload_ok=case.upload_ok,
        has_tools=bool(tools),
    )
    attachment = ""
    if branch != "over_limit_upload_fail" and file_bytes:
        attachment = file_bytes.decode("utf-8")
    effective_filename = filename if attachment else None
    return {
        "case": case.name,
        "description": case.description,
        "branch": branch,
        "model": model,
        "max_chars": max_chars,
        "thinking_level": thinking_level,
        "send_full_prompt": send_full_prompt,
        "tools_count": len(tools),
        "full_chars": len(full_content),
        "send_chars": len(send_text),
        "attachment_chars": len(attachment),
        "filename": effective_filename,
        "upload_ok": case.upload_ok and bool(attachment),
        "source": "prompt_file" if case.prompt_file is not None else "synthetic",
        "route": {
            "use_entml": route.use_entml,
            "qwen_native_enabled": route.qwen_native_enabled,
            "qwen_native_mode": route.qwen_native_mode,
        },
        "full_content": full_content,
        "send_text": send_text,
        "attachment": attachment,
        "file_bytes": attachment.encode("utf-8") if attachment else None,
    }


def default_cases(max_chars: int) -> List[PromptCase]:
    over_full = max_chars + 50_000
    huge_full = max_chars + 200_000
    cases = [
        PromptCase(
            "1_under_limit_short",
            "短对话 + tools，未超限",
            2_000,
            2,
            True,
            True,
        ),
        PromptCase(
            "2_under_limit_no_tools",
            "短对话、无 tools，未超限",
            2_000,
            0,
            True,
            False,
        ),
        PromptCase(
            "3_over_limit_tools_upload_ok",
            f"inject 后 pad→{over_full:,} + tools，上传成功",
            2_000,
            5,
            True,
            True,
            min_full_chars=over_full,
        ),
        PromptCase(
            "4_over_limit_no_tools_upload_ok",
            f"inject 后 pad→{over_full:,}、无 tools，上传成功",
            2_000,
            0,
            True,
            False,
            min_full_chars=over_full,
        ),
        PromptCase(
            "5_over_limit_upload_fail",
            f"inject 后 pad→{over_full:,} + tools，上传失败（仅截断）",
            2_000,
            5,
            False,
            True,
            min_full_chars=over_full,
        ),
        PromptCase(
            "6_over_limit_tools_upload_ok_huge",
            f"inject 后 pad→{huge_full:,} + 30 tools",
            2_000,
            30,
            True,
            True,
            min_full_chars=huge_full,
        ),
    ]
    if DEFAULT_PROMPT.is_file():
        cases.append(
            PromptCase(
                "7_real_prompt_file",
                f"真实 prompt 文件 {DEFAULT_PROMPT.name}",
                0,
                0,
                True,
                False,
                prompt_file=DEFAULT_PROMPT,
            )
        )
    return cases


def dump_prepared(prepared: Dict[str, Any], dump_dir: Path) -> Path:
    case_dir = dump_dir / str(prepared["case"])
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "full_content.txt").write_text(
        prepared["full_content"], encoding="utf-8"
    )
    (case_dir / "send_text.txt").write_text(prepared["send_text"], encoding="utf-8")
    (case_dir / "attachment.txt").write_text(
        prepared["attachment"], encoding="utf-8"
    )
    meta = {
        k: v
        for k, v in prepared.items()
        if k not in ("full_content", "send_text", "attachment", "file_bytes")
    }
    (case_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    boundary = case_dir / "boundary.txt"
    if boundary.is_file():
        boundary.unlink()
    return case_dir


def _sync_proxy(proxy: str) -> None:
    if not proxy:
        return
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


def _load_account(ext_index: int):
    import re

    from core.session.accounts import Account, accounts_for_upstream

    path = Path(
        os.environ.get(
            "QWEN_EXT_ACCOUNTS",
            str(ROOT / "config" / "upstream" / "qwen" / "ext_accounts.toml"),
        )
    )
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(
            r'\[\[accounts\]\]\s*\nusername\s*=\s*"([^"]+)"\s*\npassword\s*=\s*"([^"]+)"',
            text,
        )
        if 0 <= ext_index < len(blocks):
            user, pwd = blocks[ext_index]
            return Account(username=user, password=pwd)
    pool = accounts_for_upstream("qwen")
    if not pool:
        raise RuntimeError("无可用账号（ext_accounts / 主账号池为空）")
    return pool[0]


async def live_run_prepared(
    prepared: Dict[str, Any],
    *,
    model: str,
    proxy: str,
    ext_account: int,
    max_events: int,
) -> int:
    from upstream.qwen.client import QwenClient

    if proxy:
        _sync_proxy(proxy)
        print(f"proxy={proxy}", flush=True)

    max_chars = int(prepared["max_chars"])
    splitter = LongTextSplitter(max_chars=max_chars, send_full_prompt=False)
    client = QwenClient(splitter=splitter)
    await client._ensure_http_session()

    account = _load_account(ext_account)
    print(f"account={account.username[:6]}… model={model}", flush=True)
    t0 = time.time()
    session = await client._perform_login(account)
    if not session:
        print("login FAIL", flush=True)
        return 2
    print(f"login OK ({int((time.time() - t0) * 1000)}ms)", flush=True)

    files: List[Dict[str, Any]] = []
    filename = prepared.get("filename")
    file_bytes = prepared.get("file_bytes")
    if filename and file_bytes and prepared.get("upload_ok"):
        t1 = time.time()
        url, file_obj = await client.upload_file(session, file_bytes, filename)
        files.append(file_obj)
        parse_meta = (file_obj.get("file") or {}).get("meta", {}).get("parse_meta")
        print(
            f"upload OK ({int((time.time() - t1) * 1000)}ms) "
            f"file_class={file_obj.get('file_class')} "
            f"greenNet={file_obj.get('greenNet')!r} parse={parse_meta} "
            f"url={str(url)[:72]}",
            flush=True,
        )
    elif prepared["branch"] == "over_limit_upload_fail":
        print("skip upload（模拟失败，仅发截断正文）", flush=True)
    else:
        print("no attachment", flush=True)

    chat_id = await client.create_chat(session, model)
    print(f"chat_id={chat_id[:12]}…", flush=True)
    messages = [{"role": "user", "content": prepared["send_text"]}]
    route = prepared.get("route") or {}
    hits = 0
    first = ""
    kind = ""
    err = ""
    t2 = time.time()
    try:
        async for event in client.chat_completion(
            session,
            chat_id,
            messages,
            model,
            files,
            qwen_thinking_enabled=bool(route.get("qwen_native_enabled", True)),
            qwen_thinking_mode=str(route.get("qwen_native_mode") or "Thinking"),
        ):
            et = str(event.get("type") or "")
            if et == "error":
                err = str(event.get("message") or event)[:300]
                break
            if et in ("answer", "thinking", "thinking_summary"):
                hits += 1
                if not kind:
                    kind = et
                    first = str(event.get("content") or "")[:120]
                if hits >= max_events:
                    break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:320]
    elapsed = int((time.time() - t2) * 1000)
    try:
        await client.cleanup_chat(session, chat_id)
    except Exception:
        pass

    if err:
        print(f"[FAIL] case={prepared['case']} error={err!r} elapsed={elapsed}ms", flush=True)
        return 1
    if hits:
        print(
            f"[OK] case={prepared['case']} outcome={kind} "
            f"snippet={first!r} events={hits} elapsed={elapsed}ms",
            flush=True,
        )
        return 0
    print(f"[FAIL] case={prepared['case']} no_model_content elapsed={elapsed}ms", flush=True)
    return 1


def select_cases(
    all_cases: List[PromptCase],
    *,
    case_names: List[str],
    prompt_file: Optional[Path],
) -> List[PromptCase]:
    if prompt_file is not None:
        path = prompt_file.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [
            PromptCase(
                name="prompt_file",
                description=f"指定 prompt 文件 {path.name}",
                history_chars=0,
                tools_count=0,
                upload_ok=True,
                include_tools=False,
                prompt_file=path,
            )
        ]
    if not case_names:
        return all_cases
    wanted = set(case_names)
    selected = [c for c in all_cases if c.name in wanted]
    missing = wanted - {c.name for c in selected}
    if missing:
        raise SystemExit(f"未知 case: {', '.join(sorted(missing))}")
    return selected


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="预览 / 实网验证 Qwen prompt 分割与上传分支"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(CONFIG.qwen_send_max_chars),
        help=f"默认读配置 limits.qwen_send_max_chars（当前 {CONFIG.qwen_send_max_chars}）",
    )
    parser.add_argument("--model", default="qwen3-coder-plus")
    parser.add_argument("--thinking-level", default="medium")
    parser.add_argument(
        "--send-full-prompt",
        action="store_true",
        help="关闭分割（对齐 send_full_prompt=true）",
    )
    parser.add_argument("--dump-dir", type=str, default=str(DEFAULT_DUMP_DIR))
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="只跑指定 case（可重复）；默认全部",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="用真实 prompt 文件替代合成 case",
    )
    parser.add_argument("--list", action="store_true", help="列出 case 后退出")
    parser.add_argument(
        "--live",
        action="store_true",
        help="落盘后对选中 case 实网上传+completions",
    )
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--ext-account", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument(
        "--no-dump",
        action="store_true",
        help="不写磁盘，只打印摘要（--live 时仍用内存结果）",
    )
    args = parser.parse_args(argv)

    dump_dir = Path(args.dump_dir)
    all_cases = default_cases(args.max_chars)
    if args.list:
        for c in all_cases:
            print(f"{c.name}: {c.description}")
        return 0

    prompt_file = Path(args.prompt_file) if args.prompt_file.strip() else None
    cases = select_cases(all_cases, case_names=args.case, prompt_file=prompt_file)

    print(
        f"model={args.model}  max_chars={args.max_chars:,}  "
        f"thinking={args.thinking_level}  send_full={args.send_full_prompt}",
        flush=True,
    )
    if not args.no_dump:
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"dump_dir={dump_dir.resolve()}\n", flush=True)

    prepared_list: List[Dict[str, Any]] = []
    for case in cases:
        prepared = prepare_case(
            case,
            max_chars=args.max_chars,
            model=args.model,
            thinking_level=args.thinking_level,
            send_full_prompt=args.send_full_prompt,
        )
        prepared_list.append(prepared)
        loc = ""
        if not args.no_dump:
            case_dir = dump_prepared(prepared, dump_dir)
            loc = f"  -> {case_dir}"
        print(
            f"{prepared['case']}: {prepared['branch']}  "
            f"full={prepared['full_chars']:,}  "
            f"send={prepared['send_chars']:,}  "
            f"attachment={prepared['attachment_chars']:,}  "
            f"file={prepared['filename'] or '-'}{loc}",
            flush=True,
        )

    if not args.live:
        return 0

    print("\n--- live ---\n", flush=True)
    rc = 0
    for prepared in prepared_list:
        print(f"## {prepared['case']} ({prepared['branch']})", flush=True)
        one = asyncio.run(
            live_run_prepared(
                prepared,
                model=args.model,
                proxy=(args.proxy or "").strip(),
                ext_account=args.ext_account,
                max_events=args.max_events,
            )
        )
        rc = rc or one
        print(flush=True)
    return rc


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main())
