#!/usr/bin/env python3
"""DeepSeek V4 Pro 空响应诊断脚本。

用法（项目根目录）::

    python scripts/diag_deepseek_v4_pro.py
    python scripts/diag_deepseek_v4_pro.py --try-model-type expert
    python scripts/diag_deepseek_v4_pro.py --max-sse-lines 80

检查链路：
  1. Rogator ``/v1/chat/completions``（OpenAI 兼容层）
  2. ``DeepSeekClient.complete`` 产出
  3. 上游原始 SSE + ``StreamParser`` 累积状态
  4. 可选：手动改 ``model_type`` 对比（default / expert / flash / vision）
  5. ``openai_chat._normalize_chunk`` 是否丢正文
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL = "deepseek-v4-pro"
PROMPT = "Reply with exactly: OK-DSV4PRO"
MESSAGES = [{"role": "user", "content": PROMPT}]


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _j(obj: Any, limit: int = 4000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) > limit:
        return text[:limit] + f"\n... ({len(text) - limit} chars truncated)"
    return text


def _splitter_stub() -> Any:
    return type("S", (), {"send_full_prompt": True, "max_chars": 999999})()


def _rogator_base() -> str:
    from server.config import CONFIG

    return f"http://127.0.0.1:{CONFIG.port}"


def step_rogator_api(model: str) -> Dict[str, Any]:
    _banner(f"[1] Rogator API  POST /v1/chat/completions  model={model}")
    payload = {
        "model": model,
        "stream": False,
        "messages": MESSAGES,
    }
    url = f"{_rogator_base()}/v1/chat/completions"
    print(f"URL: {url}")
    print(f"Body: {_j(payload, 800)}")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {err}")
        return {"ok": False, "http_status": exc.code, "error": err}
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = body.get("usage") or {}
    print(f"assistant.content: {content!r}")
    print(f"usage: {usage}")
    ok = bool(str(content).strip()) and int(usage.get("completion_tokens") or 0) > 0
    print(f"结论: {'有正文' if ok else '空响应（completion_tokens=0 或无 content）'}")
    return {"ok": ok, "body": body, "content": content, "usage": usage}


def _detect_biz_error(sse_lines: List[str]) -> Optional[str]:
    for line in sse_lines:
        raw = line.strip()
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, dict):
            continue
        biz_msg = data.get("biz_msg")
        biz_code = data.get("biz_code")
        if biz_msg or biz_code is not None:
            return f"biz_code={biz_code} biz_msg={biz_msg!r}"
    return None


async def _prepare_upstream_post(
    inner: Any,
    candidate: Any,
    model: str,
    *,
    model_type_override: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    from upstream.deepseek.lib.adapter.helpers.client_helpers import (
        acquire_hif_and_pow,
        build_post_kwargs,
        build_request_context,
        prepare_session,
    )
    from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST

    from upstream.deepseek.lib.protocol.payload import make_stream_id

    ctx = build_request_context(candidate, MESSAGES, model)
    if model_type_override is not None:
        ctx["model_type"] = model_type_override

    token = ctx["token"]
    username = ctx["username"]
    hif_leim, hif_dliq, pow_resp = await acquire_hif_and_pow(
        inner._hif_managers,  # noqa: SLF001
        inner._pow,  # noqa: SLF001
        inner._session,  # noqa: SLF001
        username,
        token,
    )
    session_id, req_headers = await prepare_session(
        inner._session, token, hif_leim, hif_dliq, pow_resp  # noqa: SLF001
    )
    payload = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": ctx["model_type"],
        "prompt": ctx["prompt"],
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
        "preempt": False,
        "client_stream_id": make_stream_id(),
    }
    post_kw = build_post_kwargs(
        req_headers,
        payload,
        inner._proxy_override,  # noqa: SLF001
        inner._get_proxy_kwarg,  # noqa: SLF001
    )
    url = f"https://{DEFAULT_HOST}/api/v0/chat/completion"
    meta = {
        "url": url,
        "payload": payload,
        "candidate": candidate.meta.get("identifier"),
        "prompt_preview": ctx["prompt"][:200],
    }
    return meta, post_kw, url


async def step_upstream_raw_sse(
    inner: Any,
    candidate: Any,
    model: str,
    *,
    model_type_override: Optional[str],
    max_sse_lines: int,
) -> Dict[str, Any]:
    label = model_type_override or "(代码默认 build_request_context)"
    _banner(f"[2] 上游原始 SSE  model={model}  model_type={label}")

    from upstream.deepseek.lib.stream.strmpars import StreamParser

    meta, post_kw, url = await _prepare_upstream_post(
        inner, candidate, model, model_type_override=model_type_override
    )
    print(f"candidate: {meta['candidate']}")
    print(f"payload.model_type: {meta['payload']['model_type']}")
    print(f"prompt 前 200 字: {meta['prompt_preview']!r}")
    print(f"POST {url}")

    sse_lines: List[str] = []
    parsed_chunks: List[Any] = []
    parser = StreamParser(include_thinking=True)
    parser.begin_stream(is_continuation=False)

    async with inner._session.post(url, **post_kw) as resp:  # noqa: SLF001
        print(f"HTTP status: {resp.status}")
        if resp.status != 200:
            text = await resp.text()
            print(f"body: {text[:1000]}")
            return {"ok": False, "http_status": resp.status, "body": text}

        buf = ""
        async for raw in resp.content.iter_chunked(4096):
            if not raw:
                continue
            buf += raw.decode("utf-8", errors="ignore")
            lines = buf.split("\n")
            buf = lines[-1]
            for line in lines[:-1]:
                if line.strip():
                    sse_lines.append(line)
                    if len(sse_lines) <= max_sse_lines:
                        print(f"  SSE: {line[:400]}")
                result = parser.parse_line(line)
                if result is not None:
                    parsed_chunks.append(result)
        if buf.strip():
            sse_lines.append(buf)
            if len(sse_lines) <= max_sse_lines:
                print(f"  SSE: {buf[:400]}")
            result = parser.parse_line(buf)
            if result is not None:
                parsed_chunks.append(result)

    print(f"\nSSE 行数: {len(sse_lines)}")
    print(f"解析 chunk 数: {len(parsed_chunks)}")
    content_chunks = [c for c in parsed_chunks if c.get("type") == "content"]
    think_chunks = [c for c in parsed_chunks if c.get("type") == "thinking"]
    status_chunks = [c for c in parsed_chunks if c.get("type") in ("status", "event")]
    print(
        f"content: {len(content_chunks)}  thinking: {len(think_chunks)}  "
        f"status/event: {len(status_chunks)}"
    )
    if content_chunks:
        sample = "".join(c.get("content", "") for c in content_chunks)
        print(f"content 拼接样例: {sample[:300]!r}")
    print(f"parser.status: {parser.status}")
    print(f"parser.message_id: {parser.message_id}")
    print(f"parser.should_continue: {parser.should_continue}")
    print(f"parser.accumulated_content: {parser.accumulated_content[:400]!r}")
    print(f"parser.accumulated_thinking: {parser.accumulated_thinking[:400]!r}")

    biz_err = _detect_biz_error(sse_lines)
    if biz_err:
        print(f"上游业务错误: {biz_err}")

    ok = bool(parser.accumulated_content.strip()) and not biz_err
    if biz_err:
        print("结论: 上游拒绝请求（非 parser/model_type 问题）")
    else:
        print(f"结论: {'上游有正文' if parser.accumulated_content.strip() else '上游 SSE 未解析出正文'}")
    return {
        "ok": ok,
        "http_status": 200,
        "model_type": meta["payload"]["model_type"],
        "sse_line_count": len(sse_lines),
        "parsed_chunks": parsed_chunks[:20],
        "accumulated_content": parser.accumulated_content,
        "parser_status": parser.status,
        "should_continue": parser.should_continue,
        "biz_error": biz_err,
    }


async def step_client_complete(inner: Any, candidate: Any, model: str) -> Dict[str, Any]:
    _banner(f"[3] DeepSeekClient.complete  model={model}")
    chunks: List[Any] = []
    async for chunk in inner.complete(
        candidate, MESSAGES, model, stream=True, thinking=False, search=False
    ):
        chunks.append(chunk)
    texts = [c for c in chunks if isinstance(c, str)]
    dicts = [c for c in chunks if isinstance(c, dict)]
    print(f"yield 总数: {len(chunks)}  str: {len(texts)}  dict: {len(dicts)}")
    for i, ch in enumerate(chunks[:15]):
        print(f"  [{i}] {ch!r}")
    joined = "".join(texts)
    usage = next((d for d in dicts if "usage" in d), None)
    print(f"拼接正文: {joined[:300]!r}")
    print(f"usage chunk: {usage}")
    ok = bool(joined.strip())
    print(f"结论: {'complete() 有正文' if ok else 'complete() 无 str 正文'}")
    return {"ok": ok, "chunks": chunks, "text": joined, "usage": usage}


def step_normalize_chunk() -> None:
    _banner("[4] openai_chat._normalize_chunk 行为")
    from upstream.deepseek.openai_chat import _normalize_chunk

    samples = [
        "hello",
        {"thinking": "plan"},
        {"usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}},
        {"type": "content", "content": "ignored-if-dict"},
        {"content": "also-ignored"},
    ]
    for s in samples:
        out = _normalize_chunk(s)
        print(f"  in={s!r}")
        print(f"  out={out!r}")
    print(
        textwrap.dedent(
            """
            注意: _normalize_chunk 仅处理 str / usage / thinking；
            dict 且无 thinking 键时返回 None —— 若 complete() 曾 yield dict 正文会被 Rogator 层丢弃。
            """
        ).strip()
    )


def step_build_request_context(model: str) -> None:
    _banner(f"[5] build_request_context 是否忽略 model 参数  model={model}")
    from upstream.deepseek.lib.adapter.helpers.client_helpers import build_request_context

    fake_candidate = type(
        "C",
        (),
        {"meta": {"token": "tok", "identifier": "user@example.com"}},
    )()
    ctx = build_request_context(fake_candidate, MESSAGES, model)
    print(_j(ctx))
    if ctx.get("model_type") == "default" and model == "deepseek-v4-pro":
        print(
            "WARN: model 参数未映射：deepseek-v4-pro 仍发送 model_type='default'。"
            "可用 --try-model-type expert 对比上游行为。"
        )


async def step_openai_chat_layer(client: Any, model: str) -> Dict[str, Any]:
    _banner(f"[6] upstream.deepseek.openai_chat 整链路  model={model}")
    from upstream.deepseek.openai_chat import stream_openai_chat

    class _State:
        protocol = __import__("echotools.exec.fncall", fromlist=["get_protocol"]).get_protocol("entml")
        splitter = _splitter_stub()

    events: List[Any] = []
    async for ev in stream_openai_chat(
        _State(),
        client,
        MESSAGES,
        model,
        tools=None,
        req_id="diag-dsv4pro",
        protocol_options={"thinking_mode": "off"},
    ):
        events.append(ev)
    answers = [e.get("content", "") for e in events if e.get("type") == "answer"]
    joined = "".join(answers)
    print(f"events: {len(events)}  answer events: {len(answers)}")
    for ev in events[:12]:
        print(f"  {ev}")
    print(f"answer 拼接: {joined[:300]!r}")
    ok = bool(joined.strip())
    print(f"结论: {'openai_chat 有可见 answer' if ok else 'openai_chat answer 为空'}")
    return {"ok": ok, "events": events, "text": joined}


async def step_try_all_candidates(
    inner: Any,
    model: str,
    *,
    max_sse_lines: int,
) -> None:
    _banner(f"[7] 逐账号探测上游  model={model}")
    candidates = list(inner._candidates)  # noqa: SLF001
    if not candidates:
        print("无 candidate")
        return
    for cand in candidates:
        name = cand.meta.get("identifier", cand.resource_id)
        print(f"\n--- candidate: {name} ---")
        res = await step_upstream_raw_sse(
            inner,
            cand,
            model,
            model_type_override="expert",
            max_sse_lines=max_sse_lines,
        )
        if res.get("ok"):
            print(f"账号 {name} 可用")
            return
    print("所有 candidate 均失败")


async def _get_inner_and_candidate() -> tuple[Any, Any]:
    from upstream.deepseek.client import DeepSeekClient

    client = DeepSeekClient(splitter=_splitter_stub())
    await client.startup()
    inner = await client._ensure_ready()  # noqa: SLF001
    candidate = await client.pick_candidate()
    if candidate is None:
        raise RuntimeError("无可用 DeepSeek candidate（检查 persist/deepseek 账号与会话）")
    return inner, candidate


async def _run(args: argparse.Namespace) -> int:
    print(f"项目根: {ROOT}")
    print(f"诊断模型: {args.model}")
    print(f"测试 prompt: {PROMPT}")

    step_build_request_context(args.model)
    step_normalize_chunk()

    rogator = step_rogator_api(args.model)
    inner, candidate = await _get_inner_and_candidate()
    client_res = await step_client_complete(inner, candidate, args.model)
    upstream = await step_upstream_raw_sse(
        inner,
        candidate,
        args.model,
        model_type_override=None,
        max_sse_lines=args.max_sse_lines,
    )

    from upstream.deepseek.client import DeepSeekClient

    rogator_client = DeepSeekClient(splitter=_splitter_stub())
    await rogator_client.startup()
    openai_res = await step_openai_chat_layer(rogator_client, args.model)

    alt_results: List[Dict[str, Any]] = []
    for mt in args.try_model_type:
        alt_results.append(
            await step_upstream_raw_sse(
                inner,
                candidate,
                args.model,
                model_type_override=mt,
                max_sse_lines=args.max_sse_lines,
            )
        )

    if args.all_candidates:
        await step_try_all_candidates(
            inner, args.model, max_sse_lines=min(args.max_sse_lines, 5)
        )

    _banner("[汇总]")
    rows = [
        ("Rogator API", rogator.get("ok")),
        ("DeepSeekClient.complete", client_res.get("ok")),
        ("上游 SSE parser (默认 model_type)", upstream.get("ok")),
        ("openai_chat 层", openai_res.get("ok")),
    ]
    for mt, res in zip(args.try_model_type, alt_results):
        rows.append((f"上游 SSE model_type={mt}", res.get("ok")))
    for name, ok in rows:
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}")

    if not any(ok for _, ok in rows):
        muted = upstream.get("biz_error") or ""
        if "muted" in muted.lower():
            print(
                textwrap.dedent(
                    f"""
                    根因: 上游返回账号封禁/静音 — {muted}
                    代码链路本身能到达 DeepSeek，但账号无法生成内容。
                    处理: 换账号、等待 mute_until 过期，或检查 persist/deepseek/sessions.json。
                    """
                ).strip()
            )
        else:
            print(
                textwrap.dedent(
                    """
                    可能原因（按优先级）：
                      1. 上游 biz 错误（看 [2] SSE 是否含 biz_code/biz_msg）
                      2. build_request_context 硬编码 model_type='default'，未映射 deepseek-v4-pro → expert
                      3. 上游 SSE 有数据但 parser 未产出 content
                      4. complete() 有 str 但 openai_chat._normalize_chunk 丢弃
                    建议: python scripts/diag_deepseek_v4_pro.py --all-candidates --try-model-type expert
                    """
                ).strip()
            )
    return 0 if any(ok for _, ok in rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek V4 Pro 空响应诊断")
    parser.add_argument("--model", default=MODEL, help="Rogator 模型 id（默认 deepseek-v4-pro）")
    parser.add_argument(
        "--try-model-type",
        nargs="*",
        default=[],
        help="额外测试的上游 model_type，如 expert flash vision",
    )
    parser.add_argument("--max-sse-lines", type=int, default=40, help="打印 SSE 行上限")
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="逐个账号探测上游（检测 mute 等 biz 错误）",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
