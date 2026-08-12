#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""二分探测各 Qwen 上游模型 completions 最大发送字符数。

硬约束：禁止任务附件 / 文件上传；整段正文只走 message content（files=None）。
判定：出现「被挤爆」/ FAIL_SYS_USER_VALIDATE / RGV587_ERROR::SM → 触及上限。
每次请求最多重试 3 次（含首次）。

用法：
  python -u scripts/probe_qwen_send_char_limit.py --until-complete --resume
  python -u scripts/probe_qwen_send_char_limit.py --resume --epsilon 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, TextIO

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import Account, accounts_for_upstream
from core.dispatch import stream_openai_chat
from server.formats.errors import BaxiaSmBlockedError
from server.model.model_registry import MODEL_REGISTRY_FILE, load_model_registry
from server.retry import stream_with_session_retry
from state import AppState

Outcome = Literal["ok", "sm", "error"]

PREFIX = "只回复一个字：好\n"
SANITY_CHARS = 200
DEFAULT_OUT = ROOT / "logs" / "qwen_send_char_limits.jsonl"
DEFAULT_RUN_LOG = ROOT / "logs" / "qwen_send_char_limits.run.log"


class LiveLog:
    """stdout + UTF-8 文件双写，立即 flush，便于实时盯日志。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO = path.open("a", encoding="utf-8", newline="\n")
        self.path = path

    def line(self, msg: str) -> None:
        print(msg, flush=True)
        self._fp.write(msg + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def _sync_proxy(proxy: str) -> None:
    proxy = (proxy or "").strip()
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    if not proxy:
        for key in keys:
            os.environ.pop(key, None)
        return
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


def _load_account(ext_index: int) -> Account:
    import re

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
        raise RuntimeError("无可用账号")
    return pool[0]


def _is_sm_limit(err: str) -> bool:
    markers = (
        "被挤爆",
        "FAIL_SYS_USER_VALIDATE",
        "RGV587_ERROR::SM",
        "BaxiaSmBlockedError",
    )
    return any(m in err for m in markers)


def make_prompt(n: int) -> str:
    if n < len(PREFIX):
        return ("好" * n) if n > 0 else "好"
    return PREFIX + ("x" * (n - len(PREFIX)))


@dataclass
class ProbeResult:
    external_id: str
    internal_id: str
    max_ok_chars: Optional[int]
    lo: int
    hi: int
    probes: int
    status: str
    note: str = ""


def is_complete(row: Optional[ProbeResult]) -> bool:
    return row is not None and row.status == "ok" and row.max_ok_chars is not None


async def try_send(
    state: AppState,
    *,
    model: str,
    n_chars: int,
    max_events: int,
    retries: int,
    log: LiveLog,
    tag: str,
) -> tuple[Outcome, str, int]:
    """走网关同源 stream_openai_chat + session_retry；返回 wire_chars=prompt_meta。"""
    messages = [{"role": "user", "content": make_prompt(n_chars)}]
    req_id = f"probe-{tag}-{n_chars}-{int(time.time() * 1000)}"
    hits = 0
    wire_chars = 0
    t0 = time.time()
    log.line(
        f"  [{tag}] TRY user={n_chars:,} attempt=1/{retries} "
        f"(gateway stream_openai_chat, files=[], send_full_prompt)"
    )

    async def make_stream():
        async for event in stream_openai_chat(
            state,
            messages,
            model,
            None,
            req_id,
            files=[],
        ):
            yield event

    try:
        async for event in stream_with_session_retry(
            req_id,
            state,
            make_stream,
            max_retry=max(0, retries - 1),
        ):
            et = str(event.get("type") or "")
            if et == "prompt_meta":
                wire_chars = int(event.get("prompt_chars") or 0)
            if et == "error":
                raise RuntimeError(str(event.get("message") or event)[:300])
            if et in ("answer", "thinking", "thinking_summary"):
                hits += 1
                if hits >= max_events:
                    break
        elapsed = int((time.time() - t0) * 1000)
        if hits > 0:
            log.line(
                f"  [{tag}] OK   user={n_chars:,} wire={wire_chars:,} "
                f"hits={hits} {elapsed}ms"
            )
            return "ok", "", wire_chars
        last_err = "no_model_content"
        log.line(f"  [{tag}] ERR  user={n_chars:,} wire={wire_chars:,} {last_err} {elapsed}ms")
        return "error", last_err, wire_chars
    except BaxiaSmBlockedError as exc:
        elapsed = int((time.time() - t0) * 1000)
        msg = f"{type(exc).__name__}: {exc}"
        if _is_sm_limit(msg):
            log.line(
                f"  [{tag}] SM   user={n_chars:,} wire={wire_chars:,} "
                f"{elapsed}ms {msg[:160]}"
            )
            return "sm", msg[:240], wire_chars
        log.line(f"  [{tag}] ERR  user={n_chars:,} {elapsed}ms {msg[:160]}")
        return "error", msg[:320], wire_chars
    except Exception as exc:
        elapsed = int((time.time() - t0) * 1000)
        msg = f"{type(exc).__name__}: {exc}"
        if _is_sm_limit(msg):
            log.line(
                f"  [{tag}] SM   user={n_chars:,} wire={wire_chars:,} "
                f"{elapsed}ms {msg[:160]}"
            )
            return "sm", msg[:240], wire_chars
        log.line(f"  [{tag}] ERR  user={n_chars:,} {elapsed}ms {msg[:160]}")
        return "error", msg[:320], wire_chars


async def binary_search_model(
    state: AppState,
    *,
    external_id: str,
    internal_id: str,
    lo: int,
    hi: int,
    retries: int,
    max_events: int,
    pause_s: float,
    epsilon: int,
    log: LiveLog,
    cool_s: float,
) -> ProbeResult:
    probes = 0
    note = ""
    tag = external_id
    best_wire = 0

    async def one(n: int) -> tuple[Outcome, str, int]:
        nonlocal probes
        out, err, wire = await try_send(
            state,
            model=internal_id,
            n_chars=n,
            max_events=max_events,
            retries=retries,
            log=log,
            tag=tag,
        )
        probes += 1
        await asyncio.sleep(pause_s)
        return out, err, wire

    log.line(f"  [{tag}] verify lo={lo:,} (user chars, max_ok=wire prompt_meta)")
    out, err, wire = await one(lo)
    if out == "sm":
        log.line(f"  [{tag}] lo SM → cool {cool_s:.0f}s then retry once")
        await asyncio.sleep(cool_s)
        out, err, wire = await one(lo)
    if out == "sm":
        return ProbeResult(
            external_id, internal_id, None, lo, hi, probes, "lo_sm",
            note=f"lo={lo} user wire={wire} 已 SM: {err}",
        )
    if out != "ok":
        return ProbeResult(
            external_id, internal_id, None, lo, hi, probes, "lo_error",
            note=f"lo={lo} 失败: {err}",
        )
    lo_user = lo
    best_wire = wire

    cur_hi = hi
    for _ in range(4):
        log.line(f"  [{tag}] verify hi={cur_hi:,}")
        out, err, wire = await one(cur_hi)
        if out == "sm":
            break
        if out == "ok":
            lo_user = cur_hi
            best_wire = wire
            cur_hi = min(cur_hi * 2, 1_024_000)
            if cur_hi == lo_user:
                return ProbeResult(
                    external_id, internal_id, best_wire, lo_user, cur_hi, probes, "ok",
                    note=f"user={lo_user} wire={best_wire} 触及脚本上界仍 OK",
                )
            continue
        return ProbeResult(
            external_id, internal_id, None, lo_user, cur_hi, probes, "hi_error",
            note=f"hi={cur_hi} 非 SM 失败: {err}",
        )
    else:
        return ProbeResult(
            external_id, internal_id, lo_user, lo_user, cur_hi, probes, "hi_not_sm",
            note="抬高上界后仍未出现 SM",
        )

    hi_user = cur_hi
    eps = max(1, int(epsilon))
    while hi_user - lo_user > eps:
        mid = (lo_user + hi_user) // 2
        log.line(f"  [{tag}] mid={mid:,}  (lo={lo_user:,} hi={hi_user:,})")
        out, err, wire = await one(mid)
        if out == "ok":
            lo_user = mid
            best_wire = wire
        elif out == "sm":
            hi_user = mid
        else:
            note = f"mid={mid} 非 SM 失败，按触及上限收紧: {err}"
            log.line(f"  [{tag}] WARN {note}")
            hi_user = mid
    if hi_user - lo_user > 1:
        note = (
            (note + "; " if note else "")
            + f"user≤{eps} 取保守 user={lo_user} wire={best_wire} (sm_at_user≈{hi_user})"
        )
    return ProbeResult(
        external_id, internal_id, best_wire, lo_user, hi_user, probes, "ok", note=note,
    )


async def wait_account_ready(
    state: AppState,
    *,
    model: str,
    log: LiveLog,
    retries: int,
    max_events: int,
    round_cool_s: float,
    max_waits: int,
) -> bool:
    for attempt in range(1, max_waits + 1):
        log.line(
            f"  [account] sanity user={SANITY_CHARS:,} model={model} wait={attempt}/{max_waits}"
        )
        out, err, wire = await try_send(
            state,
            model=model,
            n_chars=SANITY_CHARS,
            max_events=max_events,
            retries=retries,
            log=log,
            tag="account",
        )
        if out == "ok":
            log.line(f"  [account] ready wire={wire:,}")
            return True
        log.line(
            f"  [account] still blocked ({out}) cool {round_cool_s:.0f}s "
            f"{err[:120] if err else ''}"
        )
        if attempt < max_waits:
            await asyncio.sleep(round_cool_s)
    return False


async def init_probe_state() -> AppState:
    state = AppState()
    state.splitter.send_full_prompt = True
    await state.startup_upstreams()
    qwen = state._clients.get("qwen")
    if qwen is not None:
        prelogin = getattr(qwen, "prelogin_accounts", None)
        if callable(prelogin):
            await prelogin(1)
        ensure = getattr(qwen, "ensure_prelogin", None)
        if callable(ensure):
            await ensure()
    return state


def qwen_registry_models() -> List[tuple[str, str]]:
    reg = load_model_registry(MODEL_REGISTRY_FILE)
    out: List[tuple[str, str]] = []
    for entry in reg.entries_in_order:
        if entry.internal_id.startswith("deepseek-"):
            continue
        out.append((entry.external_id, entry.internal_id))
    return out


def load_done(path: Path) -> dict[str, ProbeResult]:
    done: dict[str, ProbeResult] = {}
    if not path.is_file():
        return done
    raw = path.read_text(encoding="utf-8-sig")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "external_id" not in obj:
            continue
        try:
            done[str(obj["external_id"])] = ProbeResult(
                external_id=str(obj["external_id"]),
                internal_id=str(obj["internal_id"]),
                max_ok_chars=obj.get("max_ok_chars"),
                lo=int(obj.get("lo") or 0),
                hi=int(obj.get("hi") or 0),
                probes=int(obj.get("probes") or 0),
                status=str(obj.get("status") or ""),
                note=str(obj.get("note") or ""),
            )
        except (TypeError, ValueError):
            continue
    return done


def save_all_results(
    out_path: Path,
    models: List[tuple[str, str]],
    done: dict[str, ProbeResult],
) -> None:
    rows: List[ProbeResult] = []
    for ext, internal in models:
        if ext in done:
            rows.append(done[ext])
        else:
            rows.append(
                ProbeResult(ext, internal, None, 0, 0, 0, "pending", note="未测")
            )
    out_path.write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    md_path = out_path.with_suffix(".md")
    lines = [
        "| external | internal | max_ok_chars | status | probes | note |",
        "|---|---|---:|---|---:|---|",
    ]
    for r in rows:
        max_s = f"{r.max_ok_chars:,}" if r.max_ok_chars is not None else "-"
        note = (r.note or "").replace("|", "/")
        lines.append(
            f"| {r.external_id} | {r.internal_id} | {max_s} | {r.status} | {r.probes} | {note} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: List[ProbeResult], log: LiveLog) -> None:
    log.line("")
    log.line("## 结果表")
    log.line("")
    log.line(
        f"{'external':<28} {'internal':<28} {'max_ok':>10} {'status':<10} probes note"
    )
    log.line("-" * 110)
    for r in rows:
        max_s = f"{r.max_ok_chars:,}" if r.max_ok_chars is not None else "-"
        note = (r.note or "")[:48]
        log.line(
            f"{r.external_id:<28} {r.internal_id:<28} {max_s:>10} {r.status:<10} "
            f"{r.probes:>5} {note}"
        )


async def main_async(args: argparse.Namespace) -> int:
    _sync_proxy(args.proxy)
    if args.until_complete:
        args.resume = True

    run_log = Path(args.run_log)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    if not args.append_log:
        run_log.write_text("", encoding="utf-8")
    log = LiveLog(run_log)
    log.line(
        f"=== probe start gateway=stream_openai_chat proxy={'none' if not (args.proxy or '').strip() else args.proxy} "
        f"lo={args.lo} hi={args.hi} epsilon={args.epsilon} retries={args.retries} "
        f"until_complete={args.until_complete} max_ok=wire(prompt_meta) ==="
    )
    log.line(f"run_log={run_log.resolve()}")

    models = qwen_registry_models()
    all_models = models
    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m[0] in wanted or m[1] in wanted]
        if not models:
            log.line("无匹配模型")
            log.close()
            return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else {}
    if not args.resume:
        done = {}

    account = _load_account(args.ext_account)
    log.line(f"account={account.username[:6]}…  models={len(models)} resume={bool(args.resume)}")
    state = await init_probe_state()
    log.line("gateway AppState ready (send_full_prompt=True, files=[])")

    sanity_model = next(
        (internal for ext, internal in models if ext == "qwen3-coder-plus"),
        models[0][1],
    )

    round_num = 0
    while True:
        round_num += 1
        pending = [(ext, internal) for ext, internal in models if not is_complete(done.get(ext))]
        if not pending:
            log.line(f"\n=== all {len(models)} models complete (rounds={round_num - 1}) ===")
            break

        log.line(
            f"\n--- round {round_num} pending={len(pending)}/{len(models)} "
            f"done={len(models) - len(pending)} ---"
        )

        if not await wait_account_ready(
            state,
            model=sanity_model,
            log=log,
            retries=args.retries,
            max_events=args.max_events,
            round_cool_s=args.round_cool,
            max_waits=args.max_account_waits,
        ):
            log.line("  [account] 仍不可用，结束本轮后重试")
            if not args.until_complete:
                break
            log.line(f"  [account] until-complete → sleep {args.round_cool:.0f}s")
            await asyncio.sleep(args.round_cool)
            continue

        for i, (ext, internal) in enumerate(pending, 1):
            prev = done.get(ext)
            if is_complete(prev):
                continue
            if prev and prev.status == "lo_sm":
                log.line(f"\n[{i}/{len(pending)}] {ext} RETRY (prev lo_sm)")
            else:
                log.line(f"\n[{i}/{len(pending)}] {ext} ({internal})")
            t0 = time.time()
            row = await binary_search_model(
                state,
                external_id=ext,
                internal_id=internal,
                lo=args.lo,
                hi=args.hi,
                retries=args.retries,
                max_events=args.max_events,
                pause_s=args.pause,
                epsilon=args.epsilon,
                log=log,
                cool_s=args.cool,
            )
            elapsed = int(time.time() - t0)
            log.line(
                f"  => max_ok={row.max_ok_chars} status={row.status} "
                f"probes={row.probes} elapsed={elapsed}s {row.note}"
            )
            done[ext] = row
            save_all_results(out_path, all_models, done)

        pending_after = [ext for ext, _ in models if not is_complete(done.get(ext))]
        if not pending_after:
            break
        if not args.until_complete:
            log.line(f"\n未完成 {len(pending_after)} 个模型（未开 until-complete）")
            break
        log.line(
            f"\nround {round_num} 仍缺 {len(pending_after)} 个: "
            f"{', '.join(pending_after[:5])}{'…' if len(pending_after) > 5 else ''}"
        )
        log.line(f"sleep {args.round_cool:.0f}s before next round")
        await asyncio.sleep(args.round_cool)

    rows = [done[ext] for ext, _ in all_models if ext in done]
    for ext, internal in all_models:
        if ext not in done:
            rows.append(ProbeResult(ext, internal, None, 0, 0, 0, "pending"))
    # 保持注册表顺序
    order = {ext: i for i, (ext, _) in enumerate(all_models)}
    rows.sort(key=lambda r: order.get(r.external_id, 999))
    save_all_results(out_path, all_models, done)
    print_table(rows, log)
    log.line(f"\njsonl={out_path.resolve()}")
    log.line(f"md={out_path.with_suffix('.md').resolve()}")
    log.line("=== probe done ===")

    try:
        await state.shutdown()
    except Exception:
        pass
    log.close()
    return 0 if all(is_complete(done.get(ext)) for ext, _ in all_models) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="二分探测 Qwen 发送字符上限（纯 msg）")
    parser.add_argument("--lo", type=int, default=500, help="二分下界（user 字符，inject 后 wire 见 prompt_meta）")
    parser.add_argument("--hi", type=int, default=160_000, help="预期触及 SM 的上界")
    parser.add_argument(
        "--epsilon",
        type=int,
        default=512,
        help="二分停止带宽（hi-lo≤epsilon 即收束，默认 512）",
    )
    parser.add_argument("--retries", type=int, default=3, help="每次请求最大尝试次数")
    parser.add_argument("--pause", type=float, default=0.8, help="探测间隔秒")
    parser.add_argument("--cool", type=float, default=20.0, help="单模型下界 SM 后冷却秒")
    parser.add_argument(
        "--round-cool",
        type=float,
        default=300.0,
        help="整轮未完成 / 账号仍 SM 时的冷却秒（默认 300）",
    )
    parser.add_argument(
        "--max-account-waits",
        type=int,
        default=12,
        help="每轮开始前账号 sanity 最多等待次数",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="循环重测直至 21 模型均有 max_ok（lo_sm 等会重测）",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="追加 run.log（默认每轮启动清空）",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="HTTP(S) 代理 URL；默认空=直连并清除环境变量代理",
    )
    parser.add_argument("--ext-account", type=int, default=99)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--run-log", default=str(DEFAULT_RUN_LOG))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 jsonl 里已 status=ok 的模型",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="只测指定 external/internal（可重复）",
    )
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    # 尽量无缓冲（配合 python -u）
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    raise SystemExit(main())
