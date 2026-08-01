# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rogator v2.1.0 — Qwen AI adapter server. Python 3.14, aiohttp. Proxies Qwen (Alibaba LLM) through OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) compatible endpoints.

This repository is a VERY EARLY WIP. Proposing sweeping changes that improve long-term maintainability is encouraged.

## Core Priorities

1. Performance first.
2. Reliability first.
3. Keep behavior predictable under load and during failures (stability assurance mechanism for long-lived connections and streaming services).

If a tradeoff is required, choose correctness and robustness over short-term convenience.

## Maintainability

Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

## Output Style

- 回复语言默认简体中文，除非用户明确要求其他语言。
- Length: match response length to task complexity; default to fewer than four lines for conversational replies.
- Order: lead with the outcome; reasoning follows.
- Markdown: use Markdown for code and structured data, not for prose chat.
- Code comments: write comments only when the reason is non-obvious to a future reader.
- Filler phrases: do not open with affirmations such as certainly, sure, of course, or absolutely.
- End-of-turn note: write one concise past-tense summary at the end of each turn.
- Tool-call labels: write a short past-tense label after each tool call completes; do not place a colon immediately before a tool call invocation.

## Running

```
python main.py
```

## Architecture

- `handlers/` — OpenAI / Anthropic 协议适配
- `core/` — 平台核心：upstream registry、dispatch（能力+模型过滤后随机选上游）、共享错误/HTTP/SSE
- `upstream/qwen/` — Qwen 上游（client、账号、auth、chat、media）
- `server/` — 全局配置、formats、model registry/catalog、records、retry（无 client）
- `src/state.py` — AppState、调度器；经 core 缓存上游客户端
- `main.py` — aiohttp 入口
- `achecker.py` / `amerger.py` — 合规检查与合并工具
- 数据：`persist/`（全局 registry）+ `persist/qwen/`（账号、sessions、login_history）

## Dependencies

- **echotools** — Core library for logging (`echotools.base.logger`), protocol abstraction (`echotools.exec.protocol.base.ToolProtocol`), and function call injection (`echotools.exec.fncall.inject_fncall`)
- No pyproject.toml or requirements.txt — dependencies are managed externally

## Code Style

- **Chinese** comments and docstrings throughout
- Type hints everywhere; `from __future__ import annotations` in all files
- Mixin-based class composition in upstream clients
- `async/await` throughout — no sync blocking calls
- Constants: `UPPER_CASE` with `Final` annotations
- Logger: `echotools.base.logger.get_logger("rogator")`
- Section separators: `# ==== blocks`
- achecker.py enforces: max 7 children per dir, ≤400 line files (hard max 800), 50-line functions, depth 4 nesting

## Key Gotchas

- **No system role in Qwen** — system messages are folded into the last user message. Don't pass system messages directly.
- **Baxia anti-bot** — requests require `bx-ua` and `bx-umidtoken` headers (Alibaba fingerprinting). Override via `QWEN_BX_UMIDTOKEN` env var.
- **SSL disabled** (`ssl=False`) on all outbound requests.
- **Long text overflow** — inject 后超长由 splitter / OSS 处理。
- **Tool calling** — entml 协议；网关 `thinking` 恒 true（core 硬编码，与上游无关）。
- **TTS & Video** — `upstream/qwen/media/`。
- **Default model** — `qwen3.7-max`（`server/formats`）。
- **Accounts** — `persist/qwen/accounts.csv`（`upstream.qwen.accounts`）。

@AGENTS.md
