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
python main.py [--port 8932] [--host 0.0.0.0] [--prelogin 1] [--log-level DEBUG]
```

## Architecture

- `handlers/` — API endpoint handlers (OpenAI chat completions, Anthropic messages)
- `core/` — Main QwenClient (mixin-based: AuthMixin, UploadMixin, MediaMixin, VideoGenMixin, LogsMixin), with submodules: crypto/, transport/, compat/, storage/, media/
- `server/` — Simplified QwenClient (used by state.py), format builders, OSS upload
- `state.py` — AppState, RequestScheduler, LongTextSplitter with resilient executor
- `accounts.py` — Hardcoded disposable account pool (758KB), auto token refresh rotation
- `main.py` — Entry point, aiohttp server lifecycle
- `mvp/` — Standalone smoke-test scripts (chat.py, test_upload_analysis.py)
- `achecker.py` — Project compliance checker (dir children, file lines, function length, nesting depth)
- `amerger.py` — File content merge utility (all text files → single document)
- Data directory: `persist/qwen/` (models cache, state persistence)

## Dependencies

- **echotools** — Core library for logging (`echotools.logger`), protocol abstraction (`echotools.protocol.base.ToolProtocol`), and function call injection (`echotools.fncall.inject_fncall`)
- No pyproject.toml or requirements.txt — dependencies are managed externally

## Code Style

- **Chinese** comments and docstrings throughout
- Type hints everywhere; `from __future__ import annotations` in all files
- Mixin-based class composition (see core/client.py)
- `async/await` throughout — no sync blocking calls
- Constants: `UPPER_CASE` with `Final` annotations
- Logger: `echotools.logger.get_logger("rogator")`
- Section separators: `# ==== blocks`
- achecker.py enforces: max 7 children per dir, 200–400 line files (hard max 800), 50-line functions, depth 4 nesting

## Key Gotchas

- **No system role in Qwen** — system messages are folded into the last user message. Don't pass system messages directly.
- **Baxia anti-bot** — requests require `bx-ua` and `bx-umidtoken` headers (Alibaba fingerprinting). Override via `QWEN_BX_UMIDTOKEN` env var.
- **SSL disabled** (`ssl=False`) on all outbound requests.
- **Long text overflow** — texts >10240 chars are split; overflow uploaded to Alibaba Cloud OSS via STS tokens. DNS fallback: `47.113.75.199`.
- **Tool calling** — uses custom `entml` XML protocol. `inject_fncall` imported from `echotools.fncall`; `TOOL_INSTRUCTION` template in `handlers/__init__.py`.
- **Smart proxy selection** — `ProxySelector` with persistence (`PROXY_SELECTOR_PERSIST_PATH`), uses latency heuristics for endpoint routing.
- **Two QwenClient implementations** — `core/client.py` (main, mixin-based) vs `server/qwen_client.py` (simplified, used by state.py).
- **Compat fallback pattern** — `core/client.py` uses `try: from src.core.X` / `except ModuleNotFoundError: from .compat.runtime` for Candidate, ModelsCache, ProxySelector.
- **TTS & Video** — `TtsService` and `VideoService` in `core/media/` for media generation.
- **Default model** — `qwen3.7-max` (defined in `server/formats.py`).

@AGENTS.md