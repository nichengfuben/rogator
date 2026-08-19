# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rogator — Multi-upstream AI adapter server. Python 3.8+, aiohttp. Proxies multiple LLM upstreams (Qwen, DeepSeek, Zen, Cursor, Ollama) through OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) compatible endpoints.

This repository is a VERY EARLY WIP. Proposing sweeping changes that improve long-term maintainability is encouraged.

## Core Priorities

1. Performance first.
2. Reliability first.
3. Keep behavior predictable under load and during failures (stability assurance mechanism for long-lived connections and streaming services).

If a tradeoff is required, choose correctness and robustness over short-term convenience.

## Maintainability

Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

## Output Style

- Default reply language: Simplified Chinese, unless the user explicitly requests another language.
- Length: match response length to task complexity; default to fewer than four lines for conversational replies.
- Order: lead with the outcome; reasoning follows.
- Markdown: use Markdown for code and structured data, not for prose chat.
- Code comments: write comments only when the reason is non-obvious to a future reader.
- Filler phrases: do not open with affirmations such as certainly, sure, of course, or absolutely.
- End-of-turn note: write one concise past-tense summary at the end of each turn.
- Tool-call labels: write a short past-tense label after each tool call completes; do not place a colon immediately before a tool call invocation.

## Running

```
python main.py              # Start the server
python -m pytest tests/ -q  # Run tests
python achecker.py           # Code compliance check
```

## Architecture

- `src/handlers/` — OpenAI / Anthropic protocol adaptation
- `src/core/` — Platform core: upstream registry, dispatch (capability + model filtering, random selection), shared error/HTTP/SSE
- `src/upstream/` — Upstream implementations: qwen, deepseek, zen, cursor, ollama
- `src/server/` — Global config, formats, model registry/catalog, records, retry (no client)
- `src/state.py` — AppState, scheduler; caches upstream clients via core
- `main.py` — aiohttp entry point
- `achecker.py` / `amerger.py` — Compliance checker and merge tool
- Data: `persist/` (per-upstream subdirectories: qwen/deepseek/zen/ollama)

## Dependencies

- **echotools** (`>=2.4.8`) — logging (`echotools.base.logger`), protocol (`ToolProtocol`), fncall inject, tool id (`gen_tool_id`, `fix_tool_call_id`)
- Dependencies managed via `requirements.txt` (runtime) and `requirements-dev.txt` (dev/test)

## External Libraries

- **echotools** (`>=2.4.8`) — Internal toolkit for LLM gateway infrastructure. Provides structured logging (`echotools.base.logger`), tool protocol definitions and registry (`ToolProtocol`, `get_protocol`), streaming function-call parser for SSE (`FncallStreamParser`), entml thinking-mode protocol (`entml_think.core`), prompt injection for function-calling instructions (`inject_fncall`), and tool-call ID generation/normalization (`gen_tool_id`, `fix_tool_call_id`, `ensure_toolu_tool_call_id`).
- **aiohttp-socks** (`>=0.8.0`) — SOCKS4/5 proxy connector for aiohttp; used by upstream clients that route through proxy pools (e.g., Zen).

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
- **Baxia anti-bot** — Qwen requests require `bx-ua` and `bx-umidtoken` headers (Alibaba fingerprinting). Override via `QWEN_BX_UMIDTOKEN` env var.
- **SSL disabled** (`ssl=False`) on all outbound requests.
- **Long text overflow** — post-inject overflow handled by splitter / OSS.
- **Tool calling** — entml protocol; gateway `thinking` is always true (hardcoded in core, independent of upstream).
- **TTS & Video** — `src/upstream/qwen/media/`.
- **Default model** — `qwen3.7-max` (`server/formats`).
- **Accounts** — Qwen: `persist/qwen/accounts.csv`; DeepSeek: `persist/deepseek/accounts.csv`; Zen/Cursor/Ollama have no account pool.
- **DeepSeek** — HIF token management; WAF challenge and user-muted errors require special handling.
- **Zen** — Proxy pool auto-mutes failing nodes (429/502/connection errors, 1-hour duration); does NOT log prompt/response/SSE.
- **Cursor** — Access tokens obtained via self-hosted Token Service; does NOT log prompt/response/SSE.
- **Ollama** — Pure static registry (`persist/ollama/registry.json`), maps models to server URLs, random server selection per request.

@AGENTS.md
