# AGENTS.md

This file provides behavioral instructions for AI coding assistants working on this project.

**IMPORTANT: CLAUDE.md and AGENTS.md specification files must be written in English only.**

## Language Rules

- **Always respond to users in Simplified Chinese.** All replies, explanations, suggestions, status updates, and interactions must be in Simplified Chinese.
- Code comments and docstrings must also be in Chinese (as specified in CLAUDE.md).
- Regardless of the language the user uses to ask a question, always reply in Chinese.

## Comment and Documentation Standards

### Prohibited Boilerplate (never include in source code)

- "Standard module", "Project standard module", "As a Provider-Evo project standard module"
- End-of-file "Module contract" / "Related modules" separator comment blocks
- Mechanical docstrings like "Chinese description:", "Public method/class xxx."
- Self-referential notes like "See modification guide...", "Keep single file 200-400 lines"
- Restating docs-src/, PROJECT_DECISIONS.md, coverage gates, or other documentation content inside .py files

Detailed design belongs in `docs-src/`; source code comments should only explain aspects of the current implementation that are not immediately obvious.

### Division of Responsibility Between Comments and Documentation

Comments serve **people reading the code**, not compliance word counts. Good comments answer **"why was it written this way"** and **"what would go wrong otherwise"** — they do not repeat information already conveyed by function names.

#### Recommended Comment Scenarios

| Scenario | What to write |
|----------|---------------|
| Locks / concurrency | Why a particular lock type was chosen; whether multi-event-loop or cross-thread calls exist; consequences of choosing the wrong lock |
| Cancellation / timeouts | Child tasks must be explicitly cancelled when the parent coroutine is cancelled, otherwise background requests leak connections or invalidate timeouts |
| Magic constants | Business impact of thresholds, windows, burst detection; consequences of false triggers or statistical boundary conditions |
| Compatibility / degradation | Legacy data, optional APIs, old plugin fields — what the degraded behavior is and why a more precise approach isn't feasible |
| Cross-layer contracts | Frontend/backend conventions for HTTP status codes, error code mappings |
| Trust boundaries | Which fields are trusted vs user-controlled; why nicknames/free-text must not be used for security decisions |
| Operation ordering | Step sequence during rollback, switchover, cleanup to avoid mixing old and new state |

**Style reference (not a template — do not copy verbatim):**

```python
# SQLite WAL allows only a single writer; the gateway has multiple event loops and
# cross-thread direct calls, so a process-level threading.Lock is required —
# asyncio.Lock cannot provide cross-loop mutual exclusion.
# When the caller is cancelled via wait_for, child tasks must be cancelled and
# awaited for cleanup, otherwise httpx requests continue occupying connections in
# the background.
# Use 502 instead of upstream 401/403: the frontend fetchWithAuth treats 401 as
# an expired WebUI session.
```

Public API docstrings: **one sentence describing the responsibility is sufficient**; do not enumerate parameters/return values when they can be inferred from type annotations. For complex logic, use inline `#` comments next to branches or constants rather than stacking them at the top of the file.

#### Bad Comments (Prohibited)

- Repeating function/parameter names; mechanical docstrings from the prohibited boilerplate list above
- Empty docstrings or end-of-file contract blocks added solely to pass `achecker`
- Restating docs-src/, gate rules, or architectural essays inside `.py` files
