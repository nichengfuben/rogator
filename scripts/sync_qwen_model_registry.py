#!/usr/bin/env python3
"""从 Qwen 上游拉取模型列表并更新 config/model_registry.jsonl。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import accounts_for_upstream
from server.model.model_registry import MODEL_REGISTRY_FILE, ModelRegistryEntry, load_model_registry
from upstream.qwen.client import QwenClient

NATIVE_THINKING = frozenset({"qwen3.8-max-preview"})


def internal_to_external(internal_id: str) -> str:
    if internal_id.startswith("qwen3."):
        return "qwen3-" + internal_id[len("qwen3.") :]
    return internal_id


def default_flags(internal_id: str) -> tuple[bool, bool]:
    if internal_id in NATIVE_THINKING:
        return False, True
    return True, True


def line_for(entry: ModelRegistryEntry) -> str:
    return (
        f"{entry.external_id}:{entry.internal_id}:"
        f"{str(entry.uses_entml).lower()}:{str(entry.uses_entml_tools).lower()}"
    )


async def main() -> int:
    pool = accounts_for_upstream("qwen")
    if not pool:
        print("无 Qwen 账号", flush=True)
        return 2

    client = QwenClient(splitter=None)
    await client._ensure_http_session()
    print(f"登录 {pool[0].username[:8]}...", flush=True)
    session = await client._perform_login(pool[0])
    if not session:
        print("登录失败", flush=True)
        return 2

    client._sessions = [session]
    models = await client.fetch_models(use_cache=False)
    if not models:
        print("拉取模型失败", flush=True)
        return 2

    print(f"上游模型 {len(models)} 个", flush=True)
    for model in models:
        print(f"  - {model}")

    reg = load_model_registry(MODEL_REGISTRY_FILE)
    kept: list[ModelRegistryEntry] = []
    seen_internal: set[str] = set()
    upstream_set = set(models)
    deepseek_entries: list[ModelRegistryEntry] = []
    existing_qwen: dict[str, ModelRegistryEntry] = {}

    for entry in reg.entries_in_order:
        if entry.internal_id.startswith("deepseek-"):
            deepseek_entries.append(entry)
            seen_internal.add(entry.internal_id)
            continue
        if entry.internal_id in upstream_set:
            existing_qwen[entry.internal_id] = entry
            seen_internal.add(entry.internal_id)

    added = 0
    for internal_id in models:
        if internal_id in existing_qwen:
            kept.append(existing_qwen[internal_id])
            continue
        think, tools = default_flags(internal_id)
        kept.append(
            ModelRegistryEntry(internal_to_external(internal_id), internal_id, think, tools)
        )
        seen_internal.add(internal_id)
        added += 1
        print(f"新增注册 {internal_to_external(internal_id)}:{internal_id}", flush=True)

    kept.extend(deepseek_entries)

    removed = sum(
        1
        for entry in reg.entries_in_order
        if not entry.internal_id.startswith("deepseek-")
        and entry.internal_id not in upstream_set
    )

    MODEL_REGISTRY_FILE.write_text(
        "\n".join(line_for(entry) for entry in kept) + "\n",
        encoding="utf-8",
    )
    deepseek_count = sum(1 for entry in kept if entry.internal_id.startswith("deepseek-"))
    print(
        f"注册表已更新: 保留 {len(kept) - added} 新增 {added} 移除 {removed} DeepSeek {deepseek_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
