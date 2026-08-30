"""refresh_models 强制 client.fetch_models() 实现（缺则抛 NotImplementedError）。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import pytest


class _DummyClient:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name

    def load_models_cache(self) -> List[str]:
        return ["a"]


@dataclass
class _FakeModelsInventory:
    qwen: Any = None


class _FakeAppState:
    def __init__(self, clients):
        self._clients = clients  # name → client
        self._models_inventory = {
            c.name: set(c.load_models_cache()) for c in clients.values()
        }
        self._models: List[str] = []
        modules = {
            c.name: type("M", (), {"name": c.name})() for c in clients.values()
        }
        registry = type("R", (), {"names": lambda self: list(modules.keys()), "modules": modules})()
        self._registry = registry
        self._rebuild_unified_models()

    def _rebuild_unified_models(self):
        seen: set = set()
        ordered: List[str] = []
        for name in self._registry.names():
            for mid in self._models_inventory.get(name, ()):  # type: ignore[attr-defined]
                if mid not in seen:
                    ordered.append(mid)
                    seen.add(mid)
        self._models = ordered

    def _models_by_upstream(self):
        return dict(self._models_inventory)

    def _should_skip_model_refresh(self, name, client, *, require_session, force, interval):
        return False

    async def refresh_models(self, *, require_session=False, force=False):
        interval = 60.0
        for name, client in self._clients.items():
            fetch = getattr(client, "fetch_models", None)
            if not callable(fetch):
                raise NotImplementedError(
                    f"upstream {name} client {type(client).__name__} missing fetch_models()"
                )
            models = await fetch(use_cache=not force)
            if not models:
                continue
            self._models_inventory[name] = set(models)
        if any(True for _ in self._clients):
            self._rebuild_unified_models()


class _HasFetch:
    def __init__(self):
        self.name = "qwen"

    def load_models_cache(self):
        return ["x"]

    async def fetch_models(self, *, use_cache=True):
        return ["x", "y"]


class _MissingFetch:
    def __init__(self):
        self.name = "broken"

    def load_models_cache(self):
        return []


async def test_refresh_models_raises_when_fetch_models_missing():
    state = _FakeAppState({"broken": _MissingFetch()})
    with pytest.raises(NotImplementedError):
        await state.refresh_models()


async def test_refresh_models_populates_when_fetch_models_present():
    state = _FakeAppState({"qwen": _HasFetch()})
    await state.refresh_models()
    assert "y" in state._models