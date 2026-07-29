from __future__ import annotations

"""Select an upstream by capabilities + model ownership, then random among matches."""

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from core.errors import CapabilityError, ModelNotAvailableError
from core.registry import UpstreamModule, UpstreamRegistry, get_registry


def _caps_ok(mod: UpstreamModule, required: Iterable[str]) -> bool:
    for key in required:
        if key == "thinking":
            continue
        if not mod.capabilities.get(key, False):
            return False
    return True


def _model_ok(mod: UpstreamModule, model_id: str, models_by_upstream: Dict[str, Set[str]]) -> bool:
    owned = models_by_upstream.get(mod.name) or set()
    if not owned:
        # Unknown inventory: allow (single-upstream bootstrap / cache cold)
        return True
    return model_id in owned


def candidate_upstreams(
    *,
    model_id: str,
    required_capabilities: Sequence[str],
    models_by_upstream: Optional[Dict[str, Set[str]]] = None,
    registry: Optional[UpstreamRegistry] = None,
) -> List[UpstreamModule]:
    reg = registry or get_registry()
    inventory = models_by_upstream or {}
    out: List[UpstreamModule] = []
    for mod in reg.modules.values():
        if not _caps_ok(mod, required_capabilities):
            continue
        if not _model_ok(mod, model_id, inventory):
            continue
        out.append(mod)
    return out


def select_upstream(
    *,
    model_id: str,
    required_capabilities: Sequence[str] = ("chat",),
    models_by_upstream: Optional[Dict[str, Set[str]]] = None,
    registry: Optional[UpstreamRegistry] = None,
) -> UpstreamModule:
    cands = candidate_upstreams(
        model_id=model_id,
        required_capabilities=required_capabilities,
        models_by_upstream=models_by_upstream,
        registry=registry,
    )
    if not cands:
        # Distinguish empty registry vs filter miss
        reg = registry or get_registry()
        if not reg.modules:
            raise CapabilityError("No upstreams registered")
        raise ModelNotAvailableError(
            f"No upstream for model={model_id!r} caps={list(required_capabilities)!r}"
        )
    return random.choice(cands)


def create_client_for_request(
    *,
    model_id: str,
    splitter: Any,
    required_capabilities: Sequence[str] = ("chat",),
    models_by_upstream: Optional[Dict[str, Set[str]]] = None,
    registry: Optional[UpstreamRegistry] = None,
) -> Any:
    mod = select_upstream(
        model_id=model_id,
        required_capabilities=required_capabilities,
        models_by_upstream=models_by_upstream,
        registry=registry,
    )
    return mod.create_client(splitter)
