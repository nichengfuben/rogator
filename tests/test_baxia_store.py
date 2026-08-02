from __future__ import annotations

import json
from pathlib import Path

import pytest

from upstream.qwen.auth.baxia_store import (
    BaxiaProfile,
    ensure_pool_baxia_profiles,
    ensure_profiles,
    get_profile,
    regenerate_profile,
)


def test_ensure_profiles_keeps_existing_and_fills_missing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "baxia_profiles.json"
    existing = BaxiaProfile(fingerprint="fp-a^1.0.0", bx_umidtoken="T2gA" + "a" * 40)
    path.write_text(
        json.dumps({"a@test.com": existing.to_dict()}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("upstream.qwen.auth.baxia_store.baxia_profiles_path", lambda: path)

    kept, created = ensure_profiles(["a@test.com", "b@test.com"])
    assert kept == 1
    assert created == 1

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["a@test.com"]["fingerprint"] == "fp-a^1.0.0"
    assert "b@test.com" in data
    assert data["b@test.com"]["fingerprint"] != "fp-a^1.0.0"


def test_get_profile_does_not_overwrite(tmp_path, monkeypatch) -> None:
    path = tmp_path / "baxia_profiles.json"
    monkeypatch.setattr("upstream.qwen.auth.baxia_store.baxia_profiles_path", lambda: path)

    first = get_profile("u@test.com")
    second = get_profile("u@test.com")
    assert first.fingerprint == second.fingerprint
    assert first.bx_umidtoken == second.bx_umidtoken


def test_regenerate_profile_overwrites(tmp_path, monkeypatch) -> None:
    path = tmp_path / "baxia_profiles.json"
    monkeypatch.setattr("upstream.qwen.auth.baxia_store.baxia_profiles_path", lambda: path)

    first = get_profile("u@test.com")
    regenerated = regenerate_profile("u@test.com")
    second = get_profile("u@test.com")
    assert regenerated.fingerprint == second.fingerprint
    assert first.fingerprint != second.fingerprint
    assert first.bx_umidtoken != second.bx_umidtoken


def test_ensure_pool_reads_accounts_csv(tmp_path, monkeypatch) -> None:
    qwen_dir = tmp_path / "persist" / "qwen"
    qwen_dir.mkdir(parents=True)
    (qwen_dir / "accounts.csv").write_text(
        "email,password\none@test.com,pw1\ntwo@test.com,pw2\n",
        encoding="utf-8",
    )
    profiles = qwen_dir / "baxia_profiles.json"
    monkeypatch.setattr(
        "core.session.accounts._UPSTREAM_CSV",
        {"qwen": qwen_dir / "accounts.csv"},
    )
    monkeypatch.setattr("upstream.qwen.auth.baxia_store.baxia_profiles_path", lambda: profiles)

    kept, created = ensure_pool_baxia_profiles()
    assert kept == 0
    assert created == 2
    kept2, created2 = ensure_pool_baxia_profiles()
    assert kept2 == 2
    assert created2 == 0

    data = json.loads(profiles.read_text(encoding="utf-8"))
    assert set(data) == {"one@test.com", "two@test.com"}
    assert data["one@test.com"]["bx_umidtoken"] != data["two@test.com"]["bx_umidtoken"]
