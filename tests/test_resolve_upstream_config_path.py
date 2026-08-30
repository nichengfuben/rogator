"""``resolve_upstream_config_path`` 候选路径优先级。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from server.config.app_config import resolve_upstream_config_path


class TestResolveUpstreamConfigPath:
    def test_returns_template_first(self, tmp_path: Path, monkeypatch) -> None:
        # 当所有候选路径都不存在时返回 (None, "")
        from server.config import app_config as mod

        # 替换路径构造器为全部 None
        def _factory_none(name):
            return None

        monkeypatch.setattr(mod, "_UPSTREAM_CANDIDATE_PATHS", (_factory_none,))
        path, label = mod.resolve_upstream_config_path("qwen")
        assert path is None
        assert label == ""

    def test_picks_first_existing(self, tmp_path: Path, monkeypatch) -> None:
        first = tmp_path / "template" / "upstream_config.toml"
        first.parent.mkdir(parents=True)
        first.write_text("[capabilities]\n", encoding="utf-8")

        second = tmp_path / "user.toml"
        second.write_text("[limits]\n", encoding="utf-8")

        from server.config import app_config as mod

        def _first(name):
            return first

        def _second(name):
            return second

        monkeypatch.setattr(
            mod,
            "_UPSTREAM_CANDIDATE_PATHS",
            (_first, _second),
        )
        path, label = mod.resolve_upstream_config_path("qwen")
        assert path == first
        assert label == "upstream_config.toml"

        monkeypatch.setattr(
            mod,
            "_UPSTREAM_CANDIDATE_PATHS",
            (lambda n: None, _second),
        )
        path, label = mod.resolve_upstream_config_path("qwen")
        assert path == second
        assert label == "user.toml"