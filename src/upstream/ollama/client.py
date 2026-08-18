from __future__ import annotations

"""Ollama 上游客户端：纯静态注册表，无账号、无代理池。"""

import json
import logging
import random
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from core.session.models_cache import ModelsCacheMixin
from core.transport.owned import HttpTransportMixin
from upstream.ollama.chat_stream import post_chat_stream
from upstream.ollama.routes import REGISTRY_FILE, SKIP_PATTERN

logger = logging.getLogger("rogator")


class OllamaClient(HttpTransportMixin, ModelsCacheMixin):
    UPSTREAM_NAME = "ollama"

    def __init__(self, splitter: Any = None) -> None:
        self._splitter = splitter
        self._init_http_transport()
        model_to_servers, all_models = self._load_registry()
        self._model_to_servers: Dict[str, List[str]] = model_to_servers
        self._init_models_cache(all_models)

    def _load_registry(self) -> tuple[Dict[str, List[str]], List[str]]:
        """从 registry.json 加载并过滤，返回 {model: [urls]} 和模型列表。"""
        reg_path = Path(REGISTRY_FILE)
        if not reg_path.exists():
            logger.warning("ollama registry not found: %s", REGISTRY_FILE)
            return {}, []
        try:
            raw = json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("ollama registry load failed: %s", exc)
            return {}, []

        models_section = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models_section, dict):
            logger.warning("ollama registry has no models section")
            return {}, []

        model_to_servers: Dict[str, List[str]] = {}
        for model_name, model_info in models_section.items():
            if SKIP_PATTERN.search(model_name):
                continue
            servers = model_info.get("servers") if isinstance(model_info, dict) else None
            if not isinstance(servers, list):
                continue
            urls: list[str] = []
            for srv in servers:
                if not isinstance(srv, dict):
                    continue
                base_url = srv.get("base_url")
                ip = srv.get("ip", "")
                if not base_url or not isinstance(base_url, str):
                    continue
                if SKIP_PATTERN.search(ip) or SKIP_PATTERN.search(base_url):
                    continue
                urls.append(base_url.rstrip("/"))
            if urls:
                model_to_servers[model_name] = urls

        all_models = sorted(model_to_servers.keys())
        logger.info("ollama registry loaded: %d models, %d total server entries",
                     len(all_models), sum(len(v) for v in model_to_servers.values()))
        return model_to_servers, all_models

    def load_models_cache(self) -> List[str]:
        return list(self._models)

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        await self.close_http_transport()

    async def fetch_models(self, *, use_cache: bool = True) -> List[str]:
        return list(self._models)

    async def stream_chat(
        self,
        payload: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = str(payload.get("model") or "")
        servers = self._model_to_servers.get(model)
        if not servers:
            from server.formats import UpstreamUnavailableError
            raise UpstreamUnavailableError(
                f"ollama model not found: {model}",
                upstream="ollama",
            )
        server_url = random.choice(servers)
        async for event in post_chat_stream(self, payload, server_url):
            yield event
