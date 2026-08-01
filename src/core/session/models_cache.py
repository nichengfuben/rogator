from __future__ import annotations

"""上游模型列表缓存：刷新间隔判定与字段初始化。"""

import time
from typing import Any, Dict, List


class ModelsCacheMixin:
    _models: List[str]
    _model_meta: Dict[str, Any]
    _models_fetch_time: float

    def _init_models_cache(
        self,
        models: List[str],
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self._models = list(models)
        self._model_meta = dict(meta or {})
        self._models_fetch_time = 0.0

    def models_refresh_due(self, interval: float) -> bool:
        if interval <= 0:
            return True
        if self._models_fetch_time <= 0:
            return True
        return (time.time() - self._models_fetch_time) >= interval
