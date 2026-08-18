from __future__ import annotations

"""上游命名空间与注册表加载入口。"""

from core.registry import UpstreamRegistry, load_upstreams

__all__: list[str] = ["UpstreamRegistry", "load_upstreams"]
