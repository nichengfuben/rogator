from __future__ import annotations

"""Server 子包：按职责划分。

- ``config``   — 配置加载、日志、关机
- ``client``   — Qwen 会话、聊天、上传
- ``formats``  — OpenAI/Anthropic 格式构建
- ``model``    — 模型目录、thinking、token 估算
- ``records``  — 上游响应落盘
- ``retry``    — 换号重试
"""
