from __future__ import annotations

"""Handler 层模型外键 → 上游内键解析。"""

from aiohttp import web

from server.formats import _error_response
from server.model.model_registry import ModelResolveError, resolve_request_model
from state import AppState


def resolve_handler_model(state: AppState, requested: str) -> str:
    """解析 API 模型外键，返回上游内键。"""
    return resolve_request_model(requested, state._models).internal_id


def model_resolve_error_response(exc: ModelResolveError) -> web.Response:
    return _error_response(exc.status, exc.message, exc.error_type)
