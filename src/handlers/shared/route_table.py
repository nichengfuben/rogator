from __future__ import annotations

"""HTTP 路由表。"""

from typing import Any, Callable, List, Tuple

from handlers.shared.route_handlers import load_route_handlers

RouteSpec = Tuple[str, str, Callable[..., Any]]


def build_route_specs() -> List[RouteSpec]:
    h = load_route_handlers()
    return [
        ("GET", "/", h["health"]),
        ("GET", "/health", h["health"]),
        ("GET", "/v1/health", h["health"]),
        ("GET", "/api/hello", h["api_hello"]),
        ("GET", "/v1/models", h["list_models"]),
        ("POST", "/v1/chat/completions", h["openai_chat"]),
        ("POST", "/v1/messages/count_tokens", h["count_tokens"]),
        ("GET", "/anthropic", h["anthropic_root"]),
        ("POST", "/anthropic", h["anthropic_root"]),
        ("POST", "/v1/messages", h["anthropic_messages"]),
        ("POST", "/anthropic/v1/messages", h["anthropic_messages"]),
        ("POST", "/anthropic/messages", h["anthropic_messages"]),
        ("GET", "/anthropic/v1/models", h["anthropic_list_models"]),
        ("POST", "/anthropic/v1/messages/count_tokens", h["count_tokens"]),
        ("POST", "/v1/images/generations", h["images_generations"]),
        ("POST", "/v1/audio/speech", h["audio_speech"]),
        ("POST", "/v1/audio/transcriptions", h["audio_transcriptions"]),
        ("POST", "/anthropic/v1/audio/transcriptions", h["anthropic_audio_transcriptions"]),
        ("GET", "/v1/realtime", h["oai_realtime"]),
        ("GET", "/anthropic/v1/realtime", h["anthropic_realtime"]),
        ("POST", "/v1/admin/refresh_models", h["admin_refresh_models"]),
        ("POST", "/v1/admin/switch_session", h["admin_switch_session"]),
        ("GET", "/v1/admin/sessions", h["admin_sessions"]),
        ("GET", "/v1/capabilities", h["capabilities"]),
        ("GET", "/v1/status", h["status"]),
    ]
