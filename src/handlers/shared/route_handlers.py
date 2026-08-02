from __future__ import annotations

"""HTTP 路由 handler 延迟导入（供 route_table 使用）。"""

from typing import Any, Callable, Dict


def load_route_handlers() -> Dict[str, Callable[..., Any]]:
    from handlers.anthropic import anthropic_messages_handler
    from handlers.openai import openai_chat_handler
    from handlers.shared.audio_transcriptions import (
        anthropic_audio_transcriptions_handler,
        audio_transcriptions_handler,
    )
    from handlers.realtime import anthropic_realtime_ws_handler, oai_realtime_ws_handler
    from handlers.platform_handlers import (
        admin_refresh_models_handler,
        admin_sessions_handler,
        admin_switch_session_handler,
        anthropic_list_models_handler,
        anthropic_root_handler,
        api_hello_handler,
        audio_speech_handler,
        capabilities_handler,
        count_tokens_handler,
        health_handler,
        images_generations_handler,
        list_models_handler,
        status_handler,
    )

    return {
        "health": health_handler,
        "api_hello": api_hello_handler,
        "list_models": list_models_handler,
        "openai_chat": openai_chat_handler,
        "count_tokens": count_tokens_handler,
        "anthropic_root": anthropic_root_handler,
        "anthropic_messages": anthropic_messages_handler,
        "anthropic_list_models": anthropic_list_models_handler,
        "images_generations": images_generations_handler,
        "audio_speech": audio_speech_handler,
        "audio_transcriptions": audio_transcriptions_handler,
        "anthropic_audio_transcriptions": anthropic_audio_transcriptions_handler,
        "oai_realtime": oai_realtime_ws_handler,
        "anthropic_realtime": anthropic_realtime_ws_handler,
        "admin_refresh_models": admin_refresh_models_handler,
        "admin_switch_session": admin_switch_session_handler,
        "admin_sessions": admin_sessions_handler,
        "capabilities": capabilities_handler,
        "status": status_handler,
    }
