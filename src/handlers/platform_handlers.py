from __future__ import annotations

"""Health、模型列表、TTS/图生、管理端点等 HTTP handlers。"""

import json
from datetime import datetime, timezone
from typing import Any, Dict

from aiohttp import web

from echotools.base.logger import get_logger
from handlers import get_state
from handlers.api_errors import (
    anthropic_error_response,
    model_resolve_error_response,
    resolve_handler_model,
)
from core.registry import get_registry
from server.formats import (
    ClientDisconnectedError,
    _error_response,
    _json_response,
    client_disconnected_response,
    read_request_json,
)
from server.model.model_registry import ModelResolveError
from server.model.token_estimate import (
    estimate_anthropic_injected_input_tokens,
    estimate_anthropic_request_input_tokens,
)

logger = get_logger("rogator")


async def health_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "ok",
        "platform": "rogator",
        "timestamp": int(__import__("time").time()),
    })


async def list_models_handler(request: web.Request) -> web.Response:
    from server.model.model_catalog import build_openai_models_list
    from server.model.model_registry import get_model_registry, list_external_models

    state = get_state()
    registry = get_model_registry()
    external_ids = list_external_models(state._models)
    entries = [registry.by_external[eid] for eid in external_ids]
    fetch_ts = int(state.models_fetch_timestamp()) if state.models_fetch_timestamp() > 0 else 0
    meta = state.merged_model_meta()
    payload: Dict[str, Any] = {
        "object": "list",
        "data": build_openai_models_list(
            entries,
            meta_by_id=meta,
            created=fetch_ts or int(__import__("time").time()),
            owner_of=state.owner_of_model,
        ),
    }
    if fetch_ts > 0:
        payload["updated_at"] = fetch_ts
    return _json_response(payload)


async def anthropic_list_models_handler(request: web.Request) -> web.Response:
    from server.model.model_catalog import build_openai_model_entry
    from server.model.model_registry import get_model_registry, list_external_models

    state = get_state()
    registry = get_model_registry()
    external_ids = list_external_models(state._models)
    fetch_ts = int(state.models_fetch_timestamp()) if state.models_fetch_timestamp() > 0 else 0
    created_unix = fetch_ts or int(__import__("time").time())
    created_at = datetime.fromtimestamp(created_unix, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    meta = state.merged_model_meta()
    data = [
        {
            "type": "model",
            "id": entry["id"],
            "display_name": entry["id"],
            "created_at": created_at,
            "max_input_tokens": entry["context_length"],
            "capabilities": entry.get("capabilities"),
            "modality": entry.get("modality"),
        }
        for eid in external_ids
        for entry in [build_openai_model_entry(
            eid,
            registry_entry=registry.by_external[eid],
            meta_by_id=meta,
            created=created_unix,
            owned_by=state.owner_of_model(registry.by_external[eid].internal_id),
        )]
    ]
    payload: Dict[str, Any] = {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }
    if fetch_ts > 0:
        payload["updated_at"] = fetch_ts
    return _json_response(payload)


async def anthropic_root_handler(request: web.Request) -> web.Response:
    return web.Response(
        status=200,
        headers={
            "Content-Type": "application/json",
            "Anthropic-Version": "2023-06-01",
        },
        text=json.dumps({
            "type": "message",
            "version": "2023-06-01",
            "status": "ok",
            "endpoints": ["/v1/messages", "/anthropic/v1/messages"],
        }),
    )


async def count_tokens_handler(request: web.Request) -> web.Response:
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("Client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except (json.JSONDecodeError, ValueError):
        return anthropic_error_response(400, "Invalid JSON body")
    state = get_state()
    requested = str(body.get("model") or state.model)
    try:
        model = resolve_handler_model(state, requested)
    except ModelResolveError as exc:
        return model_resolve_error_response(exc, protocol="anthropic")
    try:
        from handlers.anthro.normalize import _build_anthropic_protocol_options

        protocol_options = _build_anthropic_protocol_options(body)
    except ValueError:
        protocol_options = None
    try:
        tokens = estimate_anthropic_injected_input_tokens(
            body,
            protocol=state.protocol,
            model=model,
            protocol_options=protocol_options,
        )
    except Exception:
        logger.debug("count_tokens inject estimate failed, falling back to raw body", exc_info=True)
        tokens = estimate_anthropic_request_input_tokens(body)
    return _json_response({"input_tokens": tokens})


async def audio_speech_handler(request: web.Request) -> web.Response:
    state = get_state()
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("Client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except (json.JSONDecodeError, ValueError):
        return _error_response(400, "Invalid JSON body")
    text = body.get("input", "")
    if not text:
        return _error_response(400, "Missing required field: input")
    if not body.get("voice"):
        return _error_response(400, "Missing required field: voice")
    try:
        model = resolve_handler_model(state, str(body.get("model") or state.model))
    except ModelResolveError as exc:
        return model_resolve_error_response(exc)
    qwen = state.client_for(model, ("tts",), upstream_name="qwen")
    session = await qwen.get_valid_session()
    if not session:
        return _error_response(503, "No valid Qwen session available")
    local_path = await qwen.synthesize_tts(text, session.token, model=model)
    if not local_path:
        return _error_response(502, "TTS synthesis failed")
    from pathlib import Path
    audio_bytes = Path(local_path).read_bytes()
    return web.Response(body=audio_bytes, content_type="audio/wav")


async def images_generations_handler(request: web.Request) -> web.Response:
    state = get_state()
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("Client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except (json.JSONDecodeError, ValueError):
        return _error_response(400, "Invalid JSON body")
    prompt = body.get("prompt", "")
    image_url = body.get("image") or body.get("image_url", "")
    if not prompt:
        return _error_response(400, "Missing required field: prompt")
    if not image_url:
        return _error_response(
            400,
            "Missing required field: image (Rogator /v1/images/generations is image-to-video)",
        )
    try:
        model = resolve_handler_model(state, str(body.get("model") or state.model))
    except ModelResolveError as exc:
        return model_resolve_error_response(exc)
    size = body.get("size", "16:9")
    qwen = state.client_for(model, ("image_gen",), upstream_name="qwen")
    session = await qwen.get_valid_session()
    if not session:
        return _error_response(503, "No valid Qwen session available")
    result = await qwen.generate_video(
        prompt, image_url, session.token, session.user_id, model=model, size=size,
    )
    if not result.get("success"):
        return _error_response(502, result.get("error", "Generation failed"))
    video_url = result.get("video_url", "")
    return _json_response({
        "created": int(__import__("time").time()),
        "data": [{
            "url": video_url,
            "revised_prompt": prompt,
            "local_path": result.get("local_path", ""),
        }],
    })


async def capabilities_handler(request: web.Request) -> web.Response:
    return _json_response({
        "platform": "rogator",
        "capabilities": get_registry().merged_capabilities(),
        "protocol": "entml",
    })


async def status_handler(request: web.Request) -> web.Response:
    state = get_state()
    qwen = state._clients.get("qwen")
    session_total = qwen.session_count if qwen is not None else 0
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "running",
        "sessions": {"total": session_total},
        "scheduler": {"pending": state.scheduler.pending, "active": state.tracker.count},
        "models": {"count": len(state._models), "default": state.model},
        "upstreams": list(state._clients.keys()),
    })


async def admin_refresh_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    await state.refresh_models(force=True)
    return _json_response({
        "status": "ok",
        "models": state._models,
        "count": len(state._models),
    })


async def admin_switch_session_handler(request: web.Request) -> web.Response:
    state = get_state()
    qwen = state._clients.get("qwen")
    if qwen is None:
        return _error_response(503, "Qwen upstream not available")
    old = (
        qwen.current_session.username[:6]
        if qwen.current_session else "none"
    )
    new = await qwen.switch_to_next()
    return _json_response({
        "status": "ok",
        "previous": old,
        "current": new.username[:6] if new else "none",
    })


async def admin_sessions_handler(request: web.Request) -> web.Response:
    state = get_state()
    qwen = state._clients.get("qwen")
    if qwen is None:
        return _json_response({"sessions": [], "total": 0})
    return _json_response({
        "sessions": [
            {"username": s.username[:6] + "***", "valid": s.is_valid}
            for s in qwen._sessions
        ],
        "total": qwen.session_count,
    })
