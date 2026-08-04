from __future__ import annotations

"""平台托管模型（不来自上游 chat models API，但须在 /v1/models 暴露）。"""

from typing import Dict, Final, Mapping

from server.model.model_meta import ModelMeta

QWEN_ASR_EXTERNAL_ID: Final[str] = "qwen-asr"
QWEN_ASR_INTERNAL_ID: Final[str] = "qwen-asr"

_PLATFORM_INTERNAL_IDS: Final[frozenset[str]] = frozenset({QWEN_ASR_INTERNAL_ID})


def is_platform_model(internal_id: str) -> bool:
    return internal_id in _PLATFORM_INTERNAL_IDS


def is_transcription_model(internal_id: str) -> bool:
    return internal_id == QWEN_ASR_INTERNAL_ID


def qwen_asr_model_meta() -> ModelMeta:
    from upstream.qwen.chat.routes import ASR_MAX_DURATION_SEC, ASR_SAMPLE_RATE

    # 上下文长度按最大可转写 PCM 字节估算（对外仅作能力参考）
    max_pcm_bytes = ASR_SAMPLE_RATE * 2 * ASR_MAX_DURATION_SEC
    return ModelMeta(
        context_length=max_pcm_bytes,
        capabilities={
            "asr": True,
            "transcription": True,
            "audio": True,
        },
        modality=["audio"],
    ).finalized()


def platform_model_meta(internal_id: str) -> ModelMeta | None:
    if internal_id == QWEN_ASR_INTERNAL_ID:
        return qwen_asr_model_meta()
    return None


def capabilities_for_service_api(stored: Mapping[str, bool]) -> Dict[str, bool]:
    """转写/TTS 等平台模型：不注入 chat 专用 thinking/tools。"""
    return {str(k): bool(v) for k, v in stored.items() if k}
