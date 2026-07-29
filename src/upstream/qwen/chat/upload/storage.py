from __future__ import annotations

"""Local media persistence, MIME types, file helpers, and OSS/persistence helpers.

Merged from: storage.py, mimes.py, files.py, persistence.py, oss.py
"""

import base64
import hashlib
import hmac
import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Tuple

from upstream.qwen.accounts import Account
from upstream.qwen.chat.routes import (
    COOKIE_REFRESH_INTERVAL,
    GENERATED_IMAGE_DIR,
    GENERATED_VIDEO_DIR,
    PERSIST_PATH,
    TTS_DIR,
)


# ---------------------------------------------------------------------------
# MIME types (from mimes.py)
# ---------------------------------------------------------------------------

FILE_TYPE_MAPPING: Final[Dict[str, str]] = {
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "video/mp4": "video",
    "video/avi": "video",
    "video/mov": "video",
    "video/quicktime": "video",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/aac": "audio",
    "audio/ogg": "audio",
    "audio/m4a": "audio",
    "audio/opus": "audio",
    "application/pdf": "file",
    "text/plain": "file",
    "text/csv": "file",
    "application/json": "file",
}

EXTENSION_TO_MIME: Final[Dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".avi": "video/avi",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/m4a",
    ".opus": "audio/opus",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}

DATA_URI_EXT_MAP: Final[Dict[str, str]] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
    "application/pdf": ".pdf",
}


def get_mime_type(filename: str) -> str:
    """Infer a MIME type from a filename."""
    return EXTENSION_TO_MIME.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")


def get_file_category(content_type: str) -> Tuple[str, str]:
    """Return ``(file_type, file_class)`` for an uploaded file."""
    file_type = FILE_TYPE_MAPPING.get(content_type, "file")
    if content_type.startswith("image/") or content_type.startswith("video/"):
        return file_type, "vision"
    if content_type.startswith("audio/"):
        return file_type, "audio"
    return file_type, "document"


# ---------------------------------------------------------------------------
# File builders (from files.py)
# ---------------------------------------------------------------------------


def build_file_object(
    file_id: str,
    file_url: str,
    filename: str,
    size: int,
    content_type: str,
    user_id: str,
) -> Dict[str, Any]:
    """Build a Qwen file object for an uploaded asset."""
    file_type, file_class = get_file_category(content_type)
    return {
        "id": file_id,
        "name": filename,
        "type": file_type,
        "size": size,
        "url": file_url,
        "file_type": content_type,
        "showType": file_type,
        "file_class": file_class,
        "user_id": user_id,
        "isQuote": False,
    }


def build_url_file_object(file_url: str, file_type: str) -> Dict[str, Any]:
    """Build a quoted file object from a remote URL."""
    filename = os.path.basename(file_url.split("?", 1)[0]) or f"remote.{file_type}"
    content_type = get_mime_type(filename)
    _, file_class = get_file_category(content_type)
    return {
        "name": filename,
        "type": file_type,
        "url": file_url,
        "file_type": content_type,
        "showType": file_type,
        "file_class": file_class,
        "isQuote": True,
    }


# ---------------------------------------------------------------------------
# Local media persistence (from storage.py)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS: Final[dict] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def build_wav_from_pcm(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Wrap PCM audio bytes in a WAV container."""
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


def _make_path(save_dir: str, prefix: str, ext: str) -> Path:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return Path(save_dir) / f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"


def save_wav_file(pcm_data: bytes, save_dir: str = TTS_DIR) -> Optional[str]:
    """Save PCM bytes as a WAV file and return the path."""
    path = _make_path(save_dir, "tts", ".wav")
    path.write_bytes(build_wav_from_pcm(pcm_data))
    return str(path)


def save_image_file(
    image_data: bytes,
    content_type: str = "image/png",
    save_dir: str = GENERATED_IMAGE_DIR,
) -> Optional[str]:
    """Save image bytes and return the local path."""
    ext = _IMAGE_EXTENSIONS.get(content_type.split(";", 1)[0].strip(), ".png")
    path = _make_path(save_dir, "generated", ext)
    path.write_bytes(image_data)
    return str(path)


def save_video_file(video_data: bytes, save_dir: str = GENERATED_VIDEO_DIR) -> Optional[str]:
    """Save video bytes and return the local path."""
    path = _make_path(save_dir, "video", ".mp4")
    path.write_bytes(video_data)
    return str(path)


# ---------------------------------------------------------------------------
# OSS authorization (from oss.py)
# ---------------------------------------------------------------------------

def build_oss_authorization(
    method: str,
    content_type: str,
    date: str,
    oss_headers: Dict[str, str],
    resource: str,
    access_key_id: str,
    access_key_secret: str,
) -> str:
    """Build an OSS V1 authorization header."""
    canonicalized = ""
    if oss_headers:
        canonicalized = "\n".join(
            f"{key}:{value}" for key, value in sorted(oss_headers.items())
        ) + "\n"
    string_to_sign = (
        f"{method}\n\n{content_type}\n{date}\n{canonicalized}{resource}"
    )
    digest = hmac.new(
        access_key_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return f"OSS {access_key_id}:{base64.b64encode(digest).decode('ascii')}"


