"""``MediaDownloadMixin`` 静态辅助函数的最小覆盖。

覆盖三类不变量：
1. ``extract_base64_images`` 只挑 ``data:`` URI，且能容错非 list/dict 节点
2. ``extract_remote_media_urls`` 兼容 ``image_url / video_url / input_audio`` 三种 part
3. ``ext_from_content_type`` 在 DATA_URI_EXT_MAP 未命中时按 ``image/*`` 兜底，
   未知类型退到 ``.bin``
"""
from __future__ import annotations

from typing import Any, Dict, List

from upstream.qwen.chat.upload.files import MediaDownloadMixin


class TestExtractBase64Images:
    def test_collects_only_data_uri(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                ],
            }
        ]
        result = MediaDownloadMixin.extract_base64_images(messages)
        assert result == ["data:image/png;base64,AAA"]

    def test_skips_non_list_content(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "纯文本"},
            {"role": "user", "content": None},
            {"role": "user", "content": 42},
        ]
        assert MediaDownloadMixin.extract_base64_images(messages) == []

    def test_skips_non_image_url_type(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "no image"},
                    {"type": "image_url", "image_url": None},
                ],
            }
        ]
        assert MediaDownloadMixin.extract_base64_images(messages) == []

    def test_accepts_image_url_as_string(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,BBB"},
                ],
            }
        ]
        result = MediaDownloadMixin.extract_base64_images(messages)
        assert result == ["data:image/jpeg;base64,BBB"]


class TestExtractRemoteMediaUrls:
    def test_image_video_audio_collected(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://x.example/a.png"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://x.example/b.mp4"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"url": "https://x.example/c.mp3"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,XX"},
                    },
                    {"type": "text", "text": "ignored"},
                ],
            }
        ]
        result = MediaDownloadMixin.extract_remote_media_urls(messages)
        assert result == [
            "https://x.example/a.png",
            "https://x.example/b.mp4",
            "https://x.example/c.mp3",
        ]

    def test_skips_empty_and_data_uri(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YY"},
                    },
                    {"type": "video_url", "video_url": "https://ok.example/v.mp4"},
                ],
            }
        ]
        result = MediaDownloadMixin.extract_remote_media_urls(messages)
        assert result == ["https://ok.example/v.mp4"]

    def test_non_list_content_yields_empty(self) -> None:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "text"}]
        assert MediaDownloadMixin.extract_remote_media_urls(messages) == []


class TestExtFromContentType:
    def test_known_image_png(self) -> None:
        assert MediaDownloadMixin.ext_from_content_type("image/png") == ".png"

    def test_unknown_image_subtype_falls_back_to_subtype(self) -> None:
        # DATA_URI_EXT_MAP 没收录但属于 image/* 时，扩展名取子类型
        assert MediaDownloadMixin.ext_from_content_type("image/xyz") == ".xyz"

    def test_unknown_type_returns_bin(self) -> None:
        assert MediaDownloadMixin.ext_from_content_type("application/x-foo") == ".bin"

    def test_empty_type_returns_bin(self) -> None:
        assert MediaDownloadMixin.ext_from_content_type("") == ".bin"