from __future__ import annotations

"""_strip_media_for_inject 单元测试：验证非文本 content parts 被正确剥离。"""

import copy

from upstream.qwen.chat.upload.oss import _strip_media_for_inject


def _img_part(url: str = "data:image/jpeg;base64,/9j/4AAQ") -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _video_part(url: str = "https://example.com/v.mp4") -> dict:
    return {"type": "video_url", "video_url": {"url": url}}


def _audio_part(url: str = "https://example.com/a.wav") -> dict:
    return {"type": "input_audio", "input_audio": {"url": url}}


class TestStripMediaForInject:
    def test_string_content_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "hello"
        assert result[0] is msgs[0]

    def test_single_text_part_simplified_to_str(self):
        msgs = [{"role": "user", "content": [_text_part("hello")]}]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "hello"
        assert isinstance(result[0]["content"], str)

    def test_multiple_text_parts_joined(self):
        msgs = [
            {
                "role": "user",
                "content": [_text_part("line1"), _text_part("line2")],
            }
        ]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "line1\nline2"

    def test_image_url_stripped_text_kept(self):
        msgs = [
            {
                "role": "user",
                "content": [_text_part("describe this"), _img_part()],
            }
        ]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "describe this"
        assert "base64" not in result[0]["content"]

    def test_only_image_url_becomes_empty(self):
        msgs = [{"role": "user", "content": [_img_part()]}]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == ""

    def test_video_and_audio_stripped(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    _text_part("transcribe"),
                    _video_part(),
                    _audio_part(),
                ],
            }
        ]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "transcribe"

    def test_original_messages_not_mutated(self):
        original = [
            {
                "role": "user",
                "content": [_text_part("hi"), _img_part()],
            }
        ]
        before = copy.deepcopy(original)
        _strip_media_for_inject(original)
        assert original == before

    def test_empty_messages(self):
        assert _strip_media_for_inject([]) == []
        assert _strip_media_for_inject(None) == []

    def test_mixed_roles(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {
                "role": "user",
                "content": [_text_part("look at this"), _img_part()],
            },
            {"role": "assistant", "content": "I see it"},
        ]
        result = _strip_media_for_inject(msgs)
        assert len(result) == 3
        assert result[0]["content"] == "you are helpful"
        assert result[1]["content"] == "look at this"
        assert result[2]["content"] == "I see it"

    def test_remote_image_url_also_stripped(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    _text_part("analyze"),
                    _img_part("https://example.com/photo.jpg"),
                ],
            }
        ]
        result = _strip_media_for_inject(msgs)
        assert result[0]["content"] == "analyze"
        assert "example.com" not in result[0]["content"]
