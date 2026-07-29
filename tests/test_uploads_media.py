from __future__ import annotations

import unittest

from upstream.qwen.chat.upload.files import UploadMixin


class _Client(UploadMixin):
    pass


class TestExtractRemoteMediaUrls(unittest.TestCase):
    def test_image_video_audio_parts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    {"type": "video_url", "video_url": {"url": "https://example.com/b.mp4"}},
                    {
                        "type": "input_audio",
                        "input_audio": {"url": "https://example.com/c.wav"},
                    },
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        urls = _Client.extract_remote_media_urls(messages)
        self.assertEqual(
            urls,
            [
                "https://example.com/a.png",
                "https://example.com/b.mp4",
                "https://example.com/c.wav",
            ],
        )


if __name__ == "__main__":
    unittest.main()
