#!/usr/bin/env python3
"""Rogator 发图测试：OpenAI 兼容 ``/v1/chat/completions``（base64 + 远程 URL）。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

# 1×1 红色 PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_DEFAULT_REMOTE = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"


def _post(base: str, model: str, messages: list, *, stream: bool = False) -> dict:
    url = f"{base.rstrip('/')}/v1/chat/completions"
    payload = {"model": model, "stream": stream, "messages": messages}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _content_from_body(body: dict) -> str:
    return str(
        (body.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    ).strip()


def _run_case(name: str, base: str, model: str, messages: list) -> bool:
    print(f"\n=== {name} model={model} ===")
    try:
        body = _post(base, model, messages)
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {err[:1200]}")
        return False
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False
    content = _content_from_body(body)
    usage = body.get("usage") or {}
    print(f"content: {content[:500]!r}")
    print(f"usage: {usage}")
    ok = bool(content) and int(usage.get("completion_tokens") or 0) > 0
    print("OK" if ok else "FAIL (empty or zero completion_tokens)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="https://rogator.airoe.cn",
        help="Rogator base URL",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl-plus",
        help="Vision-capable model id",
    )
    args = parser.parse_args()
    prompt = "请用一句中文描述图片：主色是什么、有没有文字或图案。"

    b64_uri = f"data:image/png;base64,{_TINY_PNG_B64}"
    b64_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_uri}},
            ],
        }
    ]
    url_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _DEFAULT_REMOTE},
                },
            ],
        }
    ]

    print(f"base={args.base}")
    ok1 = _run_case("base64 image", args.base, args.model, b64_messages)
    ok2 = _run_case("remote image URL", args.base, args.model, url_messages)
    if not ok1 and args.model == "qwen3-vl-plus":
        print("\n--- fallback: qwen3.5-plus ---")
        ok1 = _run_case("base64 image", args.base, "qwen3.5-plus", b64_messages)
    return 0 if (ok1 or ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
