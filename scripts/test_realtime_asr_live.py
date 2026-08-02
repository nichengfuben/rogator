#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网关 Realtime ASR：OAI + Anthropic WebSocket 实网测试。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import tempfile
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

DEFAULT_TEXT = "你好，这是实时语音识别测试。"
DEFAULT_BASE = "http://127.0.0.1:8932"
MODEL = "qwen3-7-max"


import importlib.util

async def _sapi_pcm16(text: str) -> bytes:
    from upstream.qwen.media.asr import aprepare_pcm16_16k_mono

    spec = importlib.util.spec_from_file_location(
        "qwen_asr_live", ROOT / "scripts" / "test_qwen_asr_live.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wav = await mod._sapi_wav(text)
    return await aprepare_pcm16_16k_mono(wav, filename="t.wav", content_type="audio/wav")


async def _run_oai(base: str, pcm: bytes) -> bool:
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/v1/realtime?model={MODEL}"
    deltas: list[str] = []
    completed = ""
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(ws_url, heartbeat=30) as ws:
            msg = json.loads((await ws.receive()).data)
            if msg.get("type") != "session.created":
                print("OAI: no session.created", msg, flush=True)
                return False
            await ws.send_str(json.dumps({
                "type": "session.update",
                "session": {"input_audio_transcription": {"language": "zh-CN"}},
            }))
            await ws.receive()
            chunk = 3200
            for off in range(0, len(pcm), chunk):
                await ws.send_str(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[off:off + chunk]).decode(),
                }))
                await asyncio.sleep(0.02)
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            for _ in range(20):
                m = await ws.receive(timeout=30)
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                evt = json.loads(m.data)
                t = evt.get("type")
                if t == "conversation.item.input_audio_transcription.delta":
                    deltas.append(str(evt.get("delta") or ""))
                elif t == "conversation.item.input_audio_transcription.completed":
                    completed = str(evt.get("transcript") or "")
                    break
                elif t == "error":
                    print("OAI error", evt, flush=True)
                    return False
    print(f"OAI realtime: deltas={len(deltas)} completed={completed!r}", flush=True)
    return bool(completed)


async def _run_ant(base: str, pcm: bytes) -> bool:
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/anthropic/v1/realtime?model={MODEL}"
    text_parts: list[str] = []
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(ws_url, heartbeat=30) as ws:
            await ws.receive()
            await ws.send_str(json.dumps({
                "type": "session.update",
                "session": {"input_audio_transcription": {"language": "zh-CN"}},
            }))
            await ws.receive()
            chunk = 3200
            for off in range(0, len(pcm), chunk):
                await ws.send_str(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[off:off + chunk]).decode(),
                }))
                await asyncio.sleep(0.02)
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            for _ in range(30):
                m = await ws.receive(timeout=30)
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                evt = json.loads(m.data)
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text_parts.append(str(delta.get("text") or ""))
                elif evt.get("type") == "message_stop":
                    break
    full = "".join(text_parts)
    print(f"Anthropic realtime: text={full!r}", flush=True)
    return bool(full.strip())


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    pcm = await _sapi_pcm16(args.text.strip())
    print(f"PCM {len(pcm)} bytes from text: {args.text}", flush=True)
    oai_ok = await _run_oai(args.base, pcm)
    ant_ok = await _run_ant(args.base, pcm)
    return 0 if oai_ok and ant_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
