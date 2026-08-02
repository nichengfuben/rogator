#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen ASR 直连：测试文本 → TTS 音频 → 非流式 / 流式识别。"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    p = str(entry)
    if p not in sys.path:
        sys.path.insert(0, p)
import path_setup  # noqa: F401

from core.session.accounts import accounts_for_upstream
from upstream.qwen.client import QwenClient
from upstream.qwen.media.asr import AsrTranscriber, aprepare_pcm16_16k_mono


DEFAULT_TEXT = "你好，这是语音识别测试。"


async def _tts_wav(client: QwenClient, session, text: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="rogator_asr_") as tmp:
        wav_path = await client.synthesize_tts(text, session.token, save_dir=tmp)
        if not wav_path or not Path(wav_path).is_file():
            raise RuntimeError("Qwen TTS 未返回 WAV")
        return Path(wav_path).read_bytes()


async def _sapi_wav(text: str) -> bytes:
    """Windows SAPI 合成 WAV（TTS 不可用时的 fallback）。"""
    with tempfile.TemporaryDirectory(prefix="rogator_asr_") as tmp:
        tmp_path = Path(tmp)
        text_file = tmp_path / "text.txt"
        wav_file = tmp_path / "out.wav"
        text_file.write_text(text, encoding="utf-8")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "foreach ($v in $s.GetInstalledVoices()) { "
            "  if ($v.VoiceInfo.Culture.Name -like 'zh*') { "
            "    $s.SelectVoice($v.VoiceInfo.Name); break "
            "  } "
            "}; "
            f"$t = Get-Content -LiteralPath '{text_file}' -Encoding UTF8 -Raw; "
            f"$s.SetOutputToWaveFile('{wav_file}'); "
            "$s.Speak($t.Trim()); "
            "$s.Dispose()"
        )
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not wav_file.is_file() or wav_file.stat().st_size == 0:
            err = (stderr or b"").decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"SAPI 合成失败: {err or proc.returncode}")
        return wav_file.read_bytes()


async def _audio_from_text(client: QwenClient, session, text: str) -> tuple[bytes, str]:
    """优先 Qwen TTS，失败则 SAPI。"""
    try:
        wav = await _tts_wav(client, session, text)
        return wav, "qwen-tts"
    except Exception as exc:
        print(f"Qwen TTS 跳过 ({exc})，改用 Windows SAPI…", flush=True)
        return await _sapi_wav(text), "sapi"


def _normalize(s: str) -> str:
    return "".join(s.split())


async def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen ASR 实网测试")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="测试文本（经 TTS 转音频后识别）")
    args = parser.parse_args()
    source_text = args.text.strip()
    if not source_text:
        print("测试文本为空", flush=True)
        return 2

    pool = accounts_for_upstream("qwen")
    if not pool:
        print("无 Qwen 账号配置", flush=True)
        return 2
    account = pool[0]

    client = QwenClient(splitter=None)
    await client._ensure_http_session()

    print(f"登录 Qwen ({account.username[:6]}…)…", flush=True)
    session = await client._perform_login(account)
    if not session:
        print("登录失败", flush=True)
        return 2
    print("登录成功", flush=True)

    print(f"测试文本: {source_text}", flush=True)
    print("合成测试音频…", flush=True)
    try:
        wav_bytes, audio_src = await _audio_from_text(client, session, source_text)
    except Exception as exc:
        print(f"音频合成失败: {exc}", flush=True)
        await client.shutdown()
        return 1
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        dur = wf.getnframes() / float(wf.getframerate())
    print(f"WAV 就绪 ({audio_src}): {len(wav_bytes)} bytes, {dur:.1f}s", flush=True)

    # --- 非流式（client 封装）---
    print("\n=== 非流式 client.transcribe_audio ===", flush=True)
    try:
        text_client = await client.transcribe_audio(
            wav_bytes, session, filename="tts.wav", content_type="audio/wav", language="zh-CN",
        )
    except Exception as exc:
        print(f"client 非流式失败: {exc}", flush=True)
        await client.shutdown()
        return 1
    print(f"结果: {text_client}", flush=True)

    pcm = await aprepare_pcm16_16k_mono(wav_bytes, filename="tts.wav", content_type="audio/wav")
    http = await client._ensure_http_session()
    asr = AsrTranscriber(http, session.token)

    # --- 非流式（AsrTranscriber）---
    print("\n=== 非流式 transcribe ===", flush=True)
    try:
        text_sync = await asr.transcribe(pcm, language="zh-CN")
    except Exception as exc:
        print(f"非流式失败: {exc}", flush=True)
        await client.shutdown()
        return 1
    print(f"结果: {text_sync}", flush=True)

    # --- 流式 ---
    print("\n=== 流式 transcribe_stream ===", flush=True)
    stream_steps: List[str] = []
    try:
        async for partial in asr.transcribe_stream(pcm, language="zh-CN"):
            if partial != (stream_steps[-1] if stream_steps else ""):
                stream_steps.append(partial)
                print(f"  [partial] {partial}", flush=True)
    except Exception as exc:
        print(f"流式失败: {exc}", flush=True)
        await client.shutdown()
        return 1
    text_stream = stream_steps[-1] if stream_steps else ""
    print(f"最终结果: {text_stream}", flush=True)

    src_n = _normalize(source_text)
    sync_ok = bool(text_sync) and (_normalize(text_sync) in src_n or src_n in _normalize(text_sync))
    stream_ok = bool(text_stream) and (
        _normalize(text_stream) in src_n or src_n in _normalize(text_stream)
    )
    client_ok = bool(text_client) and (
        _normalize(text_client) in src_n or src_n in _normalize(text_client)
    )
    print(
        f"\n摘要: client_ok={client_ok}, sync_ok={sync_ok}, stream_ok={stream_ok}, "
        f"stream_steps={len(stream_steps)}",
        flush=True,
    )
    await client.shutdown()
    return 0 if client_ok and sync_ok and stream_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
