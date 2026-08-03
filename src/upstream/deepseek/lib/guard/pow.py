from __future__ import annotations

# src/platforms/deepseek/core/pow.py
"""DeepSeek PoW（工作量证明）求解器"""

import base64
import ctypes
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from upstream.deepseek.lib.protocol.consts import (
    WASM_META,
    WASM_PATH,
    WASM_URL,
)
from upstream.deepseek.lib.protocol.headers import build_basic_headers

logger = logging.getLogger(__name__)


def _run_wasm_solve(
    store: Any,
    ex: Any,
    challenge: str,
    salt: str,
    difficulty: int,
    expire_at: int,
) -> int:

    mem = ex["memory"]
    add_sp = ex["__wbindgen_add_to_stack_pointer"]
    alloc = ex["__wbindgen_export_0"]
    wasm_solve = ex["wasm_solve"]

    def _w(off: int, data: bytes) -> None:
        base = ctypes.cast(mem.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base + off, data, len(data))  # type: ignore[arg-type]

    def _r(off: int, sz: int) -> bytes:
        base = ctypes.cast(mem.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base + off, sz)  # type: ignore[arg-type]

    def _enc(text: str) -> Tuple[int, int]:
        d = text.encode()
        ln = len(d)
        pv = alloc(store, ln, 1)
        p = int(pv.value) if hasattr(pv, "value") else int(pv)
        _w(p, d)
        return p, ln

    ret = add_sp(store, -16)
    try:
        pc, lc = _enc(challenge)
        prefix = "{}_{}".format(salt, expire_at) + "_"
        pp, lp = _enc(prefix)
        wasm_solve(store, ret, pc, lc, pp, lp, float(difficulty))
        st = struct.unpack("<i", _r(ret, 4))[0]
        val = struct.unpack("<d", _r(ret + 8, 8))[0]
        if st == 0:
            raise RuntimeError("WASM 求解失败")
        return int(val)
    finally:
        add_sp(store, 16)


class WasmPow:
    """WASM PoW 求解器。"""

    ALGO: str = "DeepSeekHashV1"

    def __init__(self) -> None:

        self._bytes: Optional[bytes] = None
        try:
            from wasmtime import Linker, Module, Store  # noqa: F401

            self._available: bool = os.path.exists(WASM_PATH)
        except ImportError:
            self._available = False
            logger.info("wasmtime 不可用，PoW 将跳过")

    @property
    def available(self) -> bool:
        return self._available

    def reload(self) -> None:

        self._bytes = None
        self._available = os.path.exists(WASM_PATH)

    def _load(self) -> bytes:

        if self._bytes is None:
            self._bytes = Path(WASM_PATH).read_bytes()
        return self._bytes

    def solve(
        self,
        challenge: str,
        salt: str,
        difficulty: int,
        expire_at: int,
    ) -> int:

        from wasmtime import Linker, Module, Store

        store = Store()
        linker = Linker(store.engine)
        module = Module(store.engine, self._load())
        inst = linker.instantiate(store, module)
        ex = inst.exports(store)
        return _run_wasm_solve(store, ex, challenge, salt, difficulty, expire_at)


def _wasm_needs_refresh() -> bool:

    need = not os.path.exists(WASM_PATH)
    if not need and os.path.exists(WASM_META):
        try:
            meta = json.loads(Path(WASM_META).read_text(encoding="utf-8"))
            if time.time() - meta.get("downloaded_at", 0) > 86400:
                need = True
        except Exception:
            need = True
    return need


def _save_wasm_content(content: bytes) -> None:
    # 哈希相同时仅刷新元数据时间戳，避免无谓 IO。
    import hashlib

    sha = hashlib.sha256(content).hexdigest()
    if os.path.exists(WASM_PATH):
        old = hashlib.sha256(Path(WASM_PATH).read_bytes()).hexdigest()
        if old == sha:
            Path(WASM_META).write_text(
                json.dumps({"downloaded_at": time.time(), "sha256": sha}),
                encoding="utf-8",
            )
            return

    tmp = WASM_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, WASM_PATH)
    Path(WASM_META).write_text(
        json.dumps(
            {
                "downloaded_at": time.time(),
                "sha256": sha,
                "size": len(content),
            }
        ),
        encoding="utf-8",
    )
    logger.info("WASM 已更新: %d bytes", len(content))


async def download_wasm(session: Any) -> None:

    Path(WASM_PATH).parent.mkdir(parents=True, exist_ok=True)
    if not _wasm_needs_refresh():
        return

    try:
        async with session.get(
            WASM_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/147.0.0.0 Safari/537.36"
                ),
                "Referer": "https://chat.deepseek.com/",
            },
            ssl=False,
        ) as resp:
            if resp.status != 200:
                logger.warning("WASM 下载失败: HTTP %d", resp.status)
                return
            content = await resp.read()

        _save_wasm_content(content)
    except Exception as exc:
        logger.warning("WASM 下载异常: %s", exc)


def _extract_challenge_dict(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:

    biz = data["data"]["biz_data"]
    cd = biz.get("challenge") or biz  # PoW 挑战字段在部分响应里嵌套在 challenge 键下
    # 部分版本响应直接在 biz_data 层
    if "algorithm" not in cd and "challenge" in biz:
        cd = biz
    algorithm = cd.get("algorithm", "")
    if algorithm and algorithm != WasmPow.ALGO:
        logger.warning("PoW 算法不匹配: %s", algorithm)
        return None
    return cd


def _build_pow_payload(
    cd: Dict[str, Any],
    pow_solver: WasmPow,
    target_path: str,
) -> str:

    challenge = cd.get("challenge", "")
    salt = cd.get("salt", "")
    difficulty = int(cd.get("difficulty", 144000))
    expire_at = int(cd.get("expire_at", int(time.time()) + 600))
    signature = cd.get("signature", "")

    ans = pow_solver.solve(challenge, salt, difficulty, expire_at)
    pd = {
        "algorithm": WasmPow.ALGO,
        "challenge": challenge,
        "salt": salt,
        "answer": ans,
        "signature": signature,
        "target_path": target_path,
    }
    return base64.b64encode(
        json.dumps(pd, separators=(",", ":"), ensure_ascii=False).encode()
    ).decode()


async def get_pow_response(
    session: Any,
    token: str,
    pow_solver: WasmPow,
    target_path: str = "/api/v0/chat/completion",
) -> str:
    """获取并计算 PoW 响应字符串，失败时返回空字符串。"""
    if not pow_solver.available:
        return ""

    import aiohttp as _aiohttp

    from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST
    from upstream.deepseek.lib.protocol.headers import build_basic_headers

    headers = build_basic_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat/create_pow_challenge".format(DEFAULT_HOST),
            headers=headers,
            json={"target_path": target_path},
            timeout=_aiohttp.ClientTimeout(total=30),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            if data.get("code") != 0:
                return ""
            cd = _extract_challenge_dict(data)
            if cd is None:
                return ""
            return _build_pow_payload(cd, pow_solver, target_path)
    except Exception as exc:
        logger.warning("PoW 失败: %s", exc)
        return ""
