"""NSSM 代理环境变量诊断脚本。

以 Windows 服务方式运行时，将所有环境变量和 aiohttp/urllib 代理检测结果写入日志文件。
用于排查 NSSM 注入的环境变量为何 Python 无法识别。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import getproxies

LOG_DIR = Path(__file__).resolve().parent.parent / "persist" / "nssm_diag"


def dump_env(label: str) -> dict:
    """收集环境变量快照。"""
    proxy_keys = [
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
        "ALL_PROXY", "all_proxy",
        "NO_PROXY", "no_proxy",
        "PROXY", "proxy",
    ]
    env_snapshot = {}
    for k, v in sorted(os.environ.items()):
        env_snapshot[k] = v

    proxy_specific = {k: os.environ.get(k, "<NOT SET>") for k in proxy_keys}

    try:
        urllib_proxies = getproxies()
    except Exception as exc:
        urllib_proxies = {"_error": str(exc)}

    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "proxy_env_vars": proxy_specific,
        "urllib_getproxies": urllib_proxies,
        "all_env_count": len(env_snapshot),
        "all_env": env_snapshot,
    }


async def test_aiohttp_proxy() -> dict:
    """测试 aiohttp 是否能读到代理。"""
    result = {}
    try:
        import aiohttp
        connector = aiohttp.TCPConnector()
        session = aiohttp.ClientSession(connector=connector, trust_env=True)
        # aiohttp 内部通过 _proxy_from_env 读取代理
        # 检查 connector 的 _proxy 属性
        result["aiohttp_trust_env"] = True
        result["aiohttp_connector_type"] = type(connector).__name__
        await session.close()
        await connector.close()
    except Exception as exc:
        result["aiohttp_error"] = str(exc)
    return result


async def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"diag_{int(time.time())}.json"

    snapshot = dump_env("nssm_service_start")
    aiohttp_result = await test_aiohttp_proxy()
    snapshot["aiohttp_test"] = aiohttp_result

    # 额外检测：os.environ 是否在 import 时被快照
    snapshot["environ_id"] = id(os.environ)
    snapshot["environ_keys_at_runtime"] = len(os.environ)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    # 同时写一个简短的状态文件方便快速查看
    status_file = LOG_DIR / "last_status.txt"
    lines = [
        f"time: {snapshot['timestamp']}",
        f"python: {snapshot['python_executable']}",
        f"env_count: {snapshot['all_env_count']}",
    ]
    for k, v in snapshot["proxy_env_vars"].items():
        lines.append(f"{k}={v}")
    lines.append(f"urllib_proxies={snapshot['urllib_getproxies']}")
    if "aiohttp_error" in aiohttp_result:
        lines.append(f"aiohttp_error={aiohttp_result['aiohttp_error']}")
    else:
        lines.append(f"aiohttp_connector={aiohttp_result.get('aiohttp_connector_type', '?')}")
    status_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
