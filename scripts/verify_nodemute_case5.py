"""端到端验证：过期 mute 在 _save_sync 时被剔除。

上次写错过（用 if False 三元把调用绕开了），这次直接用 asyncio.run 跑。
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 让 import 找到 src/（__file__ 在 scripts/，要拼到上一级）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from upstream.zen.proxy import NodeManager


def fresh_tmp() -> str:
    fd, path = tempfile.mkstemp(prefix="nm_expired_", suffix=".json")
    os.close(fd)
    os.unlink(path)  # 起始空文件，让 _load 走 FileNotFound 分支
    return path


async def run_case_5() -> None:
    """造一个含 1 个过期 + 1 个仍存活 mute 的 NodeManager，调 _save_sync，读盘确认过期项不在。"""
    state_file = fresh_tmp()
    nm = NodeManager(
        pool=["http://1.1.1.1:80", "http://2.2.2.2:80", None],
        state_file=state_file,
    )

    # 注入：1 个已过期 + 1 个仍存活
    nm._muted["http://expired:80"] = time.time() - 5.0   # 早就该解除
    nm._muted["http://alive:80"] = time.time() + 3600.0  # 未来 1 小时

    # 真正调 _save_sync（不通过 mute_current 间接调）
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, nm._save_sync)

    # 读盘校验
    on_disk = json.loads(Path(state_file).read_text(encoding="utf-8"))
    saved_muted = on_disk.get("muted", {})

    print("=== case 5: expired mute filtered on save ===")
    print(f"saved_muted keys: {sorted(saved_muted.keys())}")
    assert "http://expired:80" not in saved_muted, f"过期项仍在盘里: {saved_muted}"
    assert "http://alive:80" in saved_muted, f"活项被误删: {saved_muted}"
    print("PASS")

    # 顺便校验：内存里过期的仍在（save 不该偷偷清掉它，否则会让 in-flight 判断逻辑出错）
    # 实际上 _save_sync 不动 self._muted，只过滤写入 payload。所以内存中"过期"项还在。
    print(f"memory _muted keys (应仍含过期项): {sorted(nm._muted.keys())}")
    assert "http://expired:80" in nm._muted
    assert "http://alive:80" in nm._muted
    print("memory untouched: PASS")

    os.unlink(state_file)


if __name__ == "__main__":
    asyncio.run(run_case_5())