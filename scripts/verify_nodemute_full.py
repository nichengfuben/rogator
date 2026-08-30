"""端到端验证：mute -> 进程退出 -> 重新构造 -> mute / index 仍在。"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from upstream.zen.proxy import NodeManager


def fresh_tmp() -> str:
    fd, path = tempfile.mkstemp(prefix="nm_persist_", suffix=".json")
    os.close(fd)
    os.unlink(path)
    return path


async def run_case_full() -> None:
    """完整生命周期：mute + switch -> 重启 -> 状态恢复。"""
    state_file = fresh_tmp()

    # ---- 阶段 1：构造 + mute 两个节点 + switch 一次 ----
    # mute_current 只 mute 当前节点（不切换），所以连 mute 两次仍 mute 同一节点。
    # 真正切换要靠 switch_next。
    nm1 = NodeManager(
        pool=["http://1.1.1.1:80", "http://2.2.2.2:80", "http://3.3.3.3:80", None],
        state_file=state_file,
    )
    await nm1.mute_current(duration=3600.0)  # mute 当前 (index=0, 1.1.1.1)
    desc_after_first = await nm1.switch_next()  # -> 1 (2.2.2.2)
    await nm1.mute_current(duration=1800.0)  # mute 当前 (index=1, 2.2.2.2)
    next_desc = await nm1.switch_next()  # -> 2 (3.3.3.3)
    print(f"[stage1] first switch: {desc_after_first} (期望: http://2.2.2.2:80)")
    print(f"[stage1] second switch: {next_desc} (期望: http://3.3.3.3:80)")
    assert desc_after_first == "http://2.2.2.2:80"
    assert next_desc == "http://3.3.3.3:80"

    on_disk_after_mute = json.loads(Path(state_file).read_text(encoding="utf-8"))
    print(f"[stage1] on_disk: {on_disk_after_mute}")
    assert on_disk_after_mute["current_node_index"] == 2
    assert on_disk_after_mute["current_node"] == "http://3.3.3.3:80"
    assert "muted" in on_disk_after_mute
    assert "http://1.1.1.1:80" in on_disk_after_mute["muted"]
    assert "http://2.2.2.2:80" in on_disk_after_mute["muted"]
    assert "http://3.3.3.3:80" not in on_disk_after_mute["muted"]

    # ---- 阶段 2：模拟进程退出，重新构造 ----
    del nm1  # 关闭

    nm2 = NodeManager(
        pool=["http://1.1.1.1:80", "http://2.2.2.2:80", "http://3.3.3.3:80", None],
        state_file=state_file,
    )
    print(f"[stage2] current: {nm2.current_description} (期望: http://3.3.3.3:80)")
    assert nm2.current_description == "http://3.3.3.3:80"
    assert "http://1.1.1.1:80" in nm2._muted
    assert "http://2.2.2.2:80" in nm2._muted
    assert "http://3.3.3.3:80" not in nm2._muted

    # ---- 阶段 3：重启后 switch_next 应跳过两个 mute 节点，直接落到 direct ----
    after_switch = await nm2.switch_next()
    print(f"[stage3] after restart switch_next: {after_switch} (期望: direct)")
    assert after_switch == "direct"

    # ---- 阶段 4：注入已过期的 mute，重启后应被 _load 剔除 ----
    nm2._muted["http://expired:80"] = time.time() - 1.0
    nm2._save_sync()  # 直接同步落盘（_load 不该再复活它）

    del nm2
    nm3 = NodeManager(
        pool=["http://1.1.1.1:80", "http://2.2.2.2:80", None],
        state_file=state_file,
    )
    print(f"[stage4] after restart _muted: {sorted(nm3._muted.keys())}")
    assert "http://expired:80" not in nm3._muted

    # ---- 阶段 5：旧格式（无 muted 字段）兜底 ----
    Path(state_file).write_text(
        json.dumps(
            {"current_node_index": 0, "current_node": "http://1.1.1.1:80", "updated_at": 0},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    del nm3
    nm4 = NodeManager(
        pool=["http://1.1.1.1:80", "http://2.2.2.2:80", None],
        state_file=state_file,
    )
    print(f"[stage5] old format -> index: {nm4._current_index} muted: {nm4._muted}")
    assert nm4._current_index == 0
    assert nm4._muted == {}

    os.unlink(state_file)
    print("\nALL CASES PASS")


if __name__ == "__main__":
    asyncio.run(run_case_full())