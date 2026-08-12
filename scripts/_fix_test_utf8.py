from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fix(path: str, mapping: dict[int, str]) -> None:
    p = ROOT / path
    lines = p.read_bytes().decode("utf-8", errors="replace").splitlines()
    for line_no, content in mapping.items():
        lines[line_no - 1] = content
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


fix(
    "tests/test_prompt_record.py",
    {
        3: '"""inject 后 prompt 落盘（logs/prompts/{req_id}.txt）与 debug 日志配置。"""',
        19: '    return [{"role": "user", "content": "杭州天气怎么样？"}]',
    },
)
fix(
    "tests/test_response_record.py",
    {
        3: '"""上游模型 thinking/answer 落盘（logs/responses/{req_id}.txt）。"""',
        72: '    """prompts/{req_id}.txt 与 responses/{req_id}.txt 使用同一 req_id。"""',
    },
)
fix(
    "tests/test_transport_stress_sim.py",
    {
        3: '"""高强度模拟：不触达上游，验证 transport 腐化后 create_chat / 登录可恢复。"""',
        75: '    """最小 Qwen 客户端探针：复用 HttpTransportMixin，request 走脚本。"""',
        331: '    """Py3.8–3.14：变更模块可 import / compile。"""',
    },
)
print("fixed")
