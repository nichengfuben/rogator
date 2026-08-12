from __future__ import annotations

from pathlib import Path

FIXES = {
    77: 'APP_DESCRIPTION: str = "Qwen 长文本处理适配服务器"',
    89: '    """检查端口是否被占用。"""',
    100: '    """验证配置参数。"""',
    116: '    """打印启动信息横幅。"""',
    144: '    """单步关机；受整段硬 deadline 与各步 timeout 双重约束。"""',
    206: '    """启动 web 服务器并等待关机信号。"""',
    244: '    """服务器异步主入口（配置来自 config.toml + template/config.toml）。"""',
    280: '    """服务器主入口。"""',
}

def main() -> None:
    path = Path(__file__).resolve().parents[1] / "main.py"
    lines = path.read_bytes().splitlines()
    for idx, text in FIXES.items():
        lines[idx - 1] = text.encode("utf-8")
    path.write_bytes(b"\n".join(lines) + b"\n")
    print("fixed", len(FIXES), "lines")

if __name__ == "__main__":
    main()
