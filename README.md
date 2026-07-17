# Rogator

Qwen AI 适配服务器 — 将阿里云通义千问 (Qwen) LLM 通过 OpenAI 和 Anthropic 兼容的 API 端点暴露。

## 功能特性

- OpenAI 兼容端点 (`/v1/chat/completions`)
- Anthropic 兼容端点 (`/v1/messages`)
- 自动账户轮换和 token 刷新
- 长文本自动分割和 OSS 上传
- 智能代理选择（延迟优化）
- TTS 语音合成支持
- 视频生成支持
- 工具调用（Function Calling）
- 思考模式（Thinking Mode）
- 联网搜索

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务器

```bash
python main.py [--port 8932] [--host 0.0.0.0] [--prelogin 1] [--log-level DEBUG]
```

参数说明：
- `--port`: 服务器端口（默认 8932）
- `--host`: 监听地址（默认 0.0.0.0）
- `--prelogin`: 预登录账户数量（默认 1）
- `--log-level`: 日志级别 DEBUG/INFO/WARNING/ERROR

### 使用示例

#### OpenAI 兼容格式

```bash
curl -X POST http://localhost:8932/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.7-max",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

#### Anthropic 兼容格式

```bash
curl -X POST http://localhost:8932/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.7-max",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## API 端点

### 核心端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 获取可用模型列表 |
| POST | `/v1/chat/completions` | OpenAI 聊天完成 |
| POST | `/v1/messages` | Anthropic 消息 |

### 管理端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/v1/admin/refresh_models` | 刷新模型列表 |
| POST | `/v1/admin/switch_session` | 切换到下一个会话 |
| GET | `/v1/admin/sessions` | 查看所有会话状态 |
| GET | `/v1/status` | 服务器详细状态 |
| GET | `/v1/capabilities` | 支持的功能列表 |

## 支持的模型

```python 
DEFAULT_MODELS: List[str] = [
    "qwen3.7-max",  #（默认）
    "qwen3.6-plus",
    "qwen3.5-plus",
    "qwen3.5-397b-a17b",
    "qwen3-max",
    "qwen3-max-2026-01-23",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-vl-30b-a3b",
    "qwen3-vl-32b",
    "qwen3-vl-plus",
    "qwen3-coder-plus",
    "qwen3-coder-30b-a3b-instruct",
    "qwen3-omni-flash",
    "qwen3-omni-flash-2025-12-01",
    "qwen2.5-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-omni-7b",
    "qwen2.5-coder-32b-instruct",
    "qwen-max-latest",
    "qwen-plus-2025-07-28",
    "qwen-plus-2025-09-11",
    "qwen-plus-2025-01-25",
    "qwen-turbo-2025-02-11",
]
```

## 环境变量

| 变量名 | 描述 | 必需 |
|--------|------|------|
| `QWEN_BX_UMIDTOKEN` | 覆盖 Baxia 反爬虫 token | 否 |
| `GENERALUSR` | MVP 测试用 Qwen 账户邮箱 | 仅测试 |
| `GENERALPWD` | MVP 测试用 Qwen 账户密码 | 仅测试 |

## 项目结构

```
.
├── main.py              # 入口点
├── state.py             # AppState, RequestScheduler
├── accounts.py          # 账户池
├── handlers/            # API 端点处理器
├── core/                # 主要 QwenClient（mixin 模式）
├── server/              # 简化的 QwenClient
├── mvp/                 # 烟雾测试脚本
├── achecker.py          # 合规检查器
└── amerger.py           # 文件合并工具
```

## 注意事项

- Qwen 不支持 system role，系统消息会被折叠到最后一条 user 消息中
- 所有出站请求禁用 SSL（`ssl=False`）
- 请求需要 `bx-ua` 和 `bx-umidtoken` 头（阿里巴巴指纹识别）

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件