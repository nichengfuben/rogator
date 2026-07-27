# Rogator

Qwen AI 适配服务器 — 将阿里云通义千问 (Qwen) 通过 OpenAI 与 Anthropic 兼容 API 暴露给上游客户端。

默认端口 **8932**，工具调用协议 **entml**，依赖 [echotools](https://pypi.org/project/echotools/) `>=2.3.42`。

**平台**：macOS / Linux / Windows  
**Python**：3.8 – 3.14

## 功能特性

- **OpenAI 兼容**：`/v1/chat/completions`、模型列表、TTS、图片/视频生成
- **Anthropic 兼容**：`/v1/messages` 及 `/anthropic/v1/*` 别名路径
- **工具调用 (Function Calling)**：由 echotools `inject_fncall` 注入 entml 协议
- **思考模式**：支持 `thinking` / `reasoning_effort`；entml 模型与原生思考模型分流（见 `persist/model_entml_thinking.jsonl`）
- **历史 reasoning**：通过 `protocol_options.include_thinking_in_history` 由 echotools 渲染 `<entml:thinking>` 块
- **多账号会话池**：预登录、JWT 过期清理、限流/鉴权失败自动换号重试
- **长文本处理**：超长内容分割 + OSS 上传
- **并发调度**：请求队列与并发上限（`config.toml` 可配）

## 快速开始

### 1. 准备账号

账号从本地 `accounts.csv` 加载（`accounts.py` / `accounts.csv` 均已 gitignore）。CSV 列格式：

```csv
email,password,name
user@example.com,your-password,optional-label
```

至少配置一行有效账号，否则无法登录 Qwen。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

开发与测试额外安装：

```bash
pip install -r requirements-dev.txt
```

| 包 | 用途 |
|----|------|
| `aiohttp>=3.9.0` | HTTP 服务端与 Qwen 上游请求 |
| `echotools>=2.3.42` | entml 工具调用、日志、thinking_level / thinking_behavior（thinking 块置于 prompt 末尾） |
| `typing-extensions>=4.7.0` | Python 3.8–3.10 类型兼容 |
| `tomli>=2.0.0` | Python 3.8–3.10 解析 `config.toml`（3.11+ 使用 stdlib `tomllib`） |

### 3. 配置（可选）

编辑仓库根目录 `config.toml`：

```toml
[server]
port = 8932
host = "0.0.0.0"
prelogin = 3          # 启动时预登录账号数；运行中不足时自动补登

[retry]
max_retry_on_error = 3   # 限流/过期等可恢复错误时的换号重试次数

[limits]
max_concurrent = 8
max_queue_size = 1000
max_chars = 1024000
qwen_send_max_chars = 21750000   # chat.qwen.ai 网关 JSON body 硬限 ~21 MiB

[timeout]
request_total = 600.0
login = 30.0
prelogin = 120.0
```

CLI 参数会覆盖 `config.toml` 中的 `port` / `host` / `prelogin`。

### 4. 启动

```bash
python main.py [--port 8932] [--host 0.0.0.0] [--prelogin 3] [--log-level DEBUG]
```

启动流程：加载/迁移会话 → 清理过期 session → 预登录至 `prelogin` 数量 → 刷新模型列表 → 监听 HTTP。

### 5. 验证

```bash
curl http://localhost:8932/health
curl http://localhost:8932/v1/models
```

## 使用示例

### OpenAI Chat Completions

```bash
curl -X POST http://localhost:8932/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.7-max",
    "messages": [{"role": "user", "content": "你好"}],
    "thinking": "on"
  }'
```

### Anthropic Messages

```bash
curl -X POST http://localhost:8932/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.7-max",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 带工具调用

请求体携带 OpenAI 格式 `tools` 即可；服务端经 entml 注入后转发至 Qwen。

## API 端点

### 核心

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health`, `/v1/health` | 健康检查 |
| GET | `/v1/models` | OpenAI 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 聊天（流式/非流式） |
| POST | `/v1/messages` | Anthropic 消息 |
| POST | `/anthropic/v1/messages` | Anthropic 别名 |
| POST | `/v1/messages/count_tokens` | Token 估算 |
| POST | `/v1/audio/speech` | TTS |
| POST | `/v1/images/generations` | 图片/视频生成 |

### 管理 / 诊断

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/status` | 运行状态（会话数、调度器） |
| GET | `/v1/capabilities` | 能力列表 |
| GET | `/v1/admin/sessions` | 会话池概览 |
| POST | `/v1/admin/switch_session` | 手动切换当前账号 |
| POST | `/v1/admin/refresh_models` | 刷新模型缓存 |

## 支持的模型

内置默认列表见 `server/formats.py` 的 `DEFAULT_MODELS`；运行时以 `/v1/models` 或 `persist/qwen/models.json` 缓存为准（可通过管理端点刷新）。

| 模型 ID | 默认 | entml 思考 |
|---------|:----:|:----------:|
| `qwen3.8-max-preview` | | 否（原生 Thinking） |
| `qwen3.7-max` | **是** | 是 |
| `qwen3.6-plus` | | 是 |
| `qwen3.5-plus` | | 是 |
| `qwen3.5-397b-a17b` | | 是 |
| `qwen3-max` | | 是 |
| `qwen3-max-2026-01-23` | | 是 |
| `qwen3-235b-a22b` | | 是 |
| `qwen3-30b-a3b` | | 是 |
| `qwen3-vl-30b-a3b` | | 是 |
| `qwen3-vl-32b` | | 是 |
| `qwen3-vl-plus` | | 是 |
| `qwen3-coder-plus` | | 是 |
| `qwen3-coder-30b-a3b-instruct` | | 是 |
| `qwen3-omni-flash` | | 是 |
| `qwen3-omni-flash-2025-12-01` | | 是 |
| `qwen2.5-72b-instruct` | | 是 |
| `qwen2.5-vl-32b-instruct` | | 是 |
| `qwen2.5-omni-7b` | | 是 |
| `qwen2.5-coder-32b-instruct` | | 是 |
| `qwen-max-latest` | | 是 |
| `qwen-plus-2025-07-28` | | 是 |
| `qwen-plus-2025-09-11` | | 是 |
| `qwen-plus-2025-01-25` | | 是 |
| `qwen-turbo-2025-02-11` | | 是 |

未在 `persist/model_entml_thinking.jsonl` 中列出的模型，默认按 entml 思考处理（`uses_entml_thinking` 缺省为 `true`）。

## 思考模式

Rogator 根据 `persist/model_entml_thinking.jsonl` 判断模型是否走 entml 思考协议：

| 映射值 | 行为 |
|--------|------|
| `true` | 上游 `Fast` + entml 解析 thinking（默认多数 Qwen3 模型） |
| `false` | 上游原生 `Thinking`（如 `qwen3.8-max-preview`） |

请求侧 `thinking` / `reasoning_effort` / `thinking_level` 会归一化为 echotools 挡位：`none` | `low` | `medium` | `high` | `xhigh` | `max` | `auto`。其中 `low`–`max` 注入 `<entml:thinking_mode>on</entml:thinking_mode>` 与对应默认 `max_thinking_length`；`auto` 注入 `auto` 模式；`none` 不注入任何思考相关内容。指引文案位于 `<thinking_behavior>` 块。历史 assistant 的 `reasoning` 由 echotools 在 `inject_fncall` 时按 `include_thinking_in_history` 渲染。

本地调试 prompt 注入结果：

```bash
python scripts/build_prompt_preview.py
```

## 会话与持久化

| 文件 | 说明 |
|------|------|
| `persist/sessions.json` | 会话池、`current_index`、`account_index`、`blocked_accounts` |
| `persist/model_entml_thinking.jsonl` | 模型 → entml 思考映射（可提交 git） |
| `persist/qwen/models.json` | 模型列表缓存 |

首次启动若存在旧路径 `persist/qwen/sessions.json`，会自动迁移至 `persist/sessions.json`。

换号重试逻辑见 `server/session_retry.py`：捕获 `TokenExpiredError`（含限流）→ 封禁当前账号 → 切换下一可用 session → 最多重试 `max_retry_on_error` 次。

## 环境变量

| 变量 | 说明 |
|------|------|
| `QWEN_BX_UMIDTOKEN` | 覆盖 Baxia 反爬 `bx-umidtoken` |
| `GENERALUSR` / `GENERALPWD` | 测试脚本用账号（可选） |

## 项目结构

```
.
├── main.py                 # 入口：aiohttp 生命周期、预登录
├── config.toml             # 运行时配置（端口、prelogin、重试、限流）
├── state.py                # AppState、RequestScheduler、长文本分割
├── accounts.py             # 从 accounts.csv 加载账号（gitignore）
├── accounts.csv            # 本地账号 CSV（gitignore，需自行创建）
├── handlers/
│   ├── openai.py           # /v1/chat/completions
│   └── anthro.py           # /v1/messages
├── server/
│   ├── qwen_client.py      # 运行时 Qwen 客户端（登录、换号、prelogin）
│   ├── session_store.py    # sessions.json 读写与迁移
│   ├── session_retry.py    # 请求级换号重试
│   ├── model_thinking.py   # entml / 原生思考分流
│   ├── config.py           # config.toml 加载
│   └── formats.py          # ID 生成、响应格式、常量
├── persist/                # 运行时数据（部分 gitignore）
├── scripts/
│   └── build_prompt_preview.py
├── tests/                  # pytest 单元测试
└── core/                   # 遗留 mixin 客户端（TTS/视频等能力仍被引用）
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## 注意事项

- Qwen 不支持独立 system role；system 消息会在发送前折叠进最后一条 user 消息
- 出站 HTTP 请求使用 `ssl=False`
- 请求需携带阿里巴巴 Baxia 指纹头（`bx-ua`、`bx-umidtoken`）
- 默认模型：`qwen3.7-max`（见 `server/formats.py` 中 `DEFAULT_MODELS`）

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
