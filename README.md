# Rogator

多上游 AI 网关 — 通过 **OpenAI** 与 **Anthropic** 兼容 API 统一暴露 Qwen、DeepSeek、Cursor 等上游。

默认端口 **8932**，默认工具协议 **entml**（Qwen / DeepSeek）；Cursor 走上游原生 Agent 流。依赖 [echotools](https://pypi.org/project/echotools/) `2.3.98`。

**平台**：macOS / Linux / Windows  
**Python**：3.8 – 3.14

变更历史见 [CHANGELOG.md](CHANGELOG.md)。

## 功能特性

- **OpenAI 兼容**：`/v1/chat/completions`、模型列表、TTS、图片/视频生成（能力因上游而异）
- **Anthropic 兼容**：`/v1/messages` 及 `/anthropic/v1/*` 别名路径
- **多上游**：`config.toml` 的 `[upstream].enabled` 按需启用 `qwen`、`deepseek`、`cursor`
- **工具调用**：Qwen / DeepSeek 经 echotools `inject_fncall` 注入 entml；Cursor 使用上游原生 tool_call（不经 entml）
- **思考模式**：`thinking` / `reasoning_effort` 挡位；entml 思考、原生思考、纯原生上游（Cursor）分流，规则见 `persist/model_registry.jsonl`
- **Qwen / DeepSeek 账号池**：预登录、JWT 过期清理、限流/鉴权失败自动换号重试
- **LinUCB schedule**：`[schedule].enabled` 门控补登（空池强制 act；连续 skip 达上限强制补登）
- **Cursor**：Star Cursor API Key 拉号，无 Rogator 账号池；Token 写入 `persist/cursor/auth.toml`
- **长文本**：超长 prompt 尾部直发、前缀 OSS 附件（Qwen；可 `send_full_prompt=true` 关闭分割）
- **并发调度**：请求队列与并发上限可配

## 上游一览

| 上游 | 启用方式 | 账号 / 凭证 | 工具与思考 |
|------|----------|-------------|------------|
| **qwen** | 默认启用 | `persist/qwen/accounts.csv` + 会话池 | entml 思考 + entml 工具（多数模型） |
| **deepseek** | `enabled` 含 `deepseek` | `persist/deepseek/accounts.csv` + 会话池 | 原生思考 + entml 工具 |
| **cursor** | `enabled` 含 `cursor` | `configs/cursor.toml` → Star Cursor `api_keys` | 原生 Agent 流（thinking / answer / tool_call） |

各上游专属配置：从 `template/upstream/<name>.toml` 复制到 **`configs/<name>.toml`**（目录 gitignore，勿提交密钥）。

## 快速开始

### 1. 准备 Qwen 账号（启用 qwen 时必需）

`persist/qwen/accounts.csv`（根目录旧 `accounts.csv` 仅兼容回退）：

```csv
email,password,name
user@example.com,your-password,optional-label
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 测试
```

| 包 | 用途 |
|----|------|
| `aiohttp>=3.9.0` | HTTP 服务端与上游请求 |
| `echotools==2.3.98` | entml、日志、thinking_level |
| `h2>=4.1.0` | Cursor HTTP/2（Agent / GetUsableModels） |
| `typing-extensions>=4.7.0` | Python 3.8–3.10 类型兼容 |
| `tomli>=2.0.0` | Python 3.8–3.10 解析 TOML |

### 3. 配置

**全局**：首次启动从 `template/config.toml` 复制到根目录 `config.toml`（gitignore）。运行时以模板为缺省，本地文件**只覆盖已写键**，不写回磁盘。

```toml
[server]
version = "2.2.2"
port = 8932
host = "0.0.0.0"
prelogin = 32
login_interval = 15.0

[upstream]
# 按需追加 "deepseek", "cursor"
enabled = ["qwen"]

[models]
refresh_interval = 3600.0

[limits]
max_concurrent = 32
max_queue_size = 512
model_context_length = 256000
send_full_prompt = false
client_max_body_bytes = 33554432

[fncall]
record_prompt = false
record_response = false

[shutdown]
wait_active_requests = 3.0
total_timeout = 8.0
hard_exit_timeout = 25.0
```

**上游**：例如启用 Cursor 时：

```bash
cp template/upstream/cursor.toml configs/cursor.toml
# 编辑 configs/cursor.toml → [token_service].api_keys
```

并在 `[upstream].enabled` 中加入 `"cursor"`。

### 4. 启动与验证

```bash
python main.py
curl http://localhost:8932/health
curl http://localhost:8932/v1/models
```

启动后立即监听 HTTP；Qwen / DeepSeek 预登录在后台补至 `prelogin`；Cursor 在后台拉号并刷新 `GetUsableModels`。

## 模型注册表

对外 **API 模型 ID（外键）** 与上游内键、entml 开关由 **`persist/model_registry.jsonl`** 定义（可提交 git）：

```
外键:内键:uses_entml_thinking:uses_entml_tools
```

示例：

| 外键 | 内键 | entml 思考 | entml 工具 | 说明 |
|------|------|:----------:|:----------:|------|
| `qwen3-7-max` | `qwen3.7-max` | true | true | 默认模型 |
| `qwen3-8-max-preview` | `qwen3.8-max-preview` | false | true | 上游原生 Thinking |
| `deepseek-v4-pro` | `deepseek-v4-pro` | false | true | DeepSeek 原生思考 |
| `cursor-auto` | `default` | false | false | Cursor Auto |
| `composer-2-5-fast` | `composer-2.5-fast` | false | false | Cursor Composer |
| `gpt-4` | `gpt-4` | false | false | 别名（上游映射见 `configs/cursor.toml`） |

完整列表见仓库内 `persist/model_registry.jsonl`。`/v1/models` 返回注册表中的**外键**；运行时模型能力缓存位于 `persist/<upstream>/models.json`。

**注意**：请求必须使用外键，直接使用上游内键会返回 400。

## 使用示例

### OpenAI Chat（Qwen）

```bash
curl -X POST http://localhost:8932/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-7-max",
    "messages": [{"role": "user", "content": "你好"}],
    "thinking": "on"
  }'
```

### OpenAI Chat（Cursor）

```bash
curl -X POST http://localhost:8932/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cursor-auto",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

### Anthropic Messages

```bash
curl -X POST http://localhost:8932/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-7-max",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

带 OpenAI 格式 `tools` 时，Qwen / DeepSeek 经 entml 注入后转发；Cursor 将 tools 转为上游 MCP 格式。

## API 端点

### 核心

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health`, `/v1/health` | 健康检查 |
| GET | `/v1/models` | 模型列表（外键 + `think_efforts` 等） |
| POST | `/v1/chat/completions` | OpenAI 聊天 |
| POST | `/v1/messages` | Anthropic 消息 |
| POST | `/anthropic/v1/messages` | Anthropic 别名 |
| POST | `/v1/messages/count_tokens` | Token 估算 |
| POST | `/v1/audio/speech` | TTS（Qwen） |
| POST | `/v1/images/generations` | 图片/视频（Qwen） |

### 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/status` | 运行状态 |
| GET | `/v1/capabilities` | 合并上游能力 |
| GET | `/v1/admin/sessions` | 会话池概览 |
| POST | `/v1/admin/switch_session` | 手动换号 |
| POST | `/v1/admin/refresh_models` | 刷新模型缓存 |

## 思考模式

Rogator 按 `model_registry.jsonl` 决定每条链路如何处理 thinking / tools：

| `uses_entml_thinking` | `uses_entml_tools` | 行为 |
|:---------------------:|:------------------:|------|
| true | true | Qwen 典型：entml 思考 + entml 工具 |
| false | true | 上游原生 Thinking（如 `qwen3-8-max-preview`）+ entml 工具 |
| false | false | 全原生（Cursor）：handler 处理 `native: true` 的 thinking / answer / tool_call |

请求侧 `thinking` / `reasoning_effort` / `thinking_level` 归一化为 echotools 挡位：`none` | `low` | `medium` | `high` | `xhigh` | `max` | `auto`。

`GET /v1/models` 对支持思考的模型返回 `think_efforts`（`valid_efforts`、`default_effort`、`off_effort`），供 Kimi Code 等客户端刷新元数据。

**Kimi Code**：手写 `[models."alias"]` 时需自行配置 `support_efforts` / `default_effort`；仅 catalog 导入或 `refresh_on_start` 才会用 `/v1/models` 的 `think_efforts` 覆盖。

本地调试 prompt 注入：

```bash
python scripts/build_prompt_preview.py
```

## 会话与持久化

| 路径 | 说明 |
|------|------|
| `persist/model_registry.jsonl` | 外键 → 内键 → entml 开关（**可提交**） |
| `persist/qwen/sessions.json` | Qwen 会话池 |
| `persist/qwen/login_history.json` | Qwen 登录历史 |
| `persist/qwen/models.json` | Qwen 模型列表缓存 |
| `persist/deepseek/*` | DeepSeek 同上结构 |
| `persist/cursor/auth.toml` | Cursor Token（**gitignore**） |
| `persist/cursor/models.json` | Cursor 模型缓存（**gitignore**） |
| `configs/*.toml` | 各上游本地配置（**gitignore**） |

换号重试（Qwen）：`TokenExpiredError` → 封禁当前账号 → 切换 session → 最多 `max_retry_on_error` 次。

## 环境变量

| 变量 | 说明 |
|------|------|
| `QWEN_BX_UMIDTOKEN` | 覆盖 Baxia `bx-umidtoken` |
| `GENERALUSR` / `GENERALPWD` | 测试脚本账号（可选） |

## 项目结构

```
.
├── main.py
├── config.toml              # 全局运行时配置（gitignore）
├── configs/                 # 上游配置（gitignore；从 template/upstream/ 复制）
├── template/
│   ├── config.toml
│   ├── upstream_config.toml
│   └── upstream/            # qwen.toml, deepseek.toml, cursor.toml
├── src/
│   ├── state.py
│   ├── core/                # registry, dispatch, session pool, persist
│   ├── upstream/            # qwen/, deepseek/, cursor/
│   ├── handlers/            # OpenAI / Anthropic 适配
│   └── server/              # config, model registry, records, retry
├── persist/
│   ├── model_registry.jsonl
│   ├── qwen/
│   ├── deepseek/
│   └── cursor/              # 运行时生成（gitignore）
├── scripts/
├── tests/
└── CHANGELOG.md
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
python achecker.py   # 本地合规检查（可选）
```

CI：push/PR 到 `main` 跑 pytest（及仓库内 `achecker.py`）。

## 注意事项

- Qwen 不支持独立 system role；system 会折叠进最后一条 user 消息
- 出站 HTTP 对 Qwen 等使用 `ssl=False`；Cursor 走 HTTPS + HTTP/2
- Qwen 请求需 Baxia 指纹头（`bx-ua`、`bx-umidtoken`）
- 默认外键模型：`qwen3-7-max`（内键 `qwen3.7-max`）
- 未配置 DeepSeek 账号时，会话池维护日志为 debug 级别，不影响 Cursor / Qwen

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
