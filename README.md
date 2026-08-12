# Rogator

多上游 AI 适配服务器 — 将 **Qwen** / **DeepSeek** / **Zen** / **Cursor** 等通过 OpenAI 与 Anthropic 兼容 API 暴露给客户端。

默认端口 **8932**，工具调用协议 **entml**，依赖 [echotools](https://pypi.org/project/echotools/) `>=2.4.5`（排除有缺陷的 `2.4.2`）。

**平台**：macOS / Linux / Windows  
**Python**：3.8 – 3.14

## 功能特性

- **OpenAI 兼容**：`/v1/chat/completions`、模型列表、TTS、图片/视频生成（视上游能力）
- **Anthropic 兼容**：`/v1/messages` 及 `/anthropic/v1/*` 别名路径
- **多上游**：`config.toml` 的 `[upstream].enabled` 选择加载 `upstream/<name>/`（如 `qwen`、`deepseek`、`zen`、`cursor`）
- **工具调用 (Function Calling)**：由 echotools `inject_fncall` 注入 entml 协议（按模型注册表开关）
- **思考模式**：支持 `thinking` / `reasoning_effort`；entml 模型与原生思考模型分流
- **历史 reasoning**：通过 `protocol_options.include_thinking_in_history` 由 echotools 渲染 `<entml:thinking>` 块
- **多账号会话池**：预登录、JWT 过期清理、限流/鉴权失败自动换号重试（按上游分桶）
- **长文本处理**：超长内容分割 + OSS 上传（Qwen）
- **并发调度**：请求队列与并发上限（`config.toml` 可配）

## 快速开始

### 1. 准备账号

按上游分文件加载（`core.session.accounts`）：

| 上游 | 路径 |
|------|------|
| Qwen | `persist/qwen/accounts.csv` |
| DeepSeek | `persist/deepseek/accounts.csv` |

CSV 列格式：

```csv
email,password,name
user@example.com,your-password,optional-label
```

至少为一个已启用上游配置有效账号。

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
| `aiohttp>=3.9.0` | HTTP 服务端与上游请求 |
| `echotools>=2.4.5,!=2.4.2` | entml 工具调用、tool id（`gen_tool_id` / `fix_tool_call_id`）、日志、thinking |
| `typing-extensions>=4.7.0` | Python 3.8–3.10 类型兼容 |
| `tomli>=2.0.0` | Python 3.8–3.10 解析 `config.toml`（3.11+ 使用 stdlib `tomllib`） |

### 3. 配置（可选）

首次启动会从 `template/config.toml` 复制到项目根目录 `config.toml`（本地文件 gitignore，不会提交）。
若仍存在旧路径 `config/config.toml`，会自动复制到根目录。

**加载策略（运行时）**：以 `template/config.toml` 为完整缺省，`config.toml` 里**只覆盖你写出的键**；未写出的节/键取自模板，**不使用代码内置默认值**。不会把模板合并写回你的 config 文件。`server.version` 与模板不一致时启动日志会提醒。

上游专属配置：`template/upstream/<name>.toml` → `configs/<name>.toml`（可覆盖）。

```toml
[server]
version = "2.2.2"
port = 8932
host = "0.0.0.0"
prelogin = 3          # session 池上限；空闲时仅保 1 个 warm session，有负载再扩池
login_interval = 15.0 # 连续预登录之间的间隔（秒）

[models]
refresh_interval = 3600.0  # 后台定时刷新模型列表（秒）

[retry]
max_retry_on_error = 3   # 限流/过期等可恢复错误时的换号重试次数

[limits]
max_concurrent = 8
max_queue_size = 1000
model_context_length = 256000
client_max_body_bytes = 33554432

[upstream]
# 启用的上游模块（对应 upstream/<name>/）
enabled = ["qwen"]
# 同时启用多个上游示例：
# enabled = ["qwen", "deepseek", "zen", "cursor"]

[fncall]
# record_prompt / print_prompt 任一为 true 时写入 logs/prompts/{req_id}.txt
# record_response = true 时写入 logs/responses/{req_id}.txt
record_response = false
# record_sse = true 时实时写入 logs/sse/{req_id}.sse（上游 SSE 原始流）
record_sse = false

[timeout]
request_total = 600.0
create_chat = 15.0
login = 30.0
prelogin = 120.0

[shutdown]
wait_active_requests = 3.0
total_timeout = 8.0
hard_exit_timeout = 25.0
```

监听地址、端口、预登录数量等均来自 `config.toml`（未写项取自 `template/config.toml`）。

### 4. 启动

```bash
python main.py
```

启动流程：加载/迁移会话 → 清理过期 session → **立即监听 HTTP**（后台仅 bootstrap 1 个 session）→ 模型列表在首批 session 就绪后刷新。

运行中后台每 60s 维护 session 池：空闲保持最少 warm session；有在途请求或排队时按负载扩至 `prelogin` 上限（受 `max_concurrent` 折中）。

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
    "model": "qwen3-7-max",
    "messages": [{"role": "user", "content": "你好"}],
    "thinking": "on"
  }'
```

请求体里的 `model` 必须是 **外键**（见下方模型注册表），不能直接传上游内键（如 `qwen3.7-max`）。

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

### 带工具调用

请求体携带 OpenAI 格式 `tools` 即可；服务端按注册表决定是否经 entml 注入后转发。

## API 端点

### 核心

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health`, `/v1/health` | 健康检查 |
| GET | `/v1/models` | OpenAI 模型列表（外键） |
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

## 模型注册表

`config/model_registry.jsonl`（可提交 git）每行：

```text
外键:内键:uses_entml_thinking:uses_entml_tools
```

- **外键**：客户端 `/v1/models` 与请求里使用的 ID（如 `qwen3-7-max`）
- **内键**：上游真实模型名（如 `qwen3.7-max`）
- **uses_entml_thinking**：是否走 entml 思考解析
- **uses_entml_tools**：是否走 entml 工具调用解析（缺省第三段时与思考开关相同）

示例：

```text
qwen3-7-max:qwen3.7-max:true:true
qwen3-8-max-preview:qwen3.8-max-preview:false:true
deepseek-v4-flash:deepseek-v4-flash:false:true
```

`GET /v1/models` 只列出注册表中、且内键仍在上游模型列表中的外键。

## 支持的模型（摘录）

运行时以 `/v1/models` 与各上游 `persist/<name>/models.json` 缓存为准。注册表中常见 Qwen 外键见 `config/model_registry.jsonl`；DeepSeek 如 `deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v4-vision`；Zen 免费模型如 `deepseek-v4-flash-free` / `mimo-v2.5-free` 等（由 zen upstream 自动同步）。

| 外键 | 默认 | entml 思考 | 说明 |
|------|:----:|:----------:|------|
| `qwen3-7-max` | **是** | 是 | |
| `qwen3-8-max-preview` | | 否 | 原生 Thinking，忽略 off/none |
| `deepseek-v4-flash` | | 否 | DeepSeek 上游 |

## 思考模式

Rogator 根据注册表 `uses_entml` 判断模型是否走 entml 思考协议：

| 映射值 | 行为 |
|--------|------|
| `true` | 上游 `Fast` + entml 解析 thinking（默认多数 Qwen3 模型） |
| `false` | 上游原生思考（如 `qwen3-8-max-preview`、部分 DeepSeek） |

请求侧 `thinking` / `reasoning_effort` / `thinking_level` 会归一化为 echotools 挡位：`none` | `low` | `medium` | `high` | `xhigh` | `max` | `auto`。`off`、`none`、`thinking: false` 等均视为关闭思考（`none`，**不注入** `<entml:thinking_mode>` / `max_thinking_length`）。若 conversation history 中已有 `<entml:thinking>` 历史块（`include_thinking_in_history`），仍会注入**强制不思考**的 `<thinking_behavior>`；无历史思考块时则不注入任何 thinking 相关内容。其中 `low`–`max` 注入 `<entml:thinking_mode>` 为挡位名（如 `medium`）与对应默认 `max_thinking_length`（12800 / 25600 / 64000 / 102400 / 134736）；仅 legacy `thinking_mode: on` 时注入 `on`；`auto` 注入 `auto` 模式且无默认长度。指引文案位于 `<thinking_behavior>` 块。历史 assistant 的 `reasoning` 由 echotools 在 `inject_fncall` 时按 `include_thinking_in_history` 渲染。

`GET /v1/models` 对支持思考的模型附带 `think_efforts`（`valid_efforts`、`default_effort`、`off_effort: none`），供 Kimi Code 等客户端刷新模型元数据。

**Kimi Code 手写 `[models."alias"]` 别名**不会自动从网关合并挡位，仍需在别名上配置 `support_efforts` / `default_effort`（以及 `off_effort = "none"` 以便选 Off 时发 `reasoning_effort: none`）；仅 catalog 导入或托管刷新才会用 `/v1/models` 里的 `think_efforts` 覆盖。

本地调试 prompt 注入结果：

```bash
python scripts/build_prompt_preview.py
```

## 会话与持久化

| 文件 | 说明 |
|------|------|
| `persist/<upstream>/sessions.json` | 该上游会话池、`current_index`、`blocked_accounts` |
| `persist/<upstream>/login_history.json` | 各账号最近一次成功登录时间（UTC+8） |
| `persist/<upstream>/accounts.csv` | 账号（gitignore，勿提交） |
| `persist/<upstream>/models.json` | 上游模型列表与能力 meta 缓存 |
| `config/model_registry.jsonl` | API 外键 → 上游内键 → entml 开关（可提交 git，支持热重载） |

换号重试逻辑见 `server/retry/session_retry.py`：捕获可恢复错误 → 封禁当前账号 → 切换下一可用 session → 最多重试 `max_retry_on_error` 次。

## 环境变量

| 变量 | 说明 |
|------|------|
| `QWEN_BX_UMIDTOKEN` | 覆盖 Baxia 反爬 `bx-umidtoken`（Qwen） |
| `GENERALUSR` / `GENERALPWD` | 测试脚本用账号（可选） |

## 项目结构

```
.
├── main.py                 # 入口：aiohttp 生命周期、预登录
├── config.toml             # 全局运行时配置（含 [upstream].enabled，gitignore）
├── configs/                # 各上游专属配置（qwen.toml、deepseek.toml 等）
├── template/
│   ├── config.toml         # 全局配置模板
│   ├── upstream_config.toml
│   └── upstream/           # 上游配置模板
├── src/
│   ├── path_setup.py
│   ├── state.py            # AppState、调度器；经 core 选上游客户端
│   ├── core/               # registry / dispatch / session 池
│   ├── upstream/           # qwen / deepseek / zen / cursor
│   ├── handlers/           # OpenAI / Anthropic 协议适配
│   └── server/             # 配置、格式、模型 registry、retry
├── config/
│   └── model_registry.jsonl
├── persist/
│   ├── qwen/
│   ├── deepseek/
│   └── zen/
├── scripts/
└── tests/
```

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## 注意事项

- Qwen 不支持独立 system role；system 消息会在发送前折叠进最后一条 user 消息
- 出站 HTTP 请求对部分上游使用 `ssl=False`
- Qwen 请求需携带阿里巴巴 Baxia 指纹头（`bx-ua`、`bx-umidtoken`）
- 默认模型以外键为准（见 `config/model_registry.jsonl` / `server/formats.py`）
- Zen / Cursor upstream 不记录 prompt / response / SSE 到 logs 目录
- Zen 节点遇到 429 / 502 / 连接失败时自动 mute 1 小时，错误日志脱敏不暴露代理 IP
- Qwen SSE 流中若出现非 assistant role（如 function），立即中断并触发客户端重试

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
