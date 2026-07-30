# Changelog

本文件记录 Rogator 的版本变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [Unreleased]

### Added

- **Cursor 上游**：Star Cursor 拉号/换号、`persist/cursor/auth.toml` 凭证、`GetUsableModels` 模型列表与 `persist/cursor/models.json` 磁盘缓存、Cursor Agent 双向流（原生 thinking / answer / tool_call，不经 entml 解析）。
- **`template/upstream/cursor.toml`**：Cursor 与 Star Cursor 配置模板；运行时复制到 `configs/cursor.toml`（gitignore）。
- **`persist/model_registry.jsonl`**：四段格式 `外键:内键:uses_entml_thinking:uses_entml_tools`；含 Qwen、DeepSeek、Cursor 外键映射。
- **原生上游事件分流**：registry 中 `uses_entml_tools=false` 的模型，handler 识别 `native: true` 事件并跳过 entml parser（OpenAI / Anthropic 路径均已适配）。
- **`tests/test_cursor_upstream.py`**：Cursor auth、模型缓存、converter、token 解析等单元测试。
- **依赖 `h2>=4.1.0`**：Cursor Agent / GetUsableModels HTTP/2。

### Changed

- **`model_registry.jsonl` 四段格式**：第三段为 entml 思考开关，第四段为 entml 工具开关；旧三字段行仍兼容（工具开关默认跟随思考开关）。
- **DeepSeek 会话池维护日志**：`No login-eligible accounts` 与 `Session pool replenish failed` 降为 `debug`，避免未启用 DeepSeek 账号时刷屏。
- **Cursor Token 拉号提示**：`Cursor 本地无 Token，自动拉号...` 降为 `debug`。
- **`.gitignore`**：忽略 `configs/`（含 API Key 的本地上游配置）与 `persist/cursor/`（Token 与模型缓存）。
- **配置布局**：删除仓库内 `configs/upstream_config.toml`；共享默认保留在 `template/upstream_config.toml`。
- **echotools** 固定为 `2.3.98`（entml 输出标签过滤等修复）。
- **README**：对齐多上游架构与 registry 说明；历史变更迁入本文件。

### Removed

- 仓库内可提交的 `configs/upstream_config.toml`（改由 `template/` 提供模板）。

---

## [2.2.2] - 2026-03

### Added

- **DeepSeek 上游**：独立账号池、会话预登录、WAF challenge 按 mute 处理（24h 登录封禁）。
- **共享会话池抽象**（`core/session/pool.py`）：Qwen / DeepSeek 共用 replenish / cleanup 逻辑。
- **persist 分桶迁移**（`core/persist/migrate.py`）：`sessions.json`、`login_history.json`、`models.json` 按 upstream 目录拆分。
- **`[upstream].enabled`**：在根 `config.toml` 声明启用的上游模块（替代旧 `roster.toml`）。

### Changed

- **模板目录**：`template/configs/` 重命名为 `template/upstream/`；新增 `template/upstream_config.toml` 作为各上游 TOML 共享缺省。
- **配置加载**：以 `template/config.toml` 为底，用户 `config.toml` 仅 overlay 已写键；`server.version` 不一致时启动告警。
- **模板默认值**：`prelogin=32`、`max_concurrent=32`、`max_queue_size=512` 等与生产环境对齐。

### Fixed

- DeepSeek WAF challenge 与 mute 账号同等对待，避免反复无效登录。

---

## [2.2.1] - 2026-02

### Added

- **并发流式**：取消单请求串行限制，支持多路并发 SSE。
- **可控关机**：`[shutdown]` 配置（`wait_active_requests`、`total_timeout`、`hard_exit_timeout`）；SIGINT 幂等、Windows  bounded HTTP teardown。
- **后台预登录**：HTTP 监听后立即接受请求，session 补登在后台 maintenance 循环进行。

### Fixed

- `tracked_request` 在 acquire 取消时仍释放 slot。
- 流式 usage 字段与 OpenAI / Anthropic 客户端合规（input/output token 来源修正）。
- `count_tokens` 估算与上游 usage 对齐。

---

## [2.2.0] - 2026-01

### Added

- **`src/` 布局**：运行时包迁入 `src/`，`main.py` 保留根目录入口。
- **响应录制**：`[fncall].record_response` 写入 `logs/responses/{req_id}.txt`（上游 thinking+answer，pre-entml）。
- **访问日志开关**：`[debug].access_log`（默认开启）。
- **配置模板版本号**：`template/config.toml` `[server].version` 与本地 config 对照提醒。

### Changed

- 全模型与附件分割阈值统一为 **256K** 上下文。
- echotools 多次 pin 升级（2.3.63 → 2.3.84 区间）：thinking 泄漏修复、fake history 过滤、stream 解析加固等。

### Fixed

- 优雅关机、上游 SSE `TimeoutError` 重试、Qwen usage 字段映射。
- 后台 session cleanup 循环自动 prelogin。

---

## [2.1.x] 及更早

- OpenAI / Anthropic 兼容 API、entml 工具调用注入、Qwen 多账号会话池与换号重试。
- 长文本分割 + OSS 上传、Baxia 指纹、thinking / reasoning_effort 挡位。
- Kimi Code 等客户端：`GET /v1/models` 返回 `think_efforts` 元数据。

[Unreleased]: https://github.com/nichengfuben/rogator/compare/main...HEAD
[2.2.2]: https://github.com/nichengfuben/rogator/releases/tag/v2.2.2
[2.2.1]: https://github.com/nichengfuben/rogator/releases/tag/v2.2.1
[2.2.0]: https://github.com/nichengfuben/rogator/releases/tag/v2.2.0
