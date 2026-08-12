#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gratiator - Zen 平台独立服务器 (支持完整 OpenAI 格式，含 reasoning_effort / stream_options)"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import ssl
import sys
import tempfile
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from aiohttp import web

# ------------------------------------------------------------
# echotools 日志配置
# ------------------------------------------------------------
from echotools import configure, get_logger

configure(
    level="DEBUG",
    color=True,
    show_time=True,
    show_level=True,
    show_name=True,
    time_format="%Y-%m-%d %H:%M:%S",
)
logger = get_logger("gratiator")

# ============================================================
# 全局常量配置
# ============================================================

PORT: int = 8931
MAX_CONCURRENT: int = 32
MAX_QUEUE_SIZE: int = 1000

# 静态代理池
PROXY_POOL: List[Optional[str]] = [
    "http://127.0.0.1:7890",
    None
]

# 动态代理池文件
PROXY_POOL_FILE: str = "proxy_pool_.json"

DATA_FILE: str = "data/zen.json"

CONNECT_TIMEOUT: float = 60.0
FIRST_CHUNK_TIMEOUT: float = 60.0
STREAM_TOTAL_TIMEOUT: float = 600.0
STREAM_READ_TIMEOUT: float = 600.0
NON_STREAM_TIMEOUT: float = 180.0
MODELS_FETCH_TIMEOUT: float = 60.0

HEARTBEAT_INTERVAL: float = 15.0

# 节点切换：遇到代理/连接错误或限流错误时自动切换节点，但不重试（fail-fast）
MAX_NODE_SWITCH_ATTEMPTS: int = 1
NODE_SWITCH_DELAY: float = 0.3

FALLBACK_MODEL: str = "mimo-v2.5-free"
FALLBACK_MODEL_ENABLED: bool = True

BASE_URL: str = "https://opencode.ai/zen/v1"
CHAT_PATH: str = "/chat/completions"
MODELS_PATH: str = "/models"

DEBUG_LOG_BODY: bool = True
AUTO_REFRESH_MODELS: bool = False
DEFAULT_USER_AGENT: str = "opencode/latest"   # 将被客户端头覆盖，但保留默认值

# 默认模型列表（包含 ling-3.0-flash-free 等）
DEFAULT_MODELS: List[str] = [
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
    "laguna-s-2.1-free"
]

CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": True,
    "tools": True,
    "native_tools": True,
    "thinking": True,
    "search": False,
}

# 优雅关闭相关配置
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 10.0
SHUTDOWN_TOTAL_TIMEOUT: float = 15.0
RUNNER_SHUTDOWN_TIMEOUT: float = 10.0

# Zen API 密钥（公开 API 使用 "public"）
API_KEY: str = "public"

# ============================================================
# 代理池加载 / 合并
# ============================================================

def _normalize_proxy_url(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return "http://{}".format(raw)


def _load_dynamic_proxy_pool(path: str) -> List[str]:
    try:
        if not os.path.exists(path):
            logger.debug("Dynamic proxy pool file not found: %s", path)
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load dynamic proxy pool from %s: %s", path, e)
        return []

    if isinstance(data, dict):
        entries = data.get("proxies", [])
    elif isinstance(data, list):
        entries = data
    else:
        logger.warning("Unrecognized dynamic proxy pool format in %s", path)
        return []

    if not isinstance(entries, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            proxy_raw = entry.get("proxy") or entry.get("url") or ""
            latency = entry.get("latency")
        elif isinstance(entry, str):
            proxy_raw = entry
            latency = None
        else:
            continue

        proxy_url = _normalize_proxy_url(proxy_raw)
        if not proxy_url:
            continue

        try:
            latency_val = float(latency) if latency is not None else float("inf")
        except (TypeError, ValueError):
            latency_val = float("inf")

        parsed.append({"proxy": proxy_url, "latency": latency_val})

    parsed.sort(key=lambda x: x["latency"])
    seen = set()
    ordered: List[str] = []
    for item in parsed:
        p = item["proxy"]
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)

    return ordered


def _build_merged_proxy_pool(
    static_pool: List[Optional[str]],
    dynamic_pool: List[str],
) -> List[Optional[str]]:
    merged: List[Optional[str]] = list(static_pool)
    existing = set()
    for p in static_pool:
        if p is not None:
            existing.add(p)
    for p in dynamic_pool:
        if p in existing:
            continue
        merged.append(p)
        existing.add(p)
    return merged

# ============================================================
# 工具函数
# ============================================================

def normalize_model_name(model: str) -> str:
    if not model:
        return model
    if "-local" in model:
        base = model.replace("-local", "")
        logger.debug("Model normalized: %s -> %s", model, base)
        return base
    return model

def _gen_id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex[:24])

def _msg_id() -> str:
    return _gen_id("msg")

def _tool_id() -> str:
    return _gen_id("toolu")

def _normalize_tool_entry(t: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(t, dict):
        return None
    func = t.get("function")
    if isinstance(func, dict) and func.get("name"):
        return {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", "") or "",
                "parameters": func.get("parameters") or func.get("input_schema") or {},
            },
        }
    name = t.get("name")
    if name:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", "") or "",
                "parameters": t.get("input_schema") or t.get("parameters") or {},
            },
        }
    logger.warning("Dropping tool (no name): %s", json.dumps(t, ensure_ascii=False)[:150])
    return None

def normalize_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not tools or not isinstance(tools, list):
        return None
    normalized = [
        nt for nt in (_normalize_tool_entry(t) for t in tools) if nt is not None
    ]
    result = normalized or None
    if result:
        names = [
            t["function"]["name"] for t in result
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        ]
        logger.debug("normalize_tools: %d raw -> %d normalized: %s",
                     len(tools), len(result), names)
    else:
        logger.debug("normalize_tools: %d raw -> 0 normalized",
                     len(tools) if isinstance(tools, list) else 0)
    return result

def build_payload(
    messages: List[Dict[str, Any]],
    model: str = "",
    stream: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    **kw: Any,
) -> Dict[str, Any]:
    """
    构建上游请求体，支持 OpenAI 全部字段，包括 reasoning_effort、stream_options 等。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    # 标准字段
    if kw.get("temperature") is not None:
        payload["temperature"] = kw["temperature"]
    if kw.get("top_p") is not None:
        payload["top_p"] = kw["top_p"]
    if kw.get("max_tokens") is not None:
        payload["max_tokens"] = kw["max_tokens"]
    if kw.get("stop"):
        payload["stop"] = kw["stop"]
    if kw.get("tool_choice") is not None:
        payload["tool_choice"] = kw["tool_choice"]
    if kw.get("response_format") is not None:
        payload["response_format"] = kw["response_format"]
    if kw.get("reasoning_effort") is not None:
        payload["reasoning_effort"] = kw["reasoning_effort"]
    if kw.get("stream_options") is not None:
        payload["stream_options"] = kw["stream_options"]
    # 其他自定义
    if kw.get("thinking"):
        payload["thinking"] = True
    if kw.get("search"):
        payload["search"] = True

    # 工具
    if tools:
        payload["tools"] = tools

    # 保留未识别的额外字段（透传）
    extra = kw.get("extra", {})
    if extra:
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v

    return payload

def _build_headers(stream: bool, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """构建请求头，包含 Authorization 和 User-Agent，并合并 extra_headers。"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        "Authorization": f"Bearer {API_KEY}",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers

def _debug_log_body(status: int, text: str) -> None:
    if not DEBUG_LOG_BODY:
        return
    snippet = (text or "")[:1500]
    logger.debug("Upstream HTTP %d raw body: %s", status, snippet)

def _extract_error_info(data: Any) -> Optional[Dict[str, str]]:
    if not isinstance(data, dict):
        return None
    error_obj = None
    if data.get("type") == "error":
        error_obj = data.get("error", {})
    elif "error" in data:
        error_obj = data["error"]
    if error_obj is None:
        return None
    if not isinstance(error_obj, dict):
        return {"type": "", "message": str(error_obj)}
    message = error_obj.get("message", "") or str(error_obj)
    err_type = error_obj.get("type", "") or ""
    param = None
    metadata = error_obj.get("metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("raw")
        if isinstance(raw, str):
            try:
                raw_obj = json.loads(raw)
                raw_err = raw_obj.get("error") if isinstance(raw_obj, dict) else None
                if isinstance(raw_err, dict):
                    message = raw_err.get("message", message) or message
                    err_type = raw_err.get("type", "") or err_type
                    param = raw_err.get("param")
            except (json.JSONDecodeError, ValueError):
                pass
    result: Dict[str, str] = {"type": err_type, "message": message}
    if param:
        result["param"] = param
    return result

def _is_model_error(err_info: Dict[str, str]) -> bool:
    err_type = (err_info.get("type") or "").lower()
    err_msg = (err_info.get("message") or "").lower()
    if "modelerror" in err_type:
        return True
    if "not supported" in err_msg and "model" in err_msg:
        return True
    return False

def _is_validation_error(err_info: Dict[str, str]) -> bool:
    if err_info.get("param"):
        return True
    msg = (err_info.get("message") or "").lower()
    for kw in ("param incorrect", "missing function.name", "invalid_request",
               "invalid request", "bad request", "is missing"):
        if kw in msg:
            return True
    return False

def _is_proxy_error(error: Exception) -> bool:
    error_str = str(error).lower()
    proxy_keywords = [
        "cannot connect to host",
        "connection refused",
        "connection reset",
        "connection aborted",
        "proxy",
        "timed out",
        "timeout",
        "ssl",
        "certificate",
        "too many open connections",
    ]
    return any(kw in error_str for kw in proxy_keywords)

def _classify_http_error(
    status: int, err_info: Optional[Dict[str, str]], raw_text: str
) -> Exception:
    msg = err_info["message"] if err_info else (raw_text or "")[:300]
    if err_info and _is_model_error(err_info):
        return ModelNotSupportedError(msg)
    if status == 429:
        return RateLimitedError("HTTP 429 - {}".format(msg))
    if status == 400:
        return ProviderValidationError(msg)
    if status == 401:
        return ModelNotSupportedError("Model requires authentication (401): {}".format(msg))
    return UpstreamError("HTTP {} - {}".format(status, msg))

def _json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(
        data, status=status,
        dumps=lambda x: json.dumps(x, ensure_ascii=False),
    )

def _error_response(
    status: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> web.Response:
    return _json_response({
        "error": {"message": message, "type": error_type, "code": status}
    }, status=status)

async def _get_json(request: web.Request) -> Optional[Dict[str, Any]]:
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError):
        return None

def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

# ============================================================
# 自定义异常
# ============================================================

class UpstreamError(RuntimeError):
    pass

class RateLimitedError(UpstreamError):
    pass

class ModelNotSupportedError(RuntimeError):
    pass

class ProviderValidationError(RuntimeError):
    pass

class ProxyError(UpstreamError):
    pass

# ============================================================
# SSE 心跳机制
# ============================================================

class StreamHeartbeat:
    def __init__(self, resp: web.StreamResponse, interval: float = HEARTBEAT_INTERVAL) -> None:
        self._resp = resp
        self._interval = interval
        self._last_write = time.monotonic()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    def notify(self) -> None:
        self._last_write = time.monotonic()

    async def _run(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(1.0)
                if self._stopped:
                    return
                if time.monotonic() - self._last_write >= self._interval:
                    try:
                        await self._resp.write(b": heartbeat\n\n")
                        self._last_write = time.monotonic()
                    except Exception:
                        return
        except asyncio.CancelledError:
            pass

    async def __aenter__(self) -> "StreamHeartbeat":
        self._task = asyncio.ensure_future(self._run())
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

# ============================================================
# 节点管理器
# ============================================================

class NodeManager:
    def __init__(self, pool: List[Optional[str]], data_file: str) -> None:
        self._pool: List[Optional[str]] = pool if pool else [None]
        self._data_file = data_file
        self._current_index: int = 0
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            dir_name = os.path.dirname(os.path.abspath(self._data_file))
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = int(data.get("current_node_index", 0))
            if 0 <= idx < len(self._pool):
                self._current_index = idx
                logger.info("NodeManager: Restored node index %d (%s)",
                            idx, self._describe(idx))
            else:
                self._current_index = 0
                logger.info("NodeManager: Reset to index 0")
        except FileNotFoundError:
            self._current_index = 0
            logger.debug("NodeManager: No data file, starting from index 0.")
        except Exception as e:
            self._current_index = 0
            logger.warning("NodeManager: Load failed: %s. Reset to 0.", e)

    def _save(self) -> None:
        data = {
            "current_node_index": self._current_index,
            "current_node": self._describe(self._current_index),
            "updated_at": int(time.time()),
        }
        tmp_path = None
        try:
            dir_name = os.path.dirname(os.path.abspath(self._data_file))
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name or None, delete=False,
                suffix=".tmp", encoding="utf-8",
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                tmp_path = f.name
            os.replace(tmp_path, self._data_file)
            logger.debug("NodeManager: Saved state to %s", self._data_file)
        except Exception as e:
            logger.error("NodeManager: Save failed: %s", e)
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def _describe(self, index: int) -> str:
        node = self._pool[index]
        return "direct" if node is None else node

    @property
    def current_proxy(self) -> Optional[str]:
        return self._pool[self._current_index]

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_description(self) -> str:
        return self._describe(self._current_index)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def pool_snapshot(self) -> List[Optional[str]]:
        return list(self._pool)

    async def switch_next(self) -> str:
        async with self._lock:
            if self.pool_size <= 1:
                logger.debug("NodeManager: Only one node in pool, cannot switch")
                return self.current_description
            old_index = self._current_index
            self._current_index = (self._current_index + 1) % self.pool_size
            self._save()
            desc = self.current_description
            logger.info("NodeManager: Switched from index %d (%s) to %d (%s)",
                        old_index, self._describe(old_index),
                        self._current_index, desc)
            return desc

# ============================================================
# 并发调度器
# ============================================================

class QueueFullError(Exception):
    pass

class RequestScheduler:
    def __init__(self, max_concurrent: int, max_queue: int) -> None:
        self._max_concurrent = max_concurrent
        self._max_queue = max_queue
        self._semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrent) if max_concurrent != -1 else None
        )
        self._pending: int = 0
        self._lock = asyncio.Lock()
        self._shutting_down: bool = False

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def mark_shutting_down(self) -> None:
        self._shutting_down = True

    async def wait_idle(self, timeout: float = 30.0) -> bool:
        start = time.time()
        while self._pending > 0 and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)
        if self._pending > 0:
            logger.warning("Shutdown timeout, %d requests still pending", self._pending)
            return False
        logger.info("All pending requests completed")
        return True

    async def submit(self, coro_factory) -> Any:
        if self._shutting_down:
            raise QueueFullError("Server is shutting down, please try again later.")
        async with self._lock:
            if self._shutting_down:
                raise QueueFullError("Server is shutting down, please try again later.")
            if self._max_queue > 0 and self._pending >= self._max_queue:
                raise QueueFullError(
                    "Server is busy, please try again later. "
                    "Queue limit ({}) reached.".format(self._max_queue)
                )
            self._pending += 1
        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    return await coro_factory()
            else:
                return await coro_factory()
        finally:
            async with self._lock:
                self._pending -= 1

# ============================================================
# 活跃请求追踪
# ============================================================

class ActiveRequestTracker:
    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register(self, req_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            self._tasks[req_id] = task

    async def unregister(self, req_id: str) -> None:
        async with self._lock:
            self._tasks.pop(req_id, None)

    async def cancel_all(self) -> int:
        current = asyncio.current_task()
        async with self._lock:
            targets = [
                t for t in self._tasks.values()
                if t is not current and not t.done()
            ]
            for t in targets:
                t.cancel()
            self._tasks = {
                rid: t for rid, t in self._tasks.items()
                if t is current
            }
            return len(targets)

    @property
    def count(self) -> int:
        return len(self._tasks)

# ============================================================
# 全局应用状态
# ============================================================

class AppState:
    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()

        dynamic_pool = _load_dynamic_proxy_pool(PROXY_POOL_FILE)
        merged_pool = _build_merged_proxy_pool(PROXY_POOL, dynamic_pool)

        self.static_pool_size: int = len(PROXY_POOL)
        self.dynamic_pool_size: int = len(dynamic_pool)
        self.dynamic_pool_file: str = PROXY_POOL_FILE

        logger.info(
            "Proxy pool loaded: static=%d, dynamic=%d, merged=%d (source: %s)",
            self.static_pool_size, self.dynamic_pool_size, len(merged_pool),
            PROXY_POOL_FILE,
        )
        if self.dynamic_pool_size > 0:
            preview_count = min(3, self.dynamic_pool_size)
            preview = dynamic_pool[:preview_count]
            logger.info(
                "Dynamic proxy pool top %d (sorted by latency): %s%s",
                preview_count, preview,
                "" if self.dynamic_pool_size <= preview_count else " ...",
            )

        self.node_manager = NodeManager(merged_pool, DATA_FILE)
        self.scheduler = RequestScheduler(MAX_CONCURRENT, MAX_QUEUE_SIZE)
        self.tracker = ActiveRequestTracker()
        self.zen_client = ZenClient(self)
        self._shutdown_requested = False

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_requested or self.shutdown_event.is_set()

    def check_shutdown(self) -> None:
        if self.is_shutting_down:
            raise asyncio.CancelledError("Server is shutting down")

    async def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.shutdown_event.set()
        logger.info("AppState: Shutting down...")
        self.scheduler.mark_shutting_down()
        cancelled = await self.tracker.cancel_all()
        if cancelled > 0:
            logger.info("AppState: Cancelled %d active request(s)", cancelled)
            await asyncio.sleep(SHUTDOWN_CANCEL_GRACE)
        await self.scheduler.wait_idle(timeout=SHUTDOWN_WAIT_IDLE_TIMEOUT)
        logger.info("AppState: Shutdown complete")

_app_state: Optional[AppState] = None

def get_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state

# ============================================================
# SSE 检查函数
# ============================================================

def _has_valid_sse_event(data: bytes) -> bool:
    try:
        text = data.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                after_prefix = line[5:].strip()
                if after_prefix and after_prefix != "[DONE]":
                    return True
            elif line.startswith("event:"):
                after_prefix = line[6:].strip()
                if after_prefix:
                    return True
        return False
    except Exception:
        return False

def _log_sse_chunk(chunk: bytes, chunk_count: int, context: str = "") -> None:
    if not DEBUG_LOG_BODY:
        return
    try:
        chunk_text = chunk.decode("utf-8", errors="replace")
        if len(chunk_text) > 500:
            logger.debug("SSE chunk #%d %s (truncated): %s...",
                         chunk_count, context, chunk_text[:500])
        else:
            logger.debug("SSE chunk #%d %s: %s", chunk_count, context, chunk_text)
    except Exception:
        logger.debug("SSE chunk #%d %s: <binary>", chunk_count, context)

def _log_sse_event(data_content: str, chunk_count: int) -> None:
    if not DEBUG_LOG_BODY:
        return
    try:
        data_json = json.loads(data_content)
        data_str = json.dumps(data_json, ensure_ascii=False)
        if len(data_str) > 300:
            logger.debug("SSE data #%d (truncated): %s...", chunk_count, data_str[:300])
        else:
            logger.debug("SSE data #%d: %s", chunk_count, data_str)
    except Exception:
        if len(data_content) > 300:
            logger.debug("SSE data #%d (truncated): %s...", chunk_count, data_content[:300])
        else:
            logger.debug("SSE data #%d: %s", chunk_count, data_content)

# ============================================================
# Zen HTTP 客户端
# ============================================================

class ZenClient:
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._models: List[str] = list(DEFAULT_MODELS)
        self._models_fetch_time: float = 0
        self._models_cache_ttl: float = 300
        logger.info("HTTP backend: aiohttp (fail-fast with node-switch on 429/proxy-error)")

    def _make_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            ssl=_make_ssl_ctx(),
            use_dns_cache=False,
            limit=0,
            force_close=True,
        )
        return aiohttp.ClientSession(
            connector=connector,
            trust_env=False,
            timeout=aiohttp.ClientTimeout(connect=CONNECT_TIMEOUT),
            max_line_size=sys.maxsize,
            max_field_size=sys.maxsize,
        )

    async def fetch_models(self, use_cache: bool = True) -> List[str]:
        self._state.check_shutdown()
        now = time.time()
        if use_cache and self._models and (now - self._models_fetch_time) < self._models_cache_ttl:
            logger.debug("Using cached models (%d models)", len(self._models))
            return self._models

        proxy = self._state.node_manager.current_proxy
        url = "{}{}".format(BASE_URL, MODELS_PATH)
        try:
            data = await self._fetch_models_aiohttp(url, proxy)
            if data is None:
                return list(DEFAULT_MODELS)
            err_info = _extract_error_info(data)
            if err_info:
                logger.warning("fetch_models error: %s", err_info["message"])
                return list(DEFAULT_MODELS)
            model_data = data.get("data", [])
            if isinstance(model_data, list):
                models = [
                    m.get("id", "") for m in model_data
                    if isinstance(m, dict) and m.get("id")
                ]
                free = [m for m in models if m.endswith("-free")]
                if free:
                    self._models = free
                    self._models_fetch_time = now
                    logger.debug("Fetched %d free models", len(free))
                    return free
                if models:
                    self._models = models
                    self._models_fetch_time = now
                    logger.info("Fetched %d models (no free models found)", len(models))
                    return models
            return list(DEFAULT_MODELS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("fetch_models exception: %s", e, exc_info=True)
            return list(DEFAULT_MODELS)

    async def _fetch_models_aiohttp(
        self, url: str, proxy: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        async with self._make_session() as session:
            kw: Dict[str, Any] = {
                "timeout": aiohttp.ClientTimeout(total=MODELS_FETCH_TIMEOUT),
                "headers": _build_headers(False),
            }
            if proxy:
                kw["proxy"] = proxy
            async with session.get(url, **kw) as resp:
                if resp.status != 200:
                    logger.warning("fetch_models aiohttp status: %d", resp.status)
                    return None
                return await resp.json()

    async def _do_request(
        self,
        proxy: Optional[str],
        payload: Dict[str, Any],
        stream: bool,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[bytes, None]:
        proxy_desc = proxy if proxy else "direct"
        logger.debug("Upstream request: proxy=%s, model=%s, stream=%s",
                     proxy_desc, payload.get("model"), stream)
        async for chunk in self._do_request_aiohttp(proxy, payload, stream, extra_headers):
            yield chunk

    async def _do_request_aiohttp(
        self,
        proxy: Optional[str],
        payload: Dict[str, Any],
        stream: bool,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[bytes, None]:
        url = "{}{}".format(BASE_URL, CHAT_PATH)
        if stream:
            timeout = aiohttp.ClientTimeout(
                total=STREAM_TOTAL_TIMEOUT,
                connect=CONNECT_TIMEOUT,
                sock_read=STREAM_READ_TIMEOUT,
            )
        else:
            timeout = aiohttp.ClientTimeout(
                total=NON_STREAM_TIMEOUT,
                connect=CONNECT_TIMEOUT,
            )

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        logger.debug("Request body: %s", body_bytes[:500] if len(body_bytes) > 500 else body_bytes)

        headers = _build_headers(stream, extra_headers)

        try:
            async with self._make_session() as session:
                kw: Dict[str, Any] = {
                    "headers": headers,
                    "data": body_bytes,
                    "timeout": timeout,
                }
                if proxy:
                    kw["proxy"] = proxy
                    logger.debug("_do_request_aiohttp: using proxy %s", proxy)

                async with session.post(url, **kw) as resp:
                    logger.debug("Upstream response status: %d", resp.status)
                    if resp.status != 200:
                        err_text = await resp.text()
                        _debug_log_body(resp.status, err_text)
                        err_info = None
                        try:
                            err_data = json.loads(err_text)
                            err_info = _extract_error_info(err_data)
                        except (json.JSONDecodeError, ValueError):
                            pass

                        if resp.status == 401:
                            msg = err_info.get("message", err_text) if err_info else err_text
                            raise ModelNotSupportedError("Model requires authentication (401): {}".format(msg))

                        raise _classify_http_error(resp.status, err_info, err_text)

                    if not stream:
                        data = await resp.json()
                        logger.debug("Non-stream response received: %s",
                                     json.dumps(data, ensure_ascii=False)[:500])

                        err_info = _extract_error_info(data)
                        if err_info:
                            if _is_model_error(err_info):
                                raise ModelNotSupportedError(err_info["message"])
                            if _is_validation_error(err_info):
                                raise ProviderValidationError(err_info["message"])
                            raise UpstreamError("API error: {}".format(err_info["message"]))

                        if data.get("error"):
                            error_obj = data["error"]
                            if isinstance(error_obj, dict):
                                error_msg = error_obj.get("message", "unknown error")
                            else:
                                error_msg = str(error_obj)
                            raise UpstreamError("API error: {}".format(error_msg))

                        logger.debug("Non-stream response OK, size: %d bytes", len(body_bytes))
                        yield json.dumps(data, ensure_ascii=False).encode("utf-8")
                        return

                    # 流式响应
                    received_data = False
                    has_valid_event = False
                    chunk_buffer = b""
                    chunk_count = 0

                    async for chunk in resp.content.iter_any():
                        if chunk:
                            received_data = True
                            chunk_count += 1
                            _log_sse_chunk(chunk, chunk_count, "(stream #{})".format(chunk_count))

                            chunk_buffer += chunk

                            if _has_valid_sse_event(chunk_buffer):
                                has_valid_event = True
                                if DEBUG_LOG_BODY:
                                    try:
                                        chunk_text = chunk_buffer.decode("utf-8", errors="replace")
                                        for line in chunk_text.split("\n"):
                                            line = line.strip()
                                            if line.startswith("data:"):
                                                data_content = line[5:].strip()
                                                if data_content and data_content != "[DONE]":
                                                    _log_sse_event(data_content, chunk_count)
                                            elif line.startswith("event:"):
                                                logger.debug("SSE event: %s", line[6:].strip())
                                    except Exception as e:
                                        logger.debug("Failed to parse SSE event: %s", e)

                            yield chunk

                    if received_data and not has_valid_event:
                        logger.warning("Stream received %d chunks but no valid SSE events",
                                       chunk_count)
                        raise UpstreamError("Empty stream response: no valid events received")

                    if not received_data:
                        logger.warning("Stream received no data at all")
                        raise UpstreamError("Empty stream response: no data received")

                    logger.debug("Stream completed successfully, received %d chunks", chunk_count)

        except aiohttp.ClientProxyConnectionError as e:
            raise ProxyError("Proxy connection failed: {}".format(e))
        except aiohttp.ClientConnectorError as e:
            if proxy or "proxy" in str(e).lower():
                raise ProxyError("Proxy connection error: {}".format(e))
            raise UpstreamError("Connection error: {}".format(e))
        except aiohttp.ServerTimeoutError as e:
            if proxy:
                raise ProxyError("Proxy timeout: {}".format(e))
            raise UpstreamError("Request timeout: {}".format(e))
        except (ModelNotSupportedError, ProviderValidationError, RateLimitedError, UpstreamError):
            raise
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            if proxy:
                raise ProxyError("Proxy timeout: {}".format(e))
            raise UpstreamError("Request timeout: {}".format(e))
        except Exception as e:
            if _is_proxy_error(e):
                raise ProxyError("Proxy error: {}".format(e))
            raise UpstreamError("Request error: {}".format(e))

    async def chat_completion(
        self,
        payload: Dict[str, Any],
        _fallback_applied: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        执行请求，支持自动节点切换（但不重试）。
        遇到 ProxyError/RateLimitedError 切换节点，其他错误直接失败或 fallback。
        """
        self._state.check_shutdown()

        stream = payload.get("stream", False)
        current_model = payload.get("model", "")

        # 模型预检查
        if FALLBACK_MODEL_ENABLED and not _fallback_applied and current_model != FALLBACK_MODEL:
            try:
                self._state.check_shutdown()
                if AUTO_REFRESH_MODELS:
                    available_models = await self.fetch_models(use_cache=False)
                    logger.debug("Auto-refresh: fetched %d models from upstream", len(available_models))
                else:
                    available_models = DEFAULT_MODELS
                    logger.debug("Using DEFAULT_MODELS for pre-check (%d models)", len(DEFAULT_MODELS))

                base_model = current_model.replace("-local", "")
                if base_model not in available_models:
                    logger.debug("Model '%s' not in available list, falling back to '%s'",
                                current_model, FALLBACK_MODEL)
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = FALLBACK_MODEL
                    async for chunk in self.chat_completion(
                        fallback_payload, _fallback_applied=True, extra_headers=extra_headers
                    ):
                        yield chunk
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Failed to fetch models for pre-check: %s", e)

        # 处理 -local 后缀
        if "-local" in current_model and not _fallback_applied:
            base_model = current_model.replace("-local", "")
            logger.info("Model %s contains '-local', auto-falling back to %s",
                        current_model, base_model)
            fallback_payload = dict(payload)
            fallback_payload["model"] = base_model
            async for chunk in self.chat_completion(
                fallback_payload, _fallback_applied=True, extra_headers=extra_headers
            ):
                yield chunk
            return

        # 单次执行，失败时切换节点（为后续请求）
        max_attempts = min(MAX_NODE_SWITCH_ATTEMPTS, self._state.node_manager.pool_size)

        for attempt in range(1, max_attempts + 1):
            self._state.check_shutdown()
            proxy = self._state.node_manager.current_proxy
            desc = self._state.node_manager.current_description

            logger.debug("Executing request via %s (attempt %d/%d)", desc, attempt, max_attempts)

            try:
                async for chunk in self._do_request(proxy, payload, stream, extra_headers):
                    yield chunk
                logger.debug("Request completed via %s", desc)
                return

            except (ProxyError, RateLimitedError) as e:
                if attempt >= max_attempts:
                    logger.error("%s via %s (final attempt): %s", e.__class__.__name__, desc, e)
                    raise
                logger.warning("%s via %s (attempt %d/%d): %s — switching node",
                               e.__class__.__name__, desc, attempt, max_attempts, e)
                await self._state.node_manager.switch_next()
                await asyncio.sleep(NODE_SWITCH_DELAY)
                continue

            except ModelNotSupportedError as e:
                logger.warning("Model not supported via %s: %s", desc, e)
                if (
                    FALLBACK_MODEL_ENABLED
                    and not _fallback_applied
                    and current_model != FALLBACK_MODEL
                ):
                    logger.info("Falling back to model: %s", FALLBACK_MODEL)
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = FALLBACK_MODEL
                    async for chunk in self.chat_completion(
                        fallback_payload, _fallback_applied=True, extra_headers=extra_headers
                    ):
                        yield chunk
                    return
                raise

            except ProviderValidationError as e:
                logger.warning("Provider validation error via %s: %s", desc, e)
                raise

            except asyncio.CancelledError:
                raise

            except Exception as e:
                logger.error("Request failed via %s: %s", desc, e)
                raise

# ============================================================
# 流式写入辅助
# ============================================================

async def _sse_write(resp: web.StreamResponse, event: str, data: Any) -> None:
    try:
        await resp.write(
            "event: {}\ndata: {}\n\n".format(
                event, json.dumps(data, ensure_ascii=False)
            ).encode("utf-8")
        )
    except Exception:
        pass

# ============================================================
# OpenAI Responses API 兼容
# ============================================================

def _responses_content_to_chat(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    converted: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            converted.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type", "")

        if part_type in ("input_text", "output_text", "text"):
            converted.append({"type": "text", "text": part.get("text", "")})
            continue

        if part_type in ("input_image", "image_url"):
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url_obj = dict(image_url)
            else:
                image_url_obj = {"url": image_url or part.get("url", "")}
                if part.get("detail"):
                    image_url_obj["detail"] = part["detail"]
            if image_url_obj.get("url"):
                converted.append({"type": "image_url", "image_url": image_url_obj})
            continue

        if part_type == "input_file":
            file_url = part.get("file_url", "")
            filename = part.get("filename", "")
            file_id = part.get("file_id", "")
            file_description = file_url or file_id
            if file_description:
                converted.append({
                    "type": "text",
                    "text": "[file: {}{}]".format(
                        filename,
                        " ({})".format(file_description) if file_description else "",
                    ),
                })
            continue

    return converted if converted else ""


def _responses_tool_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    if output is None:
        return ""
    return str(output)


def _responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type == "function":
        name = tool_choice.get("name")
        if not name:
            function_obj = tool_choice.get("function")
            if isinstance(function_obj, dict):
                name = function_obj.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def _responses_input_to_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    instructions = body.get("instructions")
    if instructions:
        instruction_text = instructions if isinstance(instructions, str) else json.dumps(instructions, ensure_ascii=False)
        messages.append({"role": "system", "content": instruction_text})

    input_data = body.get("input")

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    if not isinstance(input_data, list):
        raise ValueError("input is required and must be a string or array")

    for item in input_data:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or _gen_id("call")
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": arguments},
                }],
            })
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id") or ""
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _responses_tool_output_to_text(item.get("output")),
            })
            continue

        if item_type in ("message", "") or item.get("role"):
            role = item.get("role", "user")
            if role not in ("system", "developer", "user", "assistant", "tool"):
                role = "user"
            if role == "developer":
                role = "system"
            message: Dict[str, Any] = {
                "role": role,
                "content": _responses_content_to_chat(item.get("content", "")),
            }
            if role == "tool":
                message["tool_call_id"] = item.get("tool_call_id") or item.get("call_id") or ""
            messages.append(message)
            continue

        if item_type in ("input_text", "text"):
            messages.append({"role": "user", "content": item.get("text", "")})
            continue

        if item_type == "input_image":
            messages.append({"role": "user", "content": _responses_content_to_chat([item])})
            continue

    if not messages:
        raise ValueError("input did not contain any supported message")

    return messages


def _build_responses_payload(body: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model_raw = body.get("model", "")
    if not isinstance(model_raw, str) or not model_raw:
        raise ValueError("model is required")

    model = normalize_model_name(model_raw)
    messages = _responses_input_to_messages(body)
    tools = normalize_tools(body.get("tools"))

    extra = body.get("extra_body") or body.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}

    thinking_config = body.get("thinking")
    if isinstance(thinking_config, dict):
        thinking = (
            thinking_config.get("type") == "enabled"
            or bool(thinking_config.get("enabled", False))
        )
    else:
        thinking = bool(thinking_config)

    stream = bool(body.get("stream", False))

    payload = build_payload(
        messages=messages,
        model=model,
        stream=stream,
        tools=tools,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=(
            body.get("max_output_tokens")
            if body.get("max_output_tokens") is not None
            else body.get("max_tokens")
        ),
        stop=body.get("stop"),
        tool_choice=_responses_tool_choice_to_chat(body.get("tool_choice")),
        thinking=bool(thinking or extra.get("thinking", False)),
        search=bool(body.get("search", False) or extra.get("search", False)),
        reasoning_effort=body.get("reasoning_effort"),
        stream_options=body.get("stream_options"),
        extra=extra,
    )

    context = {"request": body, "requested_model": model}
    return payload, context


def _chat_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("text", "output_text", "input_text"):
            parts.append(str(part.get("text", "")))
        elif part.get("text") is not None:
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _normalize_function_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments or "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    if arguments is None:
        return "{}"
    return str(arguments)


def _convert_chat_to_response(
    chat_response: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    request_body = context.get("request", {})
    requested_model = context.get("requested_model", "")

    response_id = _gen_id("resp")
    created_at = int(time.time())
    output: List[Dict[str, Any]] = []

    choices = chat_response.get("choices")
    if not isinstance(choices, list):
        choices = []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}

    content_text = _chat_content_to_text(message.get("content"))
    refusal = message.get("refusal")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []

    if content_text or refusal or not tool_calls:
        content_parts: List[Dict[str, Any]] = []
        if content_text or not refusal:
            content_parts.append({
                "type": "output_text", "text": content_text,
                "annotations": [], "logprobs": [],
            })
        if refusal:
            content_parts.append({"type": "refusal", "refusal": str(refusal)})
        output.append({
            "id": _msg_id(), "type": "message", "status": "completed",
            "role": "assistant", "content": content_parts,
        })

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_obj = tool_call.get("function")
        if not isinstance(function_obj, dict):
            function_obj = {}
        call_id = tool_call.get("id") or tool_call.get("call_id") or _gen_id("call")
        output.append({
            "id": _gen_id("fc"), "type": "function_call", "status": "completed",
            "arguments": _normalize_function_arguments(function_obj.get("arguments")),
            "call_id": call_id,
            "name": function_obj.get("name", ""),
        })

    finish_reason = choice.get("finish_reason")
    incomplete = finish_reason in ("length", "max_tokens", "content_filter")

    usage_raw = chat_response.get("usage")
    if not isinstance(usage_raw, dict):
        usage_raw = {}

    input_tokens = int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0)
    output_tokens = int(usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0)
    total_tokens = int(usage_raw.get("total_tokens") or (input_tokens + output_tokens))

    prompt_details = usage_raw.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    completion_details = usage_raw.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}

    status = "incomplete" if incomplete else "completed"

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if finish_reason in ("length", "max_tokens")
            else ({"reason": "content_filter"} if finish_reason == "content_filter" else None)
        ),
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "max_tool_calls": request_body.get("max_tool_calls"),
        "model": chat_response.get("model") or requested_model,
        "output": output,
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "previous_response_id": request_body.get("previous_response_id"),
        "prompt": request_body.get("prompt"),
        "reasoning": request_body.get("reasoning"),
        "safety_identifier": request_body.get("safety_identifier"),
        "service_tier": request_body.get("service_tier", "default"),
        "store": bool(request_body.get("store", False)),
        "temperature": request_body.get("temperature"),
        "text": request_body.get("text", {"format": {"type": "text"}}),
        "tool_choice": request_body.get("tool_choice", "auto"),
        "tools": request_body.get("tools", []),
        "top_logprobs": request_body.get("top_logprobs", 0),
        "top_p": request_body.get("top_p"),
        "truncation": request_body.get("truncation", "disabled"),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0)},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0)},
            "total_tokens": total_tokens,
        },
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata", {}),
    }


async def _responses_write_event(
    resp: web.StreamResponse,
    event_name: str,
    data: Dict[str, Any],
) -> None:
    event_data = json.dumps(data, ensure_ascii=False)
    await resp.write(
        "event: {}\ndata: {}\n\n".format(event_name, event_data).encode("utf-8")
    )


async def _do_responses_real_stream(
    state: AppState,
    req_id: str,
    payload: Dict[str, Any],
    context: Dict[str, Any],
    resp: web.StreamResponse,
    heartbeat: StreamHeartbeat,
) -> None:
    request_body = context.get("request", {})
    requested_model = context.get("requested_model", "")
    response_id = _gen_id("resp")
    created_at = int(time.time())

    seq = 0
    def next_seq() -> int:
        nonlocal seq
        v = seq
        seq += 1
        return v

    in_progress_shell = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "max_tool_calls": request_body.get("max_tool_calls"),
        "model": requested_model,
        "output": [],
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "previous_response_id": request_body.get("previous_response_id"),
        "prompt": request_body.get("prompt"),
        "reasoning": request_body.get("reasoning"),
        "service_tier": request_body.get("service_tier", "default"),
        "store": bool(request_body.get("store", False)),
        "temperature": request_body.get("temperature"),
        "text": request_body.get("text", {"format": {"type": "text"}}),
        "tool_choice": request_body.get("tool_choice", "auto"),
        "tools": request_body.get("tools", []),
        "top_logprobs": request_body.get("top_logprobs", 0),
        "top_p": request_body.get("top_p"),
        "truncation": request_body.get("truncation", "disabled"),
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata", {}),
    }

    await _responses_write_event(resp, "response.created", {
        "type": "response.created",
        "response": in_progress_shell,
        "sequence_number": next_seq(),
    })
    heartbeat.notify()

    await _responses_write_event(resp, "response.in_progress", {
        "type": "response.in_progress",
        "response": in_progress_shell,
        "sequence_number": next_seq(),
    })
    heartbeat.notify()

    text_buf = ""
    text_item_id: Optional[str] = None
    text_output_index: Optional[int] = None
    text_content_started = False

    tool_bufs: Dict[int, Dict[str, Any]] = {}
    tool_order: List[int] = []

    next_output_index = 0
    finish_reason: Optional[str] = None
    model_name = requested_model

    line_buffer = b""

    task = asyncio.current_task()
    await state.tracker.register(req_id, task)
    try:
        async for chunk in state.zen_client.chat_completion(payload):
            heartbeat.notify()
            line_buffer += chunk

            while b"\n" in line_buffer:
                line_bytes, line_buffer = line_buffer.split(b"\n", 1)
                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                err_info = _extract_error_info(obj)
                if err_info:
                    raise UpstreamError(err_info["message"])

                if obj.get("model"):
                    model_name = obj["model"]

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                content_delta = delta.get("content")
                if content_delta:
                    if text_item_id is None:
                        text_item_id = _msg_id()
                        text_output_index = next_output_index
                        next_output_index += 1
                        await _responses_write_event(resp, "response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": text_output_index,
                            "item": {
                                "id": text_item_id, "type": "message",
                                "status": "in_progress", "role": "assistant", "content": [],
                            },
                            "sequence_number": next_seq(),
                        })
                        heartbeat.notify()
                    if not text_content_started:
                        text_content_started = True
                        await _responses_write_event(resp, "response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": text_item_id,
                            "output_index": text_output_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text", "text": "",
                                "annotations": [], "logprobs": [],
                            },
                            "sequence_number": next_seq(),
                        })
                        heartbeat.notify()
                    text_buf += content_delta
                    await _responses_write_event(resp, "response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": text_item_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "delta": content_delta,
                        "logprobs": [],
                        "sequence_number": next_seq(),
                    })
                    heartbeat.notify()

                tool_delta_list = delta.get("tool_calls")
                if tool_delta_list:
                    for td in tool_delta_list:
                        if not isinstance(td, dict):
                            continue
                        idx = td.get("index", 0)
                        if idx not in tool_bufs:
                            item_id = _gen_id("fc")
                            out_idx = next_output_index
                            next_output_index += 1
                            tool_bufs[idx] = {
                                "id": "", "name": "", "arguments": "",
                                "item_id": item_id, "output_index": out_idx,
                                "header_sent": False,
                            }
                            tool_order.append(idx)

                        buf = tool_bufs[idx]
                        if td.get("id"):
                            buf["id"] = td["id"]
                        func = td.get("function") or {}
                        if func.get("name"):
                            buf["name"] = func["name"]

                        if not buf["header_sent"]:
                            buf["header_sent"] = True
                            await _responses_write_event(resp, "response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": buf["output_index"],
                                "item": {
                                    "id": buf["item_id"], "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": buf["id"] or buf["item_id"],
                                    "name": buf["name"], "arguments": "",
                                },
                                "sequence_number": next_seq(),
                            })
                            heartbeat.notify()

                        new_args = func.get("arguments")
                        if new_args:
                            buf["arguments"] += new_args
                            await _responses_write_event(resp, "response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": buf["item_id"],
                                "output_index": buf["output_index"],
                                "delta": new_args,
                                "sequence_number": next_seq(),
                            })
                            heartbeat.notify()

        # 收尾
        output_items: Dict[int, Dict[str, Any]] = {}

        if text_item_id is not None:
            await _responses_write_event(resp, "response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": text_item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "text": text_buf,
                "logprobs": [],
                "sequence_number": next_seq(),
            })
            final_part = {"type": "output_text", "text": text_buf, "annotations": [], "logprobs": []}
            await _responses_write_event(resp, "response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": text_item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "part": final_part,
                "sequence_number": next_seq(),
            })
            final_message_item = {
                "id": text_item_id, "type": "message", "status": "completed",
                "role": "assistant", "content": [final_part],
            }
            await _responses_write_event(resp, "response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": text_output_index,
                "item": final_message_item,
                "sequence_number": next_seq(),
            })
            output_items[text_output_index] = final_message_item
            heartbeat.notify()

        for idx in tool_order:
            buf = tool_bufs[idx]
            args = buf["arguments"] or "{}"
            call_id = buf["id"] or buf["item_id"]
            await _responses_write_event(resp, "response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": buf["item_id"],
                "output_index": buf["output_index"],
                "arguments": args,
                "sequence_number": next_seq(),
            })
            final_tool_item = {
                "id": buf["item_id"], "type": "function_call", "status": "completed",
                "call_id": call_id, "name": buf["name"], "arguments": args,
            }
            await _responses_write_event(resp, "response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": buf["output_index"],
                "item": final_tool_item,
                "sequence_number": next_seq(),
            })
            output_items[buf["output_index"]] = final_tool_item
            heartbeat.notify()

        output_list = [output_items[i] for i in sorted(output_items.keys())]

        incomplete = finish_reason in ("length", "max_tokens", "content_filter")
        status = "incomplete" if incomplete else "completed"

        final_response = dict(in_progress_shell)
        final_response["status"] = status
        final_response["model"] = model_name
        final_response["output"] = output_list
        if incomplete:
            if finish_reason in ("length", "max_tokens"):
                final_response["incomplete_details"] = {"reason": "max_output_tokens"}
            elif finish_reason == "content_filter":
                final_response["incomplete_details"] = {"reason": "content_filter"}

        event_name = "response.incomplete" if status == "incomplete" else "response.completed"
        await _responses_write_event(resp, event_name, {
            "type": event_name,
            "response": final_response,
            "sequence_number": next_seq(),
        })
        heartbeat.notify()

    finally:
        await state.tracker.unregister(req_id)


async def responses_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()

    if state.is_shutting_down:
        return _error_response(503, "Server is shutting down, please try again later.", "server_shutdown")

    if MAX_QUEUE_SIZE > 0 and state.scheduler.pending >= MAX_QUEUE_SIZE:
        logger.warning("Queue full, rejecting Responses request")
        return _error_response(503, "Server is busy, please try again later.", "server_busy")

    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON in request body")

    try:
        payload, context = _build_responses_payload(body)
    except ValueError as e:
        return _error_response(400, str(e), "invalid_request_error")
    except Exception as e:
        logger.error("Failed to convert Responses request: %s", e, exc_info=True)
        return _error_response(400, str(e), "invalid_request_error")

    stream = bool(body.get("stream", False))
    req_id = _gen_id("req")

    logger.debug("Responses request: model=%s, stream=%s, req_id=%s",
                 payload.get("model"), stream, req_id)

    if not stream:
        try:
            result = await state.scheduler.submit(
                lambda: _do_non_stream_request(state, req_id, payload)
            )
            if result is None:
                return _error_response(502, "No response from Zen API", "upstream_error")
            try:
                chat_response = json.loads(result.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                return _error_response(502, "Invalid JSON response from Zen API: {}".format(e), "upstream_error")
            if not isinstance(chat_response, dict):
                return _error_response(502, "Invalid response object from Zen API", "upstream_error")
            return _json_response(_convert_chat_to_response(chat_response, context))
        except QueueFullError as e:
            return _error_response(503, str(e), "server_busy")
        except asyncio.CancelledError:
            return _error_response(503, "Server is shutting down.", "server_shutdown")
        except ModelNotSupportedError as e:
            return _error_response(400, str(e), "model_not_supported")
        except ProviderValidationError as e:
            return _error_response(400, str(e), "invalid_request_error")
        except UpstreamError as e:
            return _error_response(502, str(e), "upstream_error")
        except Exception as e:
            logger.error("Responses non-stream error: %s", e, exc_info=True)
            return _error_response(500, str(e), "server_error")

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)

    async with StreamHeartbeat(resp) as heartbeat:
        try:
            await state.scheduler.submit(
                lambda: _do_responses_real_stream(state, req_id, payload, context, resp, heartbeat)
            )
        except QueueFullError as e:
            await _responses_write_event(resp, "error", {
                "type": "error", "error": {"message": str(e), "type": "server_busy", "code": 503},
            })
        except asyncio.CancelledError:
            await _responses_write_event(resp, "error", {
                "type": "error", "error": {"message": "Server is shutting down.", "type": "server_shutdown", "code": 503},
            })
        except ConnectionResetError:
            pass
        except ModelNotSupportedError as e:
            await _responses_write_event(resp, "error", {
                "type": "error", "error": {"message": str(e), "type": "model_not_supported", "code": 400},
            })
        except ProviderValidationError as e:
            await _responses_write_event(resp, "error", {
                "type": "error", "error": {"message": str(e), "type": "invalid_request_error", "code": 400},
            })
        except UpstreamError as e:
            await _responses_write_event(resp, "error", {
                "type": "error", "error": {"message": str(e), "type": "upstream_error", "code": 502},
            })
        except Exception as e:
            logger.error("Responses stream error: %s", e, exc_info=True)
            try:
                await _responses_write_event(resp, "error", {
                    "type": "error", "error": {"message": str(e), "type": "server_error", "code": 500},
                })
            except Exception:
                pass

    return resp


# ============================================================
# OpenAI 兼容端点
# ============================================================

async def health_handler(request: web.Request) -> web.Response:
    state = get_state()
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "ok",
        "platform": "zen",
        "timestamp": int(time.time()),
        "node": {
            "current": state.node_manager.current_description,
            "index": state.node_manager.current_index,
            "pool_size": state.node_manager.pool_size,
            "static_pool_size": state.static_pool_size,
            "dynamic_pool_size": state.dynamic_pool_size,
        },
        "scheduler": {
            "max_concurrent": MAX_CONCURRENT,
            "max_queue": MAX_QUEUE_SIZE,
            "pending": state.scheduler.pending,
        },
        "http_backend": "aiohttp",
        "user_agent": DEFAULT_USER_AGENT,
        "node_switch": "enabled (fail-fast, 429+proxy)",
    })

async def list_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    models = await state.zen_client.fetch_models()
    return _json_response({
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 1700000000, "owned_by": "zen"}
            for m in models
        ],
    })

async def get_model_handler(request: web.Request) -> web.Response:
    model_id = request.match_info.get("model_id", "")
    state = get_state()
    models = await state.zen_client.fetch_models()
    for m in models:
        if m == model_id:
            return _json_response({"id": m, "object": "model", "created": 1700000000, "owned_by": "zen"})
    return _error_response(404, "Model not found: {}".format(model_id), "model_not_found")

async def chat_completions_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()

    if state.is_shutting_down:
        return web.Response(
            status=503,
            text="Server is shutting down, please try again later.",
            content_type="text/plain",
        )

    if MAX_QUEUE_SIZE > 0 and state.scheduler.pending >= MAX_QUEUE_SIZE:
        logger.warning("Queue full, rejecting request")
        return web.Response(
            status=503,
            text="Server is busy, please try again later.",
            content_type="text/plain",
        )

    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON in request body")
    if not body.get("messages"):
        return _error_response(400, "messages is required")
    if not body.get("model"):
        return _error_response(400, "model is required")

    # 规范化模型名
    model = normalize_model_name(body["model"])
    if model != body["model"]:
        body["model"] = model

    # 提取可能需要的额外字段
    extra = body.get("extra_body") or body.get("extra") or {}
    # 归一化工具（如果存在）
    if body.get("tools"):
        normalized_tools = normalize_tools(body["tools"])
        body["tools"] = normalized_tools

    # 构建 payload（保留所有原始字段，并补充 extra）
    payload = build_payload(
        messages=body["messages"],
        model=body["model"],
        stream=bool(body.get("stream", False)),
        tools=body.get("tools"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens"),
        stop=body.get("stop"),
        tool_choice=body.get("tool_choice"),
        response_format=body.get("response_format"),
        reasoning_effort=body.get("reasoning_effort"),
        stream_options=body.get("stream_options"),
        thinking=bool(extra.get("thinking", False) or body.get("thinking", False)),
        search=bool(extra.get("search", False) or body.get("search", False)),
        extra=extra,
    )

    stream = payload["stream"]
    req_id = _gen_id("req")

    logger.debug("Chat completions request: model=%s, stream=%s, req_id=%s", model, stream, req_id)

    # 提取客户端传入的自定义头部（如 x-opencode-*）
    client_headers = {}
    for key, value in request.headers.items():
        if key.startswith("x-opencode-"):
            client_headers[key] = value
    # 保留原始 User-Agent
    if "User-Agent" in request.headers:
        client_headers["User-Agent"] = request.headers["User-Agent"]

    if not stream:
        try:
            result = await state.scheduler.submit(
                lambda: _do_non_stream_request(state, req_id, payload, client_headers)
            )
            if result is None:
                return _error_response(500, "No response from Zen API")
            try:
                data = json.loads(result.decode("utf-8"))
                return _json_response(data)
            except Exception:
                return web.Response(body=result, content_type="application/json")
        except QueueFullError as e:
            return web.Response(status=503, text=str(e), content_type="text/plain")
        except asyncio.CancelledError:
            return web.Response(status=503, text="Server is shutting down.", content_type="text/plain")
        except ModelNotSupportedError as e:
            return _error_response(400, str(e), "model_not_supported")
        except ProviderValidationError as e:
            return _error_response(400, str(e), "invalid_request_error")
        except UpstreamError as e:
            return _error_response(502, str(e), "upstream_error")
        except Exception as e:
            logger.error("chat_completions non-stream error: %s", e, exc_info=True)
            return _error_response(500, str(e), "server_error")

    # 流式响应
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    try:
        await state.scheduler.submit(
            lambda: _do_stream_request(state, req_id, payload, resp, client_headers)
        )
    except QueueFullError as e:
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": str(e), "type": "server_busy"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass
    except asyncio.CancelledError:
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": "Server is shutting down.", "type": "server_shutdown"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass
    except ConnectionResetError:
        pass
    except ModelNotSupportedError as e:
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": str(e), "type": "model_not_supported"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass
    except ProviderValidationError as e:
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass
    except UpstreamError as e:
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": str(e), "type": "upstream_error"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass
    except Exception as e:
        logger.error("chat_completions stream error: %s", e, exc_info=True)
        try:
            await resp.write(("data: " + json.dumps({"error": {"message": str(e), "type": "server_error"}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        except Exception:
            pass

    return resp

async def _do_non_stream_request(
    state: AppState,
    req_id: str,
    payload: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[bytes]:
    task = asyncio.current_task()
    await state.tracker.register(req_id, task)
    try:
        result = None
        async for chunk in state.zen_client.chat_completion(payload, extra_headers=extra_headers):
            result = chunk
            break
        return result
    finally:
        await state.tracker.unregister(req_id)

async def _do_stream_request(
    state: AppState,
    req_id: str,
    payload: Dict[str, Any],
    resp: web.StreamResponse,
    extra_headers: Optional[Dict[str, str]] = None,
) -> None:
    task = asyncio.current_task()
    await state.tracker.register(req_id, task)
    try:
        async for chunk in state.zen_client.chat_completion(payload, extra_headers=extra_headers):
            if chunk:
                try:
                    await resp.write(chunk)
                except ConnectionResetError:
                    raise
                except Exception:
                    return
    finally:
        await state.tracker.unregister(req_id)

# ============================================================
# Anthropic 兼容端点
# ============================================================

def _anthropic_convert_messages(
    body: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    system = body.get("system", "")
    if isinstance(system, list):
        system = "\n".join(
            p.get("text", "") for p in system
            if isinstance(p, dict) and p.get("type") == "text"
        )
    elif not isinstance(system, str):
        system = str(system)

    oai_messages: List[Dict[str, Any]] = []
    if system:
        oai_messages.append({"role": "system", "content": system})

    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            oai_messages.append({"role": role, "content": str(content)})
            continue

        converted: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            pt = part.get("type", "")

            if pt == "text":
                converted.append({"type": "text", "text": part.get("text", "")})
            elif pt == "image":
                source = part.get("source", {})
                st = source.get("type", "")
                if st == "url":
                    converted.append({
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    })
                elif st == "base64":
                    converted.append({
                        "type": "image_url",
                        "image_url": {
                            "url": "data:{};base64,{}".format(
                                source.get("media_type", "image/jpeg"),
                                source.get("data", ""),
                            )
                        },
                    })
            elif pt == "tool_use":
                pass
            elif pt == "tool_result":
                tool_id = part.get("tool_use_id", "")
                tc_content = part.get("content", "")
                if isinstance(tc_content, list):
                    tc_content = "\n".join(
                        p.get("text", "") for p in tc_content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": str(tc_content),
                })
                converted = []
                break

        if converted:
            oai_messages.append({"role": role, "content": converted})

    oai_tools = normalize_tools(body.get("tools"))
    return oai_messages, oai_tools

def _convert_to_anthropic(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        return {
            "id": _msg_id(), "type": "message", "role": "assistant",
            "content": [], "model": response.get("model", ""),
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    message = choices[0].get("message", {})
    content_text = message.get("content", "")
    reasoning = message.get("reasoning") or message.get("reasoning_content", "")
    tool_calls = message.get("tool_calls", [])

    anth_content: List[Dict[str, Any]] = []
    if reasoning:
        anth_content.append({"type": "thinking", "thinking": reasoning})
    if content_text:
        anth_content.append({"type": "text", "text": content_text})

    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, dict):
            args_json = args
        else:
            try:
                args_json = json.loads(args)
            except json.JSONDecodeError:
                args_json = {}

        tool_id = tc.get("id") or _tool_id()
        if not tool_id.startswith("toolu_"):
            tool_id = "toolu_" + tool_id

        anth_content.append({
            "type": "tool_use",
            "id": tool_id,
            "name": func.get("name", ""),
            "input": args_json,
        })

    if not anth_content:
        anth_content.append({"type": "text", "text": ""})

    usage = response.get("usage", {})
    return {
        "id": response.get("id", _msg_id()),
        "type": "message",
        "role": "assistant",
        "content": anth_content,
        "model": response.get("model", ""),
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

async def anthropic_messages_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()

    if state.is_shutting_down:
        return web.Response(status=503, text="Server is shutting down, please try again later.", content_type="text/plain")

    if MAX_QUEUE_SIZE > 0 and state.scheduler.pending >= MAX_QUEUE_SIZE:
        logger.warning("Queue full, rejecting Anthropic request")
        return web.Response(status=503, text="Server is busy, please try again later.", content_type="text/plain")

    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON")
    if not body.get("messages"):
        return _error_response(400, "messages is required")
    if not body.get("model"):
        return _error_response(400, "model is required")

    model = normalize_model_name(body["model"])
    if model != body["model"]:
        body["model"] = model

    stream = bool(body.get("stream", False))
    oai_messages, oai_tools = _anthropic_convert_messages(body)

    thinking = False
    t = body.get("thinking")
    if isinstance(t, bool):
        thinking = t
    elif isinstance(t, dict):
        thinking = t.get("type") == "enabled" or bool(t.get("enabled", False))

    payload = build_payload(
        messages=oai_messages,
        model=model,
        stream=stream,
        tools=oai_tools,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens", 4096),
        stop=body.get("stop_sequences"),
        tool_choice=body.get("tool_choice"),
        thinking=thinking,
        search=body.get("search", False),
    )

    req_id = _gen_id("req")
    logger.debug("Anthropic messages request: model=%s, stream=%s, req_id=%s", model, stream, req_id)

    if not stream:
        try:
            result = await state.scheduler.submit(lambda: _do_non_stream_request(state, req_id, payload))
            if result is None:
                return _error_response(500, "No response from Zen API")
            try:
                data = json.loads(result.decode("utf-8"))
                return _json_response(_convert_to_anthropic(data))
            except Exception:
                return _json_response(_convert_to_anthropic({"choices": []}))
        except QueueFullError as e:
            return web.Response(status=503, text=str(e), content_type="text/plain")
        except asyncio.CancelledError:
            return web.Response(status=503, text="Server is shutting down.", content_type="text/plain")
        except ModelNotSupportedError as e:
            return _error_response(400, str(e), "model_not_supported")
        except ProviderValidationError as e:
            return _error_response(400, str(e), "invalid_request_error")
        except UpstreamError as e:
            return _error_response(502, str(e), "upstream_error")
        except Exception as e:
            logger.error("anthropic_messages non-stream error: %s", e, exc_info=True)
            return _error_response(500, str(e), "server_error")

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)
    msg_id = _msg_id()

    try:
        await state.scheduler.submit(lambda: _do_anthropic_stream(state, req_id, payload, resp, model, msg_id))
    except QueueFullError as e:
        await _sse_write(resp, "error", {"error": {"message": str(e), "type": "server_busy"}})
    except asyncio.CancelledError:
        await _sse_write(resp, "error", {"error": {"message": "Server is shutting down.", "type": "server_shutdown"}})
    except ConnectionResetError:
        pass
    except ModelNotSupportedError as e:
        await _sse_write(resp, "error", {"error": {"message": str(e), "type": "model_not_supported"}})
    except ProviderValidationError as e:
        await _sse_write(resp, "error", {"error": {"message": str(e), "type": "invalid_request_error"}})
    except UpstreamError as e:
        await _sse_write(resp, "error", {"error": {"message": str(e), "type": "upstream_error"}})
    except Exception as e:
        logger.error("anthropic_messages stream error: %s", e, exc_info=True)
        await _sse_write(resp, "error", {"error": {"message": str(e), "type": "server_error"}})

    return resp

async def _do_anthropic_stream(
    state: AppState,
    req_id: str,
    payload: Dict[str, Any],
    resp: web.StreamResponse,
    model: str,
    msg_id: str
) -> None:
    task = asyncio.current_task()
    await state.tracker.register(req_id, task)
    try:
        tool_buf: Dict[int, Dict[str, str]] = {}
        text_started = False
        text_index = 0
        buffer = b""

        await _sse_write(resp, "message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        await _sse_write(resp, "ping", {"type": "ping"})

        chunk_count = 0
        async for chunk in state.zen_client.chat_completion(payload):
            chunk_count += 1
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                    logger.debug("Anthropic SSE data #%d: %s", chunk_count,
                                 json.dumps(obj, ensure_ascii=False)[:300])
                except Exception:
                    continue

                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta", {})

                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    if not text_started:
                        text_started = True
                        await _sse_write(resp, "content_block_start", {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    await _sse_write(resp, "content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    })
                    continue

                content = delta.get("content", "")
                if content:
                    if not text_started:
                        text_started = True
                        await _sse_write(resp, "content_block_start", {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    await _sse_write(resp, "content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": content},
                    })
                    continue

                tc = delta.get("tool_calls")
                if tc:
                    for t in tc:
                        idx = t.get("index", 0)
                        if idx not in tool_buf:
                            tool_buf[idx] = {"id": "", "name": "", "arguments": ""}
                        buf = tool_buf[idx]
                        if t.get("id"):
                            buf["id"] = t["id"]
                        func = t.get("function", {})
                        if func.get("name"):
                            buf["name"] = func["name"]
                        if func.get("arguments"):
                            buf["arguments"] += func["arguments"]

                finish_reason = choice.get("finish_reason")
                if finish_reason == "stop":
                    if text_started:
                        await _sse_write(resp, "content_block_stop", {
                            "type": "content_block_stop", "index": text_index,
                        })

                    has_tools = bool(tool_buf)
                    for tc_idx in sorted(tool_buf.keys()):
                        buf = tool_buf[tc_idx]
                        tool_id = buf["id"] or _tool_id()
                        if not tool_id.startswith("toolu_"):
                            tool_id = "toolu_" + tool_id
                        args_str = buf["arguments"] or "{}"
                        try:
                            json.loads(args_str)
                        except json.JSONDecodeError:
                            args_str = "{}"
                        block_index = text_index + 1 + tc_idx
                        await _sse_write(resp, "content_block_start", {
                            "type": "content_block_start", "index": block_index,
                            "content_block": {"type": "tool_use", "id": tool_id,
                                              "name": buf["name"], "input": {}},
                        })
                        await _sse_write(resp, "content_block_delta", {
                            "type": "content_block_delta", "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": args_str},
                        })
                        await _sse_write(resp, "content_block_stop", {
                            "type": "content_block_stop", "index": block_index,
                        })

                    await _sse_write(resp, "message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use" if has_tools else "end_turn",
                                  "stop_sequence": None},
                        "usage": {"output_tokens": 0},
                    })
                    await _sse_write(resp, "message_stop", {"type": "message_stop"})
                    return
    finally:
        await state.tracker.unregister(req_id)

# ============================================================
# 其他端点
# ============================================================

async def anthropic_root_handler(request: web.Request) -> web.Response:
    return web.Response(
        status=200,
        headers={"Content-Type": "application/json", "Anthropic-Version": "2023-06-01"},
        text=json.dumps({
            "type": "message", "version": "2023-06-01", "status": "ok",
            "endpoints": ["/v1/messages", "/anthropic/v1/messages"]
        })
    )

async def anthropic_list_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    models = await state.zen_client.fetch_models()
    now = int(time.time())
    data = [{"type": "model", "id": m, "display_name": m, "created_at": now} for m in models]
    return _json_response({
        "type": "list", "data": data, "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })

async def anthropic_retrieve_model_handler(request: web.Request) -> web.Response:
    model_id = request.match_info.get("model_id", "")
    return _json_response({
        "type": "model", "id": model_id,
        "display_name": model_id, "created_at": int(time.time()),
    })

async def anthropic_count_tokens_handler(request: web.Request) -> web.Response:
    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON")
    estimated = 0
    for m in body.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            estimated += len(content) // 3
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    estimated += len(part["text"]) // 3
    for t in body.get("tools", []):
        estimated += len(json.dumps(t, ensure_ascii=False)) // 3
    return _json_response({"input_tokens": estimated})

async def admin_refresh_models_handler(request: web.Request) -> web.Response:
    state = get_state()
    models = await state.zen_client.fetch_models(use_cache=False)
    logger.info("Admin: Refreshed models, got %d models", len(models))
    return _json_response({"status": "ok", "models": models, "count": len(models), "timestamp": int(time.time())})

async def admin_switch_node_handler(request: web.Request) -> web.Response:
    state = get_state()
    old = state.node_manager.current_description
    new = await state.node_manager.switch_next()
    cancelled = await state.tracker.cancel_all()
    logger.info("Admin: Switched node from %s to %s, cancelled %d requests", old, new, cancelled)
    return _json_response({
        "status": "ok", "previous_node": old, "current_node": new,
        "cancelled_requests": cancelled, "timestamp": int(time.time()),
    })

async def capabilities_handler(request: web.Request) -> web.Response:
    return _json_response({
        "platform": "zen", "capabilities": CAPABILITIES,
        "models": DEFAULT_MODELS, "timestamp": int(time.time()),
    })

async def status_handler(request: web.Request) -> web.Response:
    state = get_state()
    models = await state.zen_client.fetch_models()
    pool_snapshot = ["direct" if p is None else p for p in state.node_manager.pool_snapshot]
    return _json_response({
        "status": "shutting_down" if state.is_shutting_down else "running",
        "platform": "zen",
        "node": {
            "current": state.node_manager.current_description,
            "index": state.node_manager.current_index,
            "pool": pool_snapshot,
            "pool_size": state.node_manager.pool_size,
            "static_pool_size": state.static_pool_size,
            "dynamic_pool_size": state.dynamic_pool_size,
            "dynamic_pool_file": state.dynamic_pool_file,
        },
        "scheduler": {
            "max_concurrent": MAX_CONCURRENT,
            "max_queue": MAX_QUEUE_SIZE,
            "pending": state.scheduler.pending,
            "active_upstream": state.tracker.count,
        },
        "models": {"available": models, "count": len(models), "default": DEFAULT_MODELS},
        "fallback": {"enabled": FALLBACK_MODEL_ENABLED, "model": FALLBACK_MODEL},
        "capabilities": CAPABILITIES,
        "http_backend": "aiohttp",
        "user_agent": DEFAULT_USER_AGENT,
        "node_switch": "enabled (fail-fast, 429+proxy)",
        "timestamp": int(time.time()),
    })

async def count_tokens_handler(request: web.Request) -> web.Response:
    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON")
    estimated = 0
    for m in body.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            estimated += len(content) // 3
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    estimated += len(part["text"]) // 3
    for t in body.get("tools", []):
        estimated += len(json.dumps(t, ensure_ascii=False)) // 3
    return _json_response({"input_tokens": estimated})

_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}

async def function_call_handler(request: web.Request) -> web.Response:
    body = await _get_json(request)
    if body is None:
        return _error_response(400, "Invalid JSON")
    name = body.get("name", "")
    if name not in _FUNCTION_REGISTRY:
        return _error_response(404, "Function not found: {}".format(name))
    logger.debug("Function call: %s", name)
    return _json_response({"name": name, "arguments": body.get("arguments", {}), "output": "Executed {}".format(name)})

async def list_functions_handler(request: web.Request) -> web.Response:
    return _json_response({"functions": list(_FUNCTION_REGISTRY.values())})

# ============================================================
# 路由注册
# ============================================================

def setup_routes(app: web.Application) -> None:
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/v1/health", health_handler)

    app.router.add_get("/v1/models", list_models_handler)
    app.router.add_get("/v1/models/{model_id}", get_model_handler)
    app.router.add_post("/v1/chat/completions", chat_completions_handler)

    app.router.add_post("/v1/responses", responses_handler)

    app.router.add_post("/v1/messages/count_tokens", count_tokens_handler)

    app.router.add_get("/anthropic", anthropic_root_handler)
    app.router.add_post("/anthropic", anthropic_root_handler)

    app.router.add_post("/v1/messages", anthropic_messages_handler)
    app.router.add_post("/anthropic/v1/messages", anthropic_messages_handler)
    app.router.add_post("/anthropic/messages", anthropic_messages_handler)

    app.router.add_get("/anthropic/v1/models", anthropic_list_models_handler)
    app.router.add_get("/anthropic/v1/models/{model_id}", anthropic_retrieve_model_handler)

    app.router.add_post("/anthropic/v1/messages/count_tokens", anthropic_count_tokens_handler)

    app.router.add_post("/v1/function/call", function_call_handler)
    app.router.add_get("/v1/functions", list_functions_handler)

    app.router.add_post("/v1/admin/refresh_models", admin_refresh_models_handler)
    app.router.add_post("/v1/admin/switch_node", admin_switch_node_handler)
    app.router.add_get("/v1/capabilities", capabilities_handler)
    app.router.add_get("/v1/status", status_handler)

# ============================================================
# 启动入口
# ============================================================

def _check_port_in_use(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return False
    except OSError:
        return True

def _install_signal_handlers(state: AppState) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _handle(sig_name: str) -> None:
        logger.info("Received signal %s, initiating graceful shutdown...", sig_name)
        state.shutdown_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _handle, sig_name)
            logger.debug("Installed signal handler for %s", sig_name)
        except (NotImplementedError, RuntimeError, AttributeError, ValueError):
            logger.debug("Signal handler for %s not available on this platform", sig_name)

async def _wait_for_shutdown(state: AppState) -> None:
    while not state.shutdown_event.is_set():
        try:
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

async def _cancel_leftover_tasks() -> None:
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if not tasks:
        return
    logger.debug("Cancelling %d leftover task(s) before exit", len(tasks))
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

async def main_async() -> None:
    if _check_port_in_use(PORT):
        logger.error("=" * 70)
        logger.error("ERROR: Port %d is already in use!", PORT)
        logger.error("  Windows: netstat -ano | findstr %d -> taskkill /PID <PID> /F", PORT)
        logger.error("  Linux:   lsof -ti:%d | xargs kill -9", PORT)
        logger.error("=" * 70)
        sys.exit(1)

    app = web.Application()
    setup_routes(app)
    state = get_state()

    log_level = "DEBUG" if logger.level <= 10 else "INFO"

    logger.info("Zen Platform Server starting...")
    logger.info("Config:")
    logger.info("  Port                : %d", PORT)
    logger.info("  Max Concurrent      : %s", "unlimited" if MAX_CONCURRENT == -1 else MAX_CONCURRENT)
    logger.info("  Max Queue Size      : %d", MAX_QUEUE_SIZE)
    logger.info("  Node Switch         : ENABLED (fail-fast, max %d attempts)", MAX_NODE_SWITCH_ATTEMPTS)
    logger.info("  Switch Triggers     : ProxyError, RateLimitedError (429)")
    logger.info("  Fallback Model      : %s", FALLBACK_MODEL if FALLBACK_MODEL_ENABLED else "disabled")
    logger.info("  Heartbeat Interval  : %.0fs", HEARTBEAT_INTERVAL)
    logger.info("Node Pool:")
    logger.info("  Static Pool Size    : %d", state.static_pool_size)
    logger.info("  Dynamic Pool Size   : %d (from %s)", state.dynamic_pool_size, PROXY_POOL_FILE)
    logger.info("  Merged Pool Size    : %d", state.node_manager.pool_size)
    logger.info("  Current Node        : %s (index=%d)", state.node_manager.current_description, state.node_manager.current_index)
    logger.info("Endpoints:")
    logger.info("  POST /v1/chat/completions  (OpenAI format, with reasoning_effort/stream_options)")
    logger.info("  POST /v1/responses")
    logger.info("  POST /v1/messages          (Anthropic)")
    logger.info("  POST /anthropic/v1/messages (Anthropic)")
    logger.info("  GET  /v1/status")
    logger.info("=" * 70)
    logger.info("Log level: %s, DEBUG_LOG_BODY: %s", log_level, DEBUG_LOG_BODY)

    runner = web.AppRunner(app, shutdown_timeout=RUNNER_SHUTDOWN_TIMEOUT)
    site: Optional[web.TCPSite] = None

    try:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()
        logger.info("Server started on http://0.0.0.0:%d", PORT)

        _install_signal_handlers(state)

        try:
            await _wait_for_shutdown(state)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, initiating graceful shutdown...")
            state.shutdown_event.set()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during startup, exiting...")
        state.shutdown_event.set()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        state.shutdown_event.set()
        raise
    finally:
        logger.info("Cleaning up...")
        if site is not None:
            try:
                await site.stop()
            except Exception as e:
                logger.debug("site.stop() error: %s", e)
        try:
            await asyncio.wait_for(state.shutdown(), timeout=SHUTDOWN_TOTAL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("AppState.shutdown() timed out after %.1fs, forcing exit", SHUTDOWN_TOTAL_TIMEOUT)
        except KeyboardInterrupt:
            logger.warning("Second KeyboardInterrupt received, forcing immediate exit")
        except Exception as e:
            logger.error("Error during AppState.shutdown(): %s", e, exc_info=True)
        try:
            await runner.cleanup()
        except Exception as e:
            logger.debug("runner.cleanup() error: %s", e)
        try:
            await _cancel_leftover_tasks()
        except Exception as e:
            logger.debug("_cancel_leftover_tasks() error: %s", e)
        logger.info("Server stopped")

def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting...")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()