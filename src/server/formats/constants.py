from __future__ import annotations

PORT: int = 8932
MAX_CONCURRENT: int = 8
MAX_QUEUE_SIZE: int = 1000
PRELOGIN_ACCOUNT_COUNT: int = 3
REQUEST_TOTAL_TIMEOUT: float = 600.0
MODELS_FETCH_TIMEOUT: float = 60.0
LOGIN_TIMEOUT: float = 30.0
MAX_REQUEST_RESTARTS: int = 3
RESTART_DELAY: float = 1.0
SHUTDOWN_CANCEL_GRACE: float = 0.3
SHUTDOWN_WAIT_IDLE_TIMEOUT: float = 3.0
SHUTDOWN_TOTAL_TIMEOUT: float = 8.0
RUNNER_SHUTDOWN_TIMEOUT: float = 10.0
KEEPALIVE_INTERVAL: float = 5.0


def gen_id(prefix: str) -> str:
    import time
    import uuid

    return f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:12]}"


def gen_chatcmpl_id() -> str:
    return gen_id("gen")


def gen_request_id() -> str:
    return gen_id("req")


def gen_msg_id() -> str:
    return gen_id("msg")
