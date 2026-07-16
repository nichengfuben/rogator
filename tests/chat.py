from __future__ import annotations

"""Standalone async chat smoke-test for the current Qwen web flow."""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Tuple

import aiohttp

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.transport.endpoints import AUTH_BASE_URL, BASE_URL, CHAT_PATH, NEW_CHAT_PATH
    from core.crypto.crypto import build_headers, build_login_headers, hash_password
    from core.transport.sse import parse_sse_event
else:
    from ..core.transport.endpoints import AUTH_BASE_URL, BASE_URL, CHAT_PATH, NEW_CHAT_PATH
    from ..core.crypto.crypto import build_headers, build_login_headers, hash_password
    from ..core.transport.sse import parse_sse_event


def get_credentials() -> Tuple[str, str]:
    """Return credentials from environment variables GENERALUSR/GENERALPWD."""
    email = os.environ.get("GENERALUSR", "").strip()
    password = os.environ.get("GENERALPWD", "").strip()
    if not email or not password:
        raise SystemExit("GENERALUSR/GENERALPWD environment variables not set")
    return email, password


async def _login(session: aiohttp.ClientSession, email: str, password: str) -> str:
    """Log in with the current v2 endpoint and return the access token."""
    async with session.post(
        f"{AUTH_BASE_URL}/api/v2/auths/signin",
        json={"email": email, "password": hash_password(password), "remember_me": True},
        headers=build_login_headers(),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"login HTTP {response.status}: {(await response.text())[:300]}")
        data = await response.json()
        if not data.get("success"):
            raise RuntimeError(f"login failed: {data}")
        token = str((data.get("data") or {}).get("access_token", ""))
        if not token:
            raise RuntimeError(f"missing access token: {data}")
        return token


async def _create_chat(session: aiohttp.ClientSession, token: str, model: str) -> str:
    """Create a chat and return the chat identifier."""
    async with session.post(
        f"{BASE_URL}{NEW_CHAT_PATH}",
        json={
            "title": "新建对话",
            "models": [model],
            "chat_mode": "local",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000),
            "project_id": "",
        },
        headers=build_headers(token, include_version=False),
        ssl=False,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"create chat HTTP {response.status}: {(await response.text())[:300]}")
        data = await response.json()
        chat_id = str((data.get("data") or {}).get("id", ""))
        if not data.get("success") or not chat_id:
            raise RuntimeError(f"invalid create chat payload: {data}")
        return chat_id


def _build_chat_payload(model: str, chat_id: str, message: str) -> Dict[str, Any]:
    """Build the chat completion payload."""
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "local",
        "model": model,
        "parent_id": None,
        "messages": [{
            "fid": str(uuid.uuid4()),
            "parentId": None,
            "childrenIds": [str(uuid.uuid4())],
            "role": "user",
            "content": message,
            "user_action": "chat",
            "files": [],
            "timestamp": int(time.time() * 1000),
            "models": [model],
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": True,
                "output_schema": "phase",
                "research_mode": "normal",
                "auto_thinking": False,
                "thinking_mode": "Thinking",
                "thinking_format": "raw",
                "auto_search": False,
            },
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
        }],
        "timestamp": int(time.time() * 1000),
    }


def _parse_sse_line(data_str: str) -> Dict[str, Any]:
    """Parse a single SSE data line."""
    try:
        payload = json.loads(data_str)
    except Exception:
        return {}
    choices = payload.get("choices") or []
    if not choices:
        return {}
    delta = choices[0].get("delta", {})
    phase = delta.get("phase", "")
    content = delta.get("content", "")
    if phase == "think" and content:
        return {"type": "thinking", "content": content}
    elif phase == "answer" and content:
        return {"type": "answer", "content": content}
    return {}


async def get_qwen_stream(message: str, model: str = "qwen3.7-max") -> AsyncGenerator[Dict[str, Any], None]:
    """Log in directly and stream one response without host-project dependencies."""
    email, password = get_credentials()
    async with aiohttp.ClientSession() as session:
        try:
            token = await _login(session, email, password)
            chat_id = await _create_chat(session, token, model)
            yield {"type": "chat_id", "content": chat_id}
            payload = _build_chat_payload(model, chat_id, message)
            async with session.post(
                f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}",
                json=payload,
                headers=build_headers(token, chat_id=chat_id, include_sse=True),
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"chat HTTP {response.status}: {(await response.text())[:300]}")
                async for raw in response.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    event = _parse_sse_line(data_str)
                    if event:
                        yield event
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "content": str(exc)}


async def get_qwen_response(message: str, model: str = "qwen3.7-max") -> str:
    """Non-streaming convenience wrapper that returns the full answer text."""
    parts: list[str] = []
    async for event in get_qwen_stream(message, model):
        etype = event.get("type")
        if etype == "answer":
            parts.append(event.get("content", ""))
        elif etype == "error":
            raise RuntimeError(event.get("content", "unknown error"))
    return "".join(parts)


async def get_qwen_with_thinking(message: str, model: str = "qwen3.7-max") -> Dict[str, str]:
    """Return both the thinking trace and the final answer."""
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    async for event in get_qwen_stream(message, model):
        etype = event.get("type")
        if etype == "thinking":
            thinking_parts.append(event.get("content", ""))
        elif etype == "answer":
            answer_parts.append(event.get("content", ""))
        elif etype == "error":
            raise RuntimeError(event.get("content", "unknown error"))
    return {"thinking": "".join(thinking_parts), "answer": "".join(answer_parts)}


async def main() -> None:
    """Run a single smoke-test chat request."""
    async for event in get_qwen_stream("你好"):
        print(event)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
