from __future__ import annotations

from echotools import get_protocol, inject_fncall

from handlers import extract_system_for_inject, normalize_message_content


def test_normalize_message_content_blocks() -> None:
    content = [
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]
    assert normalize_message_content(content) == "line one\nline two"


def test_extract_system_multiple_and_strip() -> None:
    messages = [
        {"role": "system", "content": "  sys-a  "},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "sys-b"},
    ]
    prompt, rest = extract_system_for_inject(messages)
    assert prompt == "sys-a\n\nsys-b"
    assert len(rest) == 1
    assert rest[0]["content"] == "hello"


def test_extract_system_only() -> None:
    prompt, rest = extract_system_for_inject([{"role": "system", "content": "solo"}])
    assert prompt == "solo"
    assert rest == []


def test_prepend_anthropic_system_blocks() -> None:
    from handlers import prepend_anthropic_system

    messages = prepend_anthropic_system([], [{"type": "text", "text": "block sys"}])
    assert messages == [{"role": "system", "content": "block sys"}]


def test_inject_renders_user_system_prompt_block() -> None:
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ]
    user_system_prompt, prepared = extract_system_for_inject(messages)
    prompt = inject_fncall(
        prepared,
        [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}],
        get_protocol("entml"),
        lang="zh",
        user_system_prompt=user_system_prompt,
    )[0]["content"]
    assert "<user_system_prompt>\n你是助手\n</user_system_prompt>" in prompt
    assert "你是助手" not in prompt.split("<current_user_message>")[1].split("</current_user_message>")[0]


def test_inject_no_tools_includes_user_system_prompt() -> None:
    user_system_prompt, prepared = extract_system_for_inject(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
    )
    prompt = inject_fncall(
        prepared, [], get_protocol("entml"), user_system_prompt=user_system_prompt,
    )[0]["content"]
    assert "<user_system_prompt>\nrules\n</user_system_prompt>" in prompt
