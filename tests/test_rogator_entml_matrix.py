from __future__ import annotations

"""Rogator：非流式 / 流式 × OpenAI / Anthropic × type_hint 矩阵。"""

import json
import sys
from pathlib import Path

import pytest

_QWEN_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent
for p in (_QWEN_ROOT, _TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rogator_entml_harness import (  # noqa: E402
    CHUNK_SIZES,
    assert_args_equal,
    batch_parse,
    merged_stream_json_per_invoke,
    simulate_anthropic_wire_json,
    simulate_openai_wire_json,
)
from fixtures.simulated_llm_tool_responses import (  # noqa: E402
    SimulatedCase,
    iter_cases_by_branch,
    iter_cases_with_tools,
    tools_for_case,
)
from handlers.openai import _parse_tool_calls  # noqa: E402
from state import AppState  # noqa: E402


@pytest.fixture(scope="module")
def app_state() -> AppState:
    from echotools.fncall import get_protocol

    state = AppState.__new__(AppState)
    state.protocol = get_protocol("entml")
    return state


@pytest.mark.parametrize("case", iter_cases_with_tools(), ids=lambda c: c.id)
def test_non_stream_openai_parse_matches_expect(app_state: AppState, case: SimulatedCase) -> None:
    tools = tools_for_case(case)
    clean, calls = _parse_tool_calls(app_state, case.response, tools)
    names = [c["function"]["name"] for c in calls]
    assert names == case.expect_names
    args = [json.loads(c["function"]["arguments"]) for c in calls]
    assert args == case.expect_args
    for parsed, expected in zip(args, case.expect_args):
        assert_args_equal(parsed, expected, case_id=case.id)
    for sub in case.expect_clean_substrings:
        assert sub in clean
    for bad in case.expect_clean_absent:
        assert bad not in clean


@pytest.mark.parametrize("case", iter_cases_with_tools(), ids=lambda c: c.id)
def test_non_stream_batch_protocol_matches_expect(case: SimulatedCase) -> None:
    tools = tools_for_case(case)
    calls = batch_parse(case.response, tools)
    names = [c["function"]["name"] for c in calls]
    assert names == case.expect_names
    args = [json.loads(c["function"]["arguments"]) for c in calls]
    assert args == case.expect_args


@pytest.mark.parametrize("case", iter_cases_with_tools(), ids=lambda c: c.id)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda n: f"c{n}")
def test_stream_merged_json_matches_batch(case: SimulatedCase, chunk_size: int) -> None:
    tools = tools_for_case(case)
    batch = batch_parse(case.response, tools)
    merged = merged_stream_json_per_invoke(case.response, tools, chunk_size)
    assert len(merged) == len(batch), (
        f"{case.id} chunk={chunk_size}: stream blocks {len(merged)} != batch {len(batch)}"
    )
    for m, call in zip(merged, batch):
        batch_args = json.loads(call["function"]["arguments"])
        stream_args = json.loads(m)
        assert stream_args == batch_args
        assert_args_equal(stream_args, batch_args, case_id=case.id)


@pytest.mark.parametrize("case", iter_cases_with_tools(), ids=lambda c: c.id)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda n: f"c{n}")
def test_anthropic_wire_stream_matches_batch(case: SimulatedCase, chunk_size: int) -> None:
    tools = tools_for_case(case)
    wire, batch = simulate_anthropic_wire_json(case.response, tools, chunk_size)
    assert len(wire) == len(batch), f"{case.id} ant wire={len(wire)} batch={len(batch)}"
    for w, call in zip(wire, batch):
        assert json.loads(w) == json.loads(call["function"]["arguments"])


@pytest.mark.parametrize("case", iter_cases_with_tools(), ids=lambda c: c.id)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda n: f"c{n}")
def test_openai_wire_stream_matches_batch(case: SimulatedCase, chunk_size: int) -> None:
    tools = tools_for_case(case)
    by_index, batch = simulate_openai_wire_json(case.response, tools, chunk_size)
    assert len(by_index) == len(batch), f"{case.id} oai idx={len(by_index)} batch={len(batch)}"
    for idx, call in enumerate(batch):
        assert json.loads(by_index[idx]) == json.loads(call["function"]["arguments"])


@pytest.mark.parametrize(
    "case",
    iter_cases_by_branch("type_hint_priority"),
    ids=lambda c: c.id,
)
@pytest.mark.parametrize("chunk_size", [1, 5, 64, 9999], ids=lambda n: f"c{n}")
def test_type_hint_non_stream_and_stream(case: SimulatedCase, chunk_size: int) -> None:
    tools = tools_for_case(case)
    batch = batch_parse(case.response, tools)
    assert len(batch) == len(case.expect_args)
    merged = merged_stream_json_per_invoke(case.response, tools, chunk_size)
    assert len(merged) == len(batch)
    for call, expect, m in zip(batch, case.expect_args, merged):
        batch_args = json.loads(call["function"]["arguments"])
        stream_args = json.loads(m)
        assert_args_equal(batch_args, expect, case_id=case.id)
        assert_args_equal(stream_args, expect, case_id=case.id)


@pytest.mark.parametrize("case", iter_cases_by_branch("parallel_multi_invoke"), ids=lambda c: c.id)
def test_parallel_invoke_whole_chunk(case: SimulatedCase) -> None:
    """双 invoke 一次 feed 时必须产出两段 JSON（曾缺第二段）。"""
    tools = tools_for_case(case)
    chunk = max(len(case.response), 1)
    merged = merged_stream_json_per_invoke(case.response, tools, chunk)
    assert len(merged) == len(case.expect_names)
    wire, _ = simulate_anthropic_wire_json(case.response, tools, chunk)
    assert len(wire) == len(case.expect_names)
    oai, _ = simulate_openai_wire_json(case.response, tools, chunk)
    assert len(oai) == len(case.expect_names)
