"""Streamed tool-call arguments were re-parsed in full on every chunk, open-webui v0.11.2.

Fix commit `061f5e3a6` (#28858) in `backend/open_webui/utils/anthropic.py`. The
OpenAI-to-Anthropic stream converter buffers a tool call's `arguments` string and
closes the content block once the buffer forms complete JSON. It ran
`JSONCodec.loads` over the whole buffer after every single delta, so a large tool
call cost a full re-parse per chunk. The fix only attempts the parse when the
buffer can plausibly be complete: it ends in `}`, or it never started with `{`.

Discriminates: passes on v0.11.3, fails on v0.11.1, which parses the buffer once
per argument chunk rather than once at the end.
"""

from __future__ import annotations

import asyncio
import json
from types import ModuleType

import pytest

pytestmark = pytest.mark.regression

STREAM_TIMEOUT_SECONDS = 10
MODEL = "claude-sonnet-4-5"

FLAT_ARGUMENTS = '{"query": "quarterly revenue", "limit": 25, "verbose": true}'
NESTED_ARGUMENTS = '{"filter": {"year": 2024, "region": "emea"}, "limit": 3}'


@pytest.fixture(scope="session")
def anthropic_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.anthropic")


class _CountingCodec:
    """Delegating JSON codec that records every string handed to `loads`."""

    def __init__(self, real):
        self._real = real
        self.JSONDecodeError = real.JSONDecodeError
        self.loaded = []

    def dumps(self, *args, **kwargs):
        return self._real.dumps(*args, **kwargs)

    def loads(self, value, *args, **kwargs):
        self.loaded.append(value)
        return self._real.loads(value, *args, **kwargs)


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _split(value: str, size: int) -> list[str]:
    return [value[start : start + size] for start in range(0, len(value), size)]


async def _tool_call_stream(name: str, argument_chunks: list[str], tool_id: str = "call_1"):
    """OpenAI SSE stream: a tool call opened by id, then argument-only deltas."""
    yield _sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        }
    )
    for chunk in argument_chunks:
        yield _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": chunk}}]
                        }
                    }
                ]
            }
        )
    yield _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    yield b"data: [DONE]\n\n"


async def _collect_events(module: ModuleType, source) -> list[dict]:
    """Drive the real converter to exhaustion and return the decoded SSE payloads."""

    async def _run() -> list[dict]:
        payloads = []
        async for raw in module.openai_stream_to_anthropic_stream(source, model=MODEL):
            event = raw.decode() if isinstance(raw, bytes) else raw
            for line in event.splitlines():
                if line.startswith("data:"):
                    payloads.append(json.loads(line[5:].strip()))
        return payloads

    # The source generator is finite; wait_for is only a backstop.
    return await asyncio.wait_for(_run(), timeout=STREAM_TIMEOUT_SECONDS)


def _assembled_arguments(events: list[dict]) -> str:
    return "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event["delta"].get("type") == "input_json_delta"
    )


def _block_stop_count(events: list[dict]) -> int:
    return sum(1 for event in events if event.get("type") == "content_block_stop")


def _buffer_parses(codec: _CountingCodec, arguments: str) -> list[str]:
    """The `loads` calls made against the argument buffer, not the SSE envelopes."""
    return [value for value in codec.loaded if arguments.startswith(value)]


async def _run_tool_call(module, monkeypatch, arguments: str, chunk_size: int):
    codec = _CountingCodec(module.JSONCodec)
    monkeypatch.setattr(module, "JSONCodec", codec)
    chunks = _split(arguments, chunk_size)
    events = await _collect_events(module, _tool_call_stream("search", chunks))
    return events, codec, chunks


# ═════════════════════════════════════════════════════════════════════════════
# Narrow
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_argument_buffer_is_not_reparsed_on_every_chunk(anthropic_module, monkeypatch):
    """Narrow. Twelve small deltas used to cost twelve full re-parses of the
    growing buffer. Only the last one can possibly succeed, so only it is tried."""
    events, codec, chunks = await _run_tool_call(anthropic_module, monkeypatch, FLAT_ARGUMENTS, 5)

    assert len(chunks) == 12
    assert _buffer_parses(codec, FLAT_ARGUMENTS) == [FLAT_ARGUMENTS]
    assert _assembled_arguments(events) == FLAT_ARGUMENTS
    assert json.loads(_assembled_arguments(events)) == {
        "query": "quarterly revenue",
        "limit": 25,
        "verbose": True,
    }
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_nested_arguments_parse_only_where_the_buffer_could_be_complete(
    anthropic_module, monkeypatch
):
    """Narrow. Only the two boundaries that leave the buffer ending in `}` are
    tried, and the six that cannot possibly hold complete JSON are skipped."""
    codec = _CountingCodec(anthropic_module.JSONCodec)
    chunks = [
        '{"filt',
        'er": ',
        '{"year"',
        ": 2024, ",
        '"region": "emea"}',
        ', "lim',
        'it": ',
        "3}",
    ]
    monkeypatch.setattr(anthropic_module, "JSONCodec", codec)
    events = await _collect_events(anthropic_module, _tool_call_stream("search", chunks))

    parses = _buffer_parses(codec, NESTED_ARGUMENTS)
    assert len(parses) == 2
    assert parses[-1] == NESTED_ARGUMENTS
    assert json.loads(_assembled_arguments(events)) == {
        "filter": {"year": 2024, "region": "emea"},
        "limit": 3,
    }
    assert _block_stop_count(events) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Broad: assembly stays correct however the provider splits the JSON
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_character_chunks_assemble_exactly(anthropic_module, monkeypatch):
    """Broad. Every boundary splits a JSON token, including string bodies and the
    digits of a number."""
    events, _, _ = await _run_tool_call(anthropic_module, monkeypatch, FLAT_ARGUMENTS, 1)

    assert _assembled_arguments(events) == FLAT_ARGUMENTS
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_chunk_boundary_inside_a_string_value_assembles_exactly(anthropic_module):
    """Broad. A boundary inside a quoted value must not close the block early."""
    chunks = ['{"query": "quar', 'terly rev', 'enue", "limit"', ": 25, ", '"verbose": true}']
    events = await _collect_events(
        anthropic_module, _tool_call_stream("search", chunks)
    )

    assert _assembled_arguments(events) == FLAT_ARGUMENTS
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_two_parallel_tool_calls_keep_their_own_buffers(anthropic_module):
    """Broad. Each tracked tool call assembles independently."""

    async def source():
        for index, (tool_id, name) in enumerate([("call_a", "alpha"), ("call_b", "beta")]):
            yield _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": tool_id,
                                        "function": {"name": name, "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        for index, argument in enumerate(['{"a": 1}', '{"b": 2}']):
            for piece in _split(argument, 3):
                yield _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": index, "function": {"arguments": piece}}
                                    ]
                                }
                            }
                        ]
                    }
                )
        yield b"data: [DONE]\n\n"

    events = await _collect_events(anthropic_module, source())

    by_block = {}
    for event in events:
        if event.get("type") == "content_block_delta" and event["delta"].get("type") == (
            "input_json_delta"
        ):
            by_block.setdefault(event["index"], []).append(event["delta"]["partial_json"])

    assert sorted("".join(parts) for parts in by_block.values()) == ['{"a": 1}', '{"b": 2}']
    assert _block_stop_count(events) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Nearby: behaviour that is unchanged by the fix
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_whole_json_in_one_chunk(anthropic_module, monkeypatch):
    """Nearby. One delta carrying the complete arguments parses exactly once."""
    events, codec, chunks = await _run_tool_call(
        anthropic_module, monkeypatch, FLAT_ARGUMENTS, len(FLAT_ARGUMENTS)
    )

    assert chunks == [FLAT_ARGUMENTS]
    assert _buffer_parses(codec, FLAT_ARGUMENTS) == [FLAT_ARGUMENTS]
    assert _assembled_arguments(events) == FLAT_ARGUMENTS
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_tool_call_with_no_arguments_still_opens_a_block(anthropic_module, monkeypatch):
    """Nearby. A tool call that never carries arguments emits a start and no
    argument deltas, and never attempts a parse of an empty buffer."""
    codec = _CountingCodec(anthropic_module.JSONCodec)
    monkeypatch.setattr(anthropic_module, "JSONCodec", codec)

    events = await _collect_events(anthropic_module, _tool_call_stream("ping", []))

    starts = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    ]
    assert [start["content_block"]["name"] for start in starts] == ["ping"]
    assert _assembled_arguments(events) == ""
    assert "" not in codec.loaded


@pytest.mark.asyncio
async def test_empty_json_object_arguments_close_the_block(anthropic_module):
    """Nearby. `{}` is complete JSON and closes the block."""
    events = await _collect_events(anthropic_module, _tool_call_stream("ping", ["{", "}"]))

    assert _assembled_arguments(events) == "{}"
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_closing_brace_with_trailing_whitespace_closes_the_block(anthropic_module):
    """Nearby. The guard strips trailing whitespace before looking for the closing
    brace. A trailing chunk arriving after the close is dropped, which is what
    proves the block was closed inline and not by the end-of-stream flush."""
    events = await _collect_events(
        anthropic_module, _tool_call_stream("ping", ['{"a": 1', "}\n", "late"])
    )

    assert _assembled_arguments(events) == '{"a": 1}\n'
    assert _block_stop_count(events) == 1


@pytest.mark.asyncio
async def test_arguments_that_are_not_an_object_still_close_the_block(anthropic_module):
    """Nearby. Arguments that never open with a brace are exempt from the guard, so
    a JSON array closes as soon as it parses and drops what follows."""
    events = await _collect_events(
        anthropic_module, _tool_call_stream("ping", ["[1,", " 2]", "late"])
    )

    assert _assembled_arguments(events) == "[1, 2]"
    assert _block_stop_count(events) == 1
