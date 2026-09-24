"""Streamed tool-call arguments were re-parsed in full on every chunk, open-webui v0.11.2.

Fix commit `061f5e3a6` (#28858) in `backend/open_webui/utils/anthropic.py`. The
OpenAI-to-Anthropic stream converter buffers a tool call's `arguments` string and closes the
content block once the buffer forms complete JSON. It ran `JSONCodec.loads` over the whole
buffer after every delta, so a large tool call cost a full re-parse per chunk. The fix only
attempts the parse when the buffer can plausibly be complete: it ends in `}`, or it never
started with `{`. Only the parse count shows the bug, so it stays here; how the arguments
assemble through /api/v1/messages lives in
integration/chat/test_anthropic_tool_call_streaming.py.

Discriminates: passes on v0.11.3 and on dev bbfa876af, fails with the guard removed (the
buffer is parsed once per argument chunk rather than once at the end).
"""

from __future__ import annotations

import asyncio
import json
from types import ModuleType
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.regression

FLAT_ARGUMENTS = '{"query": "quarterly revenue", "limit": 25, "verbose": true}'
NESTED_ARGUMENTS = '{"filter": {"year": 2024, "region": "emea"}, "limit": 3}'


@pytest.fixture(scope="session")
def anthropic_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.anthropic")


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _tool_call_stream(argument_chunks: list[str]):
    """An OpenAI stream: a tool call opened by id, then argument-only deltas."""
    opened = {"index": 0, "id": "call_1", "type": "function"}
    opened["function"] = {"name": "search", "arguments": ""}
    yield _sse({"choices": [{"delta": {"tool_calls": [opened]}}]})
    for piece in argument_chunks:
        delta = {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}
        yield _sse({"choices": [{"delta": delta}]})
    yield _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    yield b"data: [DONE]\n\n"


async def _buffer_parses(module: ModuleType, argument_chunks: list[str]) -> list[str]:
    """Stream the chunks through the real converter; return its parses of the argument buffer."""
    arguments = "".join(argument_chunks)
    with patch.object(module.JSONCodec, "loads", wraps=module.JSONCodec.loads) as loads:
        stream = module.openai_stream_to_anthropic_stream(
            openai_stream_generator=_tool_call_stream(argument_chunks), model="claude-sonnet-4-5"
        )
        events = [event async for event in stream]
    assert any(b"content_block_stop" in event for event in events), "the tool block never closed"
    # The codec also decodes each SSE envelope; only prefixes of the arguments are buffer parses.
    parsed = [call.args[0] for call in loads.call_args_list]
    return [value for value in parsed if isinstance(value, str) and arguments.startswith(value)]


@pytest.mark.asyncio
async def test_argument_buffer_is_not_reparsed_on_every_chunk(anthropic_module):
    """Twelve small deltas used to cost twelve full re-parses of the growing buffer."""
    chunks = [FLAT_ARGUMENTS[start : start + 5] for start in range(0, len(FLAT_ARGUMENTS), 5)]

    parses = await asyncio.wait_for(_buffer_parses(anthropic_module, chunks), timeout=10)

    assert len(chunks) == 12
    assert parses == [FLAT_ARGUMENTS]


@pytest.mark.asyncio
async def test_nested_arguments_parse_only_where_the_buffer_could_be_complete(anthropic_module):
    """Only the two boundaries that leave the buffer ending in `}` are tried."""
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

    parses = await asyncio.wait_for(_buffer_parses(anthropic_module, chunks), timeout=10)

    assert len(parses) == 2
    assert parses[-1] == NESTED_ARGUMENTS
