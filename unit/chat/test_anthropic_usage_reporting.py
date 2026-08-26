"""Regression tests for Anthropic usage reporting (#26790, #27293, docs#1328).

Fix commits: e8b59b2, 51ff386, 0576e8e, 8e74cac, 93a34bb, 4c2d864, all in
`backend/open_webui/utils/anthropic.py`.

The Anthropic-compatible `/messages` endpoint translates an OpenAI-shaped
upstream response back into Anthropic's `usage` block. The old translation was
a literal two-key copy: `input_tokens = prompt_tokens`, `output_tokens =
completion_tokens`. Three things went wrong with that. OpenAI's `prompt_tokens`
INCLUDES cached tokens while Anthropic's `input_tokens` EXCLUDES them, so any
request using prompt caching reported an inflated input count and clients
double-counted the cached part. An upstream that already speaks Anthropic
numbers (`input_tokens` / `output_tokens`) reported 0/0. And cached-token counts
carried under `prompt_tokens_details.cached_tokens` were dropped entirely.

The fix derives the block instead: cache-read falls back to
`prompt_tokens_details.cached_tokens`, `input_tokens` is used verbatim when the
upstream supplies it and otherwise computed as
`max(prompt_tokens - cache_creation - cache_read, 0)`, and `output_tokens_details`,
`server_tool_use` and `service_tier` pass through when well-formed. Both
converters also accept a caller-supplied `input_tokens` used only when the
upstream reports no input count at all (main.py resolves it up front via
`count_anthropic_tokens`). Note: v0.11.0 still emits `input_tokens: 0` in the
non-streaming path when nothing is known, so these tests pin that as the
documented fallback rather than key omission.

Discriminates: passes on v0.11.0, fails on v0.10.2, where the usage block is a
flat `prompt_tokens`/`completion_tokens` copy that never subtracts cached
tokens, never reads the Anthropic-native or `prompt_tokens_details` keys, drops
the detail fields, and raises on a null `usage`.
"""

from __future__ import annotations

import asyncio
import json
from types import ModuleType

import pytest

pytestmark = pytest.mark.regression

STREAM_TIMEOUT_SECONDS = 10


@pytest.fixture(scope="session")
def anthropic_module(owui_module) -> ModuleType:
    return owui_module("open_webui.utils.anthropic")


def _openai_response(usage: dict | None, text: str = "hi") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "claude-sonnet-4-5",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": usage,
    }


async def _sse_stream(text: str, usage: dict | None):
    """Minimal OpenAI SSE stream: one text delta, a stop, then a usage-only chunk."""
    yield f'data: {json.dumps({"choices": [{"delta": {"content": text}}]})}\n\n'.encode()
    yield f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'.encode()
    yield f'data: {json.dumps({"choices": [], "usage": usage})}\n\n'.encode()
    yield b"data: [DONE]\n\n"


async def _collect_stream_usage(module: ModuleType, text: str, usage: dict | None) -> dict:
    """Drive the streaming converter to exhaustion and return the message_delta usage."""

    async def _run() -> dict:
        events = []
        async for raw in module.openai_stream_to_anthropic_stream(
            _sse_stream(text, usage), model="claude-sonnet-4-5"
        ):
            events.append(raw.decode() if isinstance(raw, bytes) else raw)
        for event in events:
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                if payload.get("type") == "message_delta":
                    return payload["usage"]
        raise AssertionError("streaming converter emitted no message_delta")

    # The source generator is finite; wait_for is only a backstop.
    return await asyncio.wait_for(_run(), timeout=STREAM_TIMEOUT_SECONDS)


async def _collect_stream_text(module: ModuleType, text: str, usage: dict | None) -> str:
    """Concatenate the text carried by the emitted content_block_delta events."""

    async def _run() -> str:
        parts = []
        async for raw in module.openai_stream_to_anthropic_stream(
            _sse_stream(text, usage), model="claude-sonnet-4-5"
        ):
            event = raw.decode() if isinstance(raw, bytes) else raw
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                if payload.get("type") != "content_block_delta":
                    continue
                delta = payload["delta"]
                if delta.get("type") == "text_delta":
                    parts.append(delta["text"])
        return "".join(parts)

    return await asyncio.wait_for(_run(), timeout=STREAM_TIMEOUT_SECONDS)


# -----------------------------------------------------------------------------
# Narrow: exactly the reporting bug
# -----------------------------------------------------------------------------


def test_cache_read_is_subtracted_from_prompt_tokens(anthropic_module: ModuleType) -> None:
    """OpenAI prompt_tokens includes cached tokens, Anthropic input_tokens does not."""
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {"prompt_tokens": 1000, "completion_tokens": 50, "cache_read_input_tokens": 800}
        )
    )
    usage = result["usage"]
    assert usage["input_tokens"] == 200, f"cached tokens double-counted: {usage}"
    assert usage["cache_read_input_tokens"] == 800
    assert usage["output_tokens"] == 50


def test_cache_creation_is_also_subtracted(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 7,
                "cache_creation_input_tokens": 300,
                "cache_read_input_tokens": 200,
            }
        )
    )
    usage = result["usage"]
    assert usage["input_tokens"] == 500, f"cache tokens double-counted: {usage}"
    assert usage["cache_creation_input_tokens"] == 300
    assert usage["cache_read_input_tokens"] == 200


def test_anthropic_native_token_keys_are_reported_verbatim(anthropic_module: ModuleType) -> None:
    """An upstream reporting input_tokens/output_tokens used to come out 0/0."""
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"input_tokens": 321, "output_tokens": 45})
    )
    assert result["usage"]["input_tokens"] == 321
    assert result["usage"]["output_tokens"] == 45


def test_prompt_tokens_details_cached_tokens_is_promoted(anthropic_module: ModuleType) -> None:
    """OpenAI-compatible upstreams report cache hits under prompt_tokens_details."""
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 600},
            }
        )
    )
    usage = result["usage"]
    assert usage["cache_read_input_tokens"] == 600, f"cached_tokens dropped: {usage}"
    assert usage["input_tokens"] == 400


def test_explicit_cache_read_wins_over_prompt_tokens_details(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "cache_read_input_tokens": 700,
                "prompt_tokens_details": {"cached_tokens": 600},
            }
        )
    )
    assert result["usage"]["cache_read_input_tokens"] == 700
    assert result["usage"]["input_tokens"] == 300


def test_detail_fields_survive_the_conversion(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 2},
                "server_tool_use": {"web_search_requests": 4},
                "service_tier": "standard",
            }
        )
    )
    usage = result["usage"]
    assert usage["output_tokens_details"] == {"reasoning_tokens": 2}
    assert usage["server_tool_use"] == {"web_search_requests": 4}
    assert usage["service_tier"] == "standard"


def test_null_usage_does_not_raise(anthropic_module: ModuleType) -> None:
    """`usage: null` from an upstream used to hit None.get()."""
    result = anthropic_module.convert_openai_to_anthropic_response(_openai_response(None))
    assert result["usage"]["input_tokens"] == 0
    assert result["usage"]["output_tokens"] == 0


def test_caller_supplied_input_tokens_is_used_when_upstream_reports_none(
    anthropic_module: ModuleType,
) -> None:
    """main.py pre-counts input tokens; that value fills in an unknown count.

    Pre-fix this fails on the absent parameter rather than on a wrong number,
    the parameter itself being the fix.
    """
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"completion_tokens": 9}), "", 1234
    )
    assert result["usage"]["input_tokens"] == 1234
    assert result["usage"]["output_tokens"] == 9

    # An upstream that does report an input count wins over the caller's estimate.
    reported = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"prompt_tokens": 87, "completion_tokens": 9}), "", 1234
    )
    assert reported["usage"]["input_tokens"] == 87


# -----------------------------------------------------------------------------
# Broad: the streaming converter must report the same numbers
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 1000, "completion_tokens": 50, "cache_read_input_tokens": 800},
        {"input_tokens": 321, "output_tokens": 45},
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
        {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "server_tool_use": {"web_search_requests": 4},
            "service_tier": "standard",
        },
    ],
)
async def test_streaming_usage_matches_non_streaming(
    anthropic_module: ModuleType, usage: dict
) -> None:
    streamed = await _collect_stream_usage(anthropic_module, "hi", usage)
    batched = anthropic_module.convert_openai_to_anthropic_response(_openai_response(usage))[
        "usage"
    ]
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens_details",
        "server_tool_use",
        "service_tier",
    ):
        assert streamed.get(key) == batched.get(key), (
            f"streaming and non-streaming disagree on {key}: "
            f"{streamed.get(key)!r} vs {batched.get(key)!r}"
        )


# -----------------------------------------------------------------------------
# Nearby: correct on both refs, proves the fix did not over-correct
# -----------------------------------------------------------------------------


def test_uncached_request_reports_prompt_tokens_unchanged(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"prompt_tokens": 87, "completion_tokens": 12})
    )
    assert result["usage"]["input_tokens"] == 87
    assert result["usage"]["output_tokens"] == 12


def test_missing_usage_key_reports_zeroes(anthropic_module: ModuleType) -> None:
    response = _openai_response(None)
    del response["usage"]
    result = anthropic_module.convert_openai_to_anthropic_response(response)
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_null_service_tier_is_omitted(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"prompt_tokens": 10, "completion_tokens": 2, "service_tier": None})
    )
    assert "service_tier" not in result["usage"]


def test_non_dict_detail_fields_are_omitted(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response(
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "server_tool_use": "not-a-dict",
                "output_tokens_details": 5,
            }
        )
    )
    assert "server_tool_use" not in result["usage"]
    assert "output_tokens_details" not in result["usage"]


def test_content_and_stop_reason_are_unaffected(anthropic_module: ModuleType) -> None:
    result = anthropic_module.convert_openai_to_anthropic_response(
        _openai_response({"prompt_tokens": 5, "completion_tokens": 1}, text="hello")
    )
    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert result["stop_reason"] == "end_turn"
    assert result["role"] == "assistant"


@pytest.mark.asyncio
async def test_streaming_still_emits_the_text_delta(anthropic_module: ModuleType) -> None:
    usage = {"prompt_tokens": 5, "completion_tokens": 1}
    assert await _collect_stream_text(anthropic_module, "hello", usage) == "hello"
