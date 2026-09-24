"""The Anthropic-compatible /api/v1/messages endpoint reported the wrong token usage.

Fix commits e8b59b2, 51ff386, 0576e8e, 8e74cac, 93a34bb and 4c2d864 in
`utils/anthropic.py` (#26790, #27293, docs#1328). The endpoint turns the provider's
OpenAI-shaped usage into Anthropic's block. It used to copy `prompt_tokens` and
`completion_tokens` across, so cached tokens (inside `prompt_tokens`, outside Anthropic's
`input_tokens`) were counted twice, a provider already reporting `input_tokens` came out as
0, cache hits under `prompt_tokens_details` and the detail fields were dropped, the stream
reported no input count at all and a null `usage` raised. The block is now derived, and the
input count the endpoint asks the provider for fills in when the reply carries none.

Twin of unit/chat/test_anthropic_usage_reporting.py.

Discriminates: with both converters reverted to the v0.10.2 flat copy, every narrow case
fails on both paths, the null usage answers 500 and the streamed nearby cases lose their
input count; the batched nearby cases and the reply text pass on both.
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness import raw_provider
from harness import upstream as reply
from harness.raw_provider import OPENAI_CONNECTIONS, RAW_MODEL_ID
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

STREAMING = [pytest.param(False, id="batched"), pytest.param(True, id="streamed")]


def send_messages(client: httpx.Client, model: str, *, stream: bool) -> httpx.Response:
    request = {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text
    return response


def stream_events(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def reported_usage(response: httpx.Response, *, stream: bool) -> dict:
    if not stream:
        return response.json()["usage"]
    deltas = [event for event in stream_events(response) if event["type"] == "message_delta"]
    assert len(deltas) == 1, deltas
    return deltas[0]["usage"]


@pytest.fixture
def raw(admin, preserve, listener) -> raw_provider.RawProvider:
    preserve(OPENAI_CONNECTIONS)
    return raw_provider.connect(admin, listener)


def raw_completion(usage) -> dict:
    message = {"role": "assistant", "content": "hi"}
    choice = {"index": 0, "message": message, "finish_reason": "stop"}
    return {"object": "chat.completion", "model": RAW_MODEL_ID, "choices": [choice], "usage": usage}


USAGE_CASES = [
    pytest.param(
        {"prompt_tokens": 1000, "completion_tokens": 50, "cache_read_input_tokens": 800},
        {"input_tokens": 200, "output_tokens": 50, "cache_read_input_tokens": 800},
        id="cache-read-subtracted",
    ),
    pytest.param(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 7,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 200,
        },
        {
            "input_tokens": 500,
            "output_tokens": 7,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 200,
        },
        id="cache-creation-subtracted",
    ),
    pytest.param(
        {"input_tokens": 321, "output_tokens": 45},
        {"input_tokens": 321, "output_tokens": 45},
        id="anthropic-native-keys",
    ),
    pytest.param(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
        {"input_tokens": 400, "output_tokens": 10, "cache_read_input_tokens": 600},
        id="cached-tokens-promoted",
    ),
    pytest.param(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "cache_read_input_tokens": 700,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
        {"input_tokens": 300, "output_tokens": 10, "cache_read_input_tokens": 700},
        id="explicit-cache-read-wins",
    ),
    pytest.param(
        {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "server_tool_use": {"web_search_requests": 4},
            "service_tier": "standard",
        },
        {
            "input_tokens": 12,
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "server_tool_use": {"web_search_requests": 4},
            "service_tier": "standard",
        },
        id="detail-fields-kept",
    ),
]

# Correct on every ref: an uncached request and malformed detail fields.
NEARBY_USAGE_CASES = [
    pytest.param(
        {"prompt_tokens": 87, "completion_tokens": 12},
        {"input_tokens": 87, "output_tokens": 12},
        id="uncached",
    ),
    pytest.param(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "service_tier": None,
            "server_tool_use": "not-a-dict",
            "output_tokens_details": 5,
        },
        {"input_tokens": 10, "output_tokens": 2},
        id="malformed-details-omitted",
    ),
]


@pytest.mark.parametrize("stream", STREAMING)
@pytest.mark.parametrize("provider_usage,expected", USAGE_CASES + NEARBY_USAGE_CASES)
def test_usage_is_reported_in_anthropic_terms(user, upstream, provider_usage, expected, stream):
    upstream.queue(reply.text("hi", usage=provider_usage))
    with user.client() as client:
        response = send_messages(client, MOCK_MODEL_ID, stream=stream)
    assert reported_usage(response, stream=stream) == expected


@pytest.mark.parametrize("usage", [None, "absent"])
def test_a_reply_without_usage_reports_zeroes(admin, raw, usage):
    completion = raw_completion(usage)
    if usage == "absent":
        del completion["usage"]
    raw.complete(completion)
    with admin.client() as client:
        response = send_messages(client, RAW_MODEL_ID, stream=False)
    assert response.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize(
    "provider_usage,expected_input_tokens",
    [
        pytest.param({"completion_tokens": 9}, 1234, id="counted-when-unreported"),
        pytest.param({"prompt_tokens": 87, "completion_tokens": 9}, 87, id="reported-wins"),
    ],
)
def test_the_counted_input_tokens_fill_a_missing_input_count(
    admin, raw, provider_usage, expected_input_tokens
):
    raw.count_tokens(1234)
    raw.complete(raw_completion(provider_usage))
    with admin.client() as client:
        response = send_messages(client, RAW_MODEL_ID, stream=False)
    assert response.json()["usage"] == {"input_tokens": expected_input_tokens, "output_tokens": 9}


def test_the_counted_input_tokens_reach_the_stream(admin, raw):
    raw.count_tokens(1234)
    raw.stream(
        raw_provider.sse(
            raw_provider.chunk({"role": "assistant", "content": "hi"}),
            raw_provider.chunk({}, "stop", usage={"completion_tokens": 9}),
        )
    )
    with admin.client() as client:
        response = send_messages(client, RAW_MODEL_ID, stream=True)
    events = stream_events(response)
    assert events[0]["message"]["usage"]["input_tokens"] == 1234
    assert reported_usage(response, stream=True) == {"input_tokens": 1234, "output_tokens": 9}


@pytest.mark.parametrize("stream", STREAMING)
def test_reply_text_and_stop_reason_are_unaffected(user, upstream, stream):
    upstream.queue(reply.text("hello", usage={"prompt_tokens": 5, "completion_tokens": 1}))
    with user.client() as client:
        response = send_messages(client, MOCK_MODEL_ID, stream=stream)
    if stream:
        events = stream_events(response)
        text = "".join(
            event["delta"]["text"]
            for event in events
            if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
        )
        stop_reasons = [
            event["delta"]["stop_reason"] for event in events if event["type"] == "message_delta"
        ]
    else:
        body = response.json()
        assert body["role"] == "assistant"
        text = "".join(block["text"] for block in body["content"] if block["type"] == "text")
        stop_reasons = [body["stop_reason"]]
    assert text == "hello"
    assert stop_reasons == ["end_turn"]
