"""Regression: a prompt cache marker on a tool result never reached the provider.

Fix 1be39d2f3 (PR open-webui/open-webui#33757, issue #33755). An Anthropic client sends a tool
result through /api/v1/messages with `cache_control` on one of its blocks, so a gateway such as
OpenRouter or LiteLLM caches the prefix of a tool loop. Right before the Chat Completions request
left, Open WebUI flattened every tool message whose content is a list into a plain string, which
dropped the marker, so the prefix was never cached. A tool result carrying a marker now goes out
as one text block that keeps it, images still stripped; one without a marker is sent as before.

Discriminates: passes on dev f659c0d12, fails on a backend copy with 1be39d2f3 reverted (the tool
message reaches the provider as a plain string without the marker); the nearby tests pass on both.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.image_engines import PNG_BASE64
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

EPHEMERAL = {"type": "ephemeral"}
FORECAST = "Vienna: 21 degrees and sunny."
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64},
}


def tool_loop_tool_message(user, upstream, result_content) -> dict:
    """Send a tool loop whose tool result is `result_content`; returns the tool message the
    provider received for it."""
    question = f"What is the weather in Vienna? [{uuid.uuid4().hex[:8]}]"
    upstream.queue(reply.text("It is sunny.", match=reply.mentioning(question)))
    request = {
        "model": MOCK_MODEL_ID,
        "max_tokens": 64,
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Vienna"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": result_content}
                ],
            },
        ],
    }
    with user.client() as client:
        response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text
    [sent] = [
        body
        for body in upstream.chat_requests()
        if any(question == message.get("content") for message in body.get("messages", []))
    ]
    [tool_message] = [message for message in sent["messages"] if message["role"] == "tool"]
    return tool_message


# ---------------------------------------------------------------- narrow


@pytest.mark.parametrize("extra_blocks", [[], [IMAGE_BLOCK]], ids=["text-only", "with-an-image"])
def test_a_cache_marker_on_a_tool_result_reaches_the_provider(user, upstream, extra_blocks):
    marked = {"type": "text", "text": FORECAST, "cache_control": EPHEMERAL}

    tool_message = tool_loop_tool_message(user, upstream, [*extra_blocks, marked])

    assert tool_message["content"] == [
        {"type": "text", "text": FORECAST, "cache_control": EPHEMERAL}
    ], (
        "the tool result's prompt cache marker was dropped before the request left, so the "
        f"provider never caches the tool loop (#33755); it got {tool_message['content']!r}"
    )


# ---------------------------------------------------------------- nearby


def test_a_tool_result_without_a_marker_is_still_sent_as_text(user, upstream):
    blocks = [IMAGE_BLOCK, {"type": "text", "text": "Vienna: "}, {"type": "text", "text": "sunny."}]

    tool_message = tool_loop_tool_message(user, upstream, blocks)

    assert tool_message["content"] == "Vienna: sunny."


def test_a_plain_text_tool_result_is_sent_unchanged(user, upstream):
    tool_message = tool_loop_tool_message(user, upstream, FORECAST)

    assert tool_message["content"] == FORECAST
