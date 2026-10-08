"""Regression: outlet filters could not tell why the model stopped answering.

PR #32079 (commit 039867feb) hands outlet filters the finish reason the provider reported, as
`finish_reason` on the reply: for a chat in the web client with streaming on or off, for an API
client's call streamed or not, and for Ollama models, whose replies cut off by the token limit
now report `length` from Ollama's `done_reason`. Only the reply carries it, never the earlier
messages of the chat. A global outlet filter here writes the reason it was handed into each
message that carries one, which a chat stores, and logs the reply's reason, which is all an API
client's call leaves behind.

Discriminates: passes on dev 87a937459. In a backend copy with 039867feb reverted every test
goes red (the filter is handed no reason), the earlier-message test at its first reply. With only
its Ollama conversion reverted the two Ollama tests go red (the filter is handed stop).
"""

from __future__ import annotations

import textwrap
import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.instance import LaunchedInstance
from harness.listener import json_answer
from harness.ollama_provider import (
    OLLAMA_CONFIG,
    chat_line,
    connect_ollama,
    ndjson,
    serve_ollama,
)
from harness.plugins import installed_function

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"

FINISH_REASON_FILTER = textwrap.dedent(
    """
    class Filter:
        def outlet(self, body):
            for message in body["messages"]:
                reason = message.get("finish_reason")
                if reason:
                    message["content"] = f"{message['content']} [finish: {reason}]"
            prompt = body["messages"][-2]["content"]
            reason = body["messages"][-1].get("finish_reason")
            print(f"outlet saw finish reason {reason} for {prompt}", flush=True)
            return body
    """
).lstrip()


@pytest.fixture
def finish_reason_filter(admin):
    with installed_function(admin, FINISH_REASON_FILTER, is_global=True):
        yield


def prompt() -> str:
    return f"tell me about the tides {uuid.uuid4().hex[:8]}"


def stored_content(client, turn: ChatTurn, needle: str, timeout: float = 15.0) -> str:
    """The stored reply once it contains `needle`; outlet filters edit it after it finishes."""
    wait_for_reply(client, turn)
    deadline = time.monotonic() + timeout
    while True:
        messages = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]["messages"]
        content = messages[turn.assistant_message_id].get("content") or ""
        if needle in content or time.monotonic() > deadline:
            return content
        time.sleep(0.2)


def logged_reason(instance: LaunchedInstance, offset: int, question: str) -> str:
    """The finish reason the filter logged for `question`, waiting for the outlet to run."""
    marker = f"for {question}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        for line in instance.log_since(offset).splitlines():
            if line.endswith(marker) and "outlet saw finish reason " in line:
                return line.split("outlet saw finish reason ", 1)[1].removesuffix(f" {marker}")
        time.sleep(0.2)
    raise AssertionError(f"the outlet filter never ran for {question!r}")


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_a_streamed_chat_hands_the_outlet_the_providers_finish_reason(
    finish_reason_filter, make_user, upstream, finish_reason
):
    question = prompt()
    upstream.queue(
        reply.text(
            ["High water ", "is at six"],
            finish_reason=finish_reason,
            match=reply.answering(question),
        )
    )
    with make_user().client() as client:
        turn = send_message(client, question)
        content = stored_content(client, turn, "[finish:")

    assert content.endswith(f"[finish: {finish_reason}]"), content


def test_a_chat_with_streaming_off_hands_the_outlet_the_providers_finish_reason(
    finish_reason_filter, make_user, upstream
):
    question = prompt()
    upstream.queue(
        reply.text("High water is at six", finish_reason="length", match=reply.answering(question))
    )
    with make_user().client() as client:
        turn = send_message(client, question, stream=False)
        content = stored_content(client, turn, "[finish:")

    assert content.endswith("[finish: length]"), content


def test_only_the_reply_carries_a_finish_reason(finish_reason_filter, make_user, upstream):
    first, second = prompt(), prompt()
    upstream.queue(
        reply.text("High water is", finish_reason="length", match=reply.answering(first)),
        reply.text("at six", match=reply.answering(second)),
    )
    with make_user().client() as client:
        first_turn = send_message(client, first)
        first_reply = stored_content(client, first_turn, "[finish:")
        history = [
            {"role": "user", "content": first},
            {"role": "assistant", "content": first_reply},
        ]
        second_turn = send_message(
            client,
            second,
            chat_id=first_turn.chat_id,
            parent_id=first_turn.assistant_message_id,
            history=history,
        )
        second_reply = stored_content(client, second_turn, "[finish:")
        messages = client.get(f"/api/v1/chats/{first_turn.chat_id}").json()["chat"]["history"][
            "messages"
        ]

    assert first_reply.endswith("[finish: length]"), first_reply
    assert second_reply.endswith("[finish: stop]"), second_reply
    earlier = messages[first_turn.assistant_message_id]["content"]
    assert earlier.count("[finish:") == 1, (
        f"an earlier reply was handed a finish reason again: {earlier}"
    )


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "not-streamed"])
def test_an_api_clients_call_hands_the_outlet_the_providers_finish_reason(
    finish_reason_filter, make_user, upstream, instance, stream
):
    question = prompt()
    upstream.queue(
        reply.text("High water is at six", finish_reason="length", match=reply.answering(question))
    )
    offset = instance.log_size()
    with make_user().client() as client:
        answered = client.post(
            "/api/chat/completions",
            json={
                "model": reply.MOCK_MODEL_ID,
                "stream": stream,
                "messages": [{"role": "user", "content": question}],
            },
        )
    assert answered.status_code == 200, answered.text

    assert logged_reason(instance, offset, question) == "length"


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
    return server


@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "not-streamed"])
def test_an_ollama_reply_cut_off_by_the_token_limit_hands_the_outlet_length(
    finish_reason_filter, ollama, admin, stream
):
    cut_off = {"done_reason": "length", "prompt_eval_count": 4, "eval_count": 8}
    if stream:
        ollama.queue_chat(
            ndjson(
                chat_line(OLLAMA_MODEL, {"content": "High water is"}),
                chat_line(OLLAMA_MODEL, {"content": ""}, **cut_off),
            )
        )
    else:
        ollama.queue_chat(
            json_answer(chat_line(OLLAMA_MODEL, {"content": "High water is"}, **cut_off))
        )
    with admin.client() as client:
        turn = send_message(client, prompt(), model=OLLAMA_MODEL, stream=stream)
        content = stored_content(client, turn, "[finish:")

    assert content.endswith("[finish: length]"), content
