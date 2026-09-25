"""Regression: `max_tokens` from an API client never limited an Ollama model's reply.

Issue open-webui/open-webui#31432, fix PR open-webui/open-webui#31437. A request to
`/api/chat/completions` with a top-level `max_tokens` reached Ollama as a top-level
`num_predict`, which Ollama's chat endpoint does not read; it only reads it from `options`.

Discriminates: fails on dev ac00d40e3 (`num_predict` arrives outside `options`), passes with
#31437 applied.
"""

from __future__ import annotations

import pytest

from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        yield server, client


def _send(client, **body) -> None:
    client.post(
        "/api/chat/completions",
        json={
            "model": OLLAMA_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            **body,
        },
        timeout=60,
    )


def test_an_api_clients_max_tokens_reaches_ollama_as_an_option(ollama):
    server, client = ollama
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "ok"}))

    _send(client, max_tokens=32)

    sent = server.chat_requests()[-1]
    assert sent.get("options", {}).get("num_predict") == 32, sent
    assert "num_predict" not in sent


def test_it_sits_next_to_the_other_options(ollama):
    server, client = ollama
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "ok"}))

    _send(client, max_tokens=16, stop=["###"])

    options = server.chat_requests()[-1].get("options", {})
    assert options.get("num_predict") == 16
    assert options.get("stop") == ["###"]


def test_without_max_tokens_no_limit_is_sent(ollama):
    server, client = ollama
    server.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "ok"}))

    _send(client)

    sent = server.chat_requests()[-1]
    assert "num_predict" not in sent
    assert "num_predict" not in sent.get("options", {})
