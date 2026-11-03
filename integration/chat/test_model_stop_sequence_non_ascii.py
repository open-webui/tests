"""Regression: a model's Stop Sequence with non-ASCII text reached the provider garbled.

Issue open-webui/open-webui#33610. A request to `/openai/chat/completions` or `/ollama/api/chat`
for a workspace model fills in the model's saved Stop Sequence, unescaped so a typed `\\n`
becomes a newline (#1747). The unescaping read the text byte by byte, so `終` went out as
`çµ\\x82` and `café` as `cafÃ©`, and the model never stopped on them.

Discriminates: fails on dev d22c4449c for both providers (the stops arrive garbled), passes with
the fix for #33610 applied.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

OLLAMA_MODEL = "llama3:latest"
# as the model editor saves them; the last one is typed with a backslash
SAVED_STOPS = ["終", "café", "\\n\\n"]
EXPECTED_STOPS = ["終", "café", "\n\n"]


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_a_models_non_ascii_stop_sequence_reaches_the_provider_unchanged(
    admin, upstream, preserve, listener, provider
):
    with admin.client() as client:
        if provider == "ollama":
            preserve(OLLAMA_CONFIG)
            ollama = serve_ollama(listener, OLLAMA_MODEL)
            connect_ollama(client, listener)
            ollama.queue_chat(chat_stream(OLLAMA_MODEL, {"content": "ok"}))
            base_model_id, path = OLLAMA_MODEL, "/ollama/api/chat"
        else:
            upstream.queue(reply.text("ok"))
            base_model_id, path = MOCK_MODEL_ID, "/openai/chat/completions"
        client.get("/api/models").raise_for_status()

        preset_id = f"stops-{uuid.uuid4().hex[:8]}"
        form = {
            "id": preset_id,
            "name": preset_id,
            "base_model_id": base_model_id,
            "meta": {},
            "params": {"stop": SAVED_STOPS},
        }
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        try:
            client.get("/api/models").raise_for_status()
            answered = client.post(
                path,
                json={
                    "model": preset_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
        finally:
            client.post("/api/v1/models/model/delete", json={"id": preset_id})

    assert answered.status_code == 200, answered.text
    if provider == "ollama":
        sent_stops = ollama.chat_requests()[-1]["options"].get("stop")
    else:
        sent_stops = upstream.chat_requests()[-1].get("stop")
    assert sent_stops == EXPECTED_STOPS, f"the provider got the stop sequences as {sent_stops!r}"
