"""Journey: the provider routes the API docs list for a script, reached with a user's API key.

`/openai` relays chat completions to the OpenAI connection, so the official `openai` SDK pointed
at it lists the models and chats, streamed and not. `/ollama` passes Ollama's own API through
(`/api/tags`, a streamed `/api/generate`, `/api/embed`) and its OpenAI-compatible routes
(`/v1/embeddings` and `/v1/responses`, read with the SDK), each for a model the account may use.

Discriminates: on dev 0f5a58f5f, in a backend copy, `get_current_user_by_api_key` finding no
user fails every test here, and the OpenAI router stripping `data: ` from the relayed stream
fails the OpenAI test.
"""

from __future__ import annotations

import json
from typing import Iterator

import openai
import pytest

from harness import upstream as reply
from harness.listener import json_answer
from harness.ollama_provider import (
    EMBEDDING,
    GENERATED,
    OLLAMA_CONFIG,
    OllamaServer,
    connect_ollama,
    serve_ollama,
)
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
OLLAMA_MODEL = "llama3:latest"
RESPONSE_TEXT = "Rayleigh scattering."


@pytest.fixture
def api_key(admin, make_user, preserve) -> str:
    """A fresh user's key, with API keys on and allowed to users."""
    preserve("admin_config", "permissions")
    with admin.client() as client:
        config = client.get(ADMIN_CONFIG).json()
        saved = client.post(
            ADMIN_CONFIG,
            json={
                **config,
                "ENABLE_API_KEYS": True,
                "ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS": False,
            },
        )
        assert saved.status_code == 200, saved.text
        permissions = client.get(DEFAULT_PERMISSIONS).json()
        permissions["features"]["api_keys"] = True
        saved = client.post(DEFAULT_PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text
    with make_user().client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    return generated.json()["api_key"]


def _responses_answer(request) -> tuple:
    body = request.json()
    return json_answer(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": body["model"],
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": RESPONSE_TEXT, "annotations": []}],
                }
            ],
        }
    )


def _embeddings_answer(request) -> tuple:
    inputs = request.json()["input"]
    texts = inputs if isinstance(inputs, list) else [inputs]
    vectors = [
        {"object": "embedding", "index": index, "embedding": EMBEDDING}
        for index, _ in enumerate(texts)
    ]
    return json_answer({"object": "list", "data": vectors, "model": request.json()["model"]})


@pytest.fixture
def ollama(admin, preserve, listener) -> Iterator[OllamaServer]:
    """The Ollama stand-in as the only Ollama connection, its model readable by every account."""
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, OLLAMA_MODEL)
    listener.route("POST", "/v1/embeddings", _embeddings_answer)
    listener.route("POST", "/v1/responses", _responses_answer)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models").raise_for_status()
        published = client.post(
            "/api/v1/models/model/access/update",
            json={"id": OLLAMA_MODEL, "name": OLLAMA_MODEL, "access_grants": [EVERYONE_READS]},
        )
        assert published.status_code == 200, published.text
        yield server
        client.post("/api/v1/models/model/delete", json={"id": OLLAMA_MODEL})


def test_the_openai_route_serves_the_sdk_streamed_and_not(instance, api_key, upstream):
    upstream.queue(
        reply.text("plain answer", match=reply.answering("not streamed")),
        reply.text(["streamed ", "answer"], match=reply.answering("streamed please")),
    )
    client = openai.OpenAI(api_key=api_key, base_url=f"{instance.base_url}/openai", max_retries=0)

    listed = [model.id for model in client.models.list().data]
    completion = client.chat.completions.create(
        model=MOCK_MODEL_ID, messages=[{"role": "user", "content": "not streamed"}]
    )
    stream = client.chat.completions.create(
        model=MOCK_MODEL_ID,
        messages=[{"role": "user", "content": "streamed please"}],
        stream=True,
    )
    pieces = [chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices]

    assert MOCK_MODEL_ID in listed
    assert completion.choices[0].message.content == "plain answer"
    assert "".join(pieces) == "streamed answer"


def test_ollamas_own_api_answers_an_api_key(instance, api_key, ollama):
    with instance.client(api_key) as client:
        tags = client.get("/ollama/api/tags")
        generated = client.post(
            "/ollama/api/generate", json={"model": OLLAMA_MODEL, "prompt": "Why is the sky blue?"}
        )
        embedded = client.post(
            "/ollama/api/embed", json={"model": OLLAMA_MODEL, "input": ["one", "two"]}
        )

    assert tags.status_code == 200, tags.text
    assert OLLAMA_MODEL in [model["name"] for model in tags.json()["models"]]
    assert generated.status_code == 200, generated.text
    lines = [json.loads(line) for line in generated.text.splitlines() if line]
    assert "".join(line["response"] for line in lines) == GENERATED
    assert lines[-1]["done"] is True
    assert embedded.status_code == 200, embedded.text
    assert embedded.json()["embeddings"] == [EMBEDDING, EMBEDDING]
    assert ollama.sent("/api/generate")[-1]["prompt"] == "Why is the sky blue?"


def test_ollamas_openai_compatible_routes_serve_the_sdk(instance, api_key, ollama):
    client = openai.OpenAI(
        api_key=api_key, base_url=f"{instance.base_url}/ollama/v1", max_retries=0
    )

    embeddings = client.embeddings.create(model=OLLAMA_MODEL, input="Open WebUI is great!")
    response = client.responses.create(model=OLLAMA_MODEL, input="Why is the sky blue?")

    assert embeddings.data[0].embedding == EMBEDDING
    assert response.output_text == RESPONSE_TEXT
    assert ollama.sent("/v1/embeddings")[-1]["input"] == "Open WebUI is great!"
    assert ollama.sent("/v1/responses")[-1]["input"] == "Why is the sky blue?"
