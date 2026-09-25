"""Journey: managing and using the models of an Ollama connection through Open WebUI.

The admin's Ollama settings pull a model with streamed progress, create one from a base, copy
and delete one, read a model's details and the server version, and unload a running model from
the model selector. Completions and embeddings go through the same proxy. Each call is checked
on both sides: what Open WebUI answered and what the Ollama stand-in was sent.

Discriminates: in a backend copy, `pull_model` dropping `insecure`, `copy_model` sending to
`/api/show` and `/api/models/unload` leaving out `keep_alive` each turn one test red.
"""

from __future__ import annotations

import json

import pytest

from harness.ollama_provider import (
    EMBEDDING,
    GENERATED,
    OLLAMA_CONFIG,
    OLLAMA_VERSION,
    connect_ollama,
    serve_ollama,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

BASE_MODEL = "llama3:latest"


@pytest.fixture
def ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, BASE_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        yield server, client


def _listed(client) -> list[str]:
    tags = client.get("/ollama/api/tags/0")  # the uncached list the settings page reads
    assert tags.status_code == 200, tags.text
    return [model["name"] for model in tags.json()["models"]]


def _progress(streamed) -> list[dict]:
    return [json.loads(line) for line in streamed.text.splitlines() if line.strip()]


def test_a_pulled_model_streams_its_progress_and_is_listed(ollama):
    server, client = ollama

    pulled = client.post("/ollama/api/pull/0", json={"name": "qwen3:0.6b"})

    assert pulled.status_code == 200, pulled.text
    progress = _progress(pulled)
    assert progress[-1] == {"status": "success"}
    assert any(line.get("completed") == 50 for line in progress), progress
    assert server.sent("/api/pull") == [
        {"name": "qwen3:0.6b", "model": "qwen3:0.6b", "insecure": True}
    ]
    assert "qwen3:0.6b" in _listed(client)


def test_a_model_created_from_a_base_is_listed(ollama):
    server, client = ollama
    recipe = {"model": "brief:latest", "from": BASE_MODEL, "system": "Answer in one line."}

    created = client.post("/ollama/api/create/0", json=recipe)

    assert created.status_code == 200, created.text
    assert _progress(created)[-1] == {"status": "success"}
    assert server.sent("/api/create") == [recipe]
    assert "brief:latest" in _listed(client)


def test_a_copied_model_is_listed_and_a_deleted_one_is_gone(ollama):
    server, client = ollama

    copied = client.post(
        "/ollama/api/copy", json={"source": BASE_MODEL, "destination": "backup:latest"}
    )
    deleted = client.request("DELETE", "/ollama/api/delete", json={"model": BASE_MODEL})

    assert copied.status_code == 200 and copied.json() is True, copied.text
    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    assert server.sent("/api/copy") == [{"source": BASE_MODEL, "destination": "backup:latest"}]
    assert server.sent("/api/delete") == [{"model": BASE_MODEL}]
    assert _listed(client) == ["backup:latest"]


def test_deleting_a_model_ollama_does_not_have_is_refused(ollama):
    server, client = ollama

    refused = client.request("DELETE", "/ollama/api/delete/0", json={"model": "ghost:latest"})

    assert refused.status_code == 404, refused.text
    assert "ghost:latest" in refused.json()["detail"]
    assert _listed(client) == [BASE_MODEL]


def test_the_model_details_and_the_version_come_from_ollama(ollama):
    server, client = ollama

    shown = client.post("/ollama/api/show", json={"name": BASE_MODEL})
    version = client.get("/ollama/api/version")

    assert shown.status_code == 200, shown.text
    assert shown.json()["details"]["family"] == "llama"
    assert server.sent("/api/show") == [{"name": BASE_MODEL, "model": BASE_MODEL}]
    assert version.status_code == 200, version.text
    assert version.json() == {"version": OLLAMA_VERSION}


def test_a_completion_and_embeddings_are_proxied(ollama):
    server, client = ollama

    generated = client.post(
        "/ollama/api/generate", json={"model": BASE_MODEL, "prompt": "hello", "stream": False}
    )
    embedded = client.post("/ollama/api/embed", json={"model": BASE_MODEL, "input": ["a", "b"]})

    assert generated.status_code == 200, generated.text
    assert generated.json()["response"] == GENERATED
    assert server.sent("/api/generate")[-1]["prompt"] == "hello"
    assert embedded.status_code == 200, embedded.text
    assert embedded.json()["embeddings"] == [EMBEDDING, EMBEDDING]


def test_unloading_a_running_model_frees_it_in_ollama(ollama):
    server, client = ollama
    server.loaded.append(BASE_MODEL)
    client.get("/api/models").raise_for_status()  # registers the connection's models

    unloaded = client.post("/api/models/unload", json={"model": BASE_MODEL})

    assert unloaded.status_code == 200, unloaded.text
    assert server.sent("/api/generate") == [{"model": BASE_MODEL, "keep_alive": 0, "prompt": ""}]
    assert server.loaded == []
