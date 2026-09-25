"""Journey: the direct provider routes check model access, and Ollama model management is admin's.

A client can skip the chat endpoint and call a provider route itself: `/openai/chat/completions`,
Ollama's `/api/chat` and `/api/generate`, and Ollama's OpenAI-shaped `/v1/chat/completions`.
Each checks the model the way the chat does: a user is refused a provider model with no
workspace entry and one whose entry is shared with nobody, and the provider never sees the
request; the same user is served once the entry is shared with them. The deprecated catch-all
OpenAI proxy is off unless `ENABLE_OPENAI_API_PASSTHROUGH` is set, and when on it forwards with
the connection's key, never the caller's token. Pulling, creating, copying and deleting Ollama
models is refused to a user before Ollama is called.

Discriminates: in a backend copy, letting `check_model_access` in `utils/access_control` pass a
user when the model has no workspace entry turns the four `unregistered` rows red (the provider
answers the user), dropping its read-grant check turns the four `unshared` rows red, switching
`pull_model` to `get_verified_user` turns both pull rows of the management test red (Ollama pulls
for the user), and copying the caller's token into a header of the catch-all proxy's request
turns the passthrough test red.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import pytest

from harness.actors import create_user
from harness.listener import json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

OPENAI_MODEL = "gated-provider-model"
OLLAMA_MODEL = "gated:latest"
HELLO = [{"role": "user", "content": "hello"}]
OLLAMA_COMPLETION_PATHS = {"/api/chat", "/api/generate", "/v1/chat/completions"}
PASSTHROUGH = {"ENABLE_OPENAI_API_PASSTHROUGH": "true"}
REFUSED = {401, 403}

# route: (provider, path, body)
ROUTES = {
    "openai chat": (
        "openai",
        "/openai/chat/completions",
        {"model": OPENAI_MODEL, "messages": HELLO, "stream": False},
    ),
    "ollama chat": (
        "ollama",
        "/ollama/api/chat",
        {"model": OLLAMA_MODEL, "messages": HELLO, "stream": False},
    ),
    "ollama generate": (
        "ollama",
        "/ollama/api/generate",
        {"model": OLLAMA_MODEL, "prompt": "hello", "stream": False},
    ),
    "ollama openai chat": (
        "ollama",
        "/ollama/v1/chat/completions",
        {"model": OLLAMA_MODEL, "messages": HELLO, "stream": False},
    ),
}


def _completion(model: str) -> dict:
    message = {"role": "assistant", "content": "pong"}
    return {
        "id": "ollama",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


@pytest.fixture
def providers(admin, preserve, listener, upstream):
    """The scripted provider serving OPENAI_MODEL and an Ollama stand-in serving OLLAMA_MODEL."""
    preserve(OLLAMA_CONFIG)
    ollama = serve_ollama(listener, OLLAMA_MODEL)
    listener.route("POST", "/v1/chat/completions", json_answer(_completion(OLLAMA_MODEL)))
    upstream.models.append(OPENAI_MODEL)
    with admin.client() as client:
        connect_ollama(client, listener)
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield ollama
    upstream.models.remove(OPENAI_MODEL)
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"})


@contextlib.contextmanager
def _workspace_entry(admin, model_id: str, access_grants: list[dict]) -> Iterator[None]:
    """The admin's workspace entry for a provider model, shared with `access_grants`."""
    form = {
        "id": model_id,
        "base_model_id": None,
        "name": model_id,
        "meta": {},
        "params": {},
        "access_grants": access_grants,
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        try:
            yield
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _read_grant(actor) -> dict:
    return {"principal_type": "user", "principal_id": actor.id, "permission": "read"}


def _provider_calls(provider: str, upstream, listener) -> list[str]:
    """The models the provider behind `provider` was asked to complete for."""
    if provider == "openai":
        return [body["model"] for body in upstream.chat_requests()]
    return [
        request.json()["model"]
        for request in listener.received
        if request.method == "POST" and request.path in OLLAMA_COMPLETION_PATHS
    ]


def _entry_for(provider: str) -> str:
    return OPENAI_MODEL if provider == "openai" else OLLAMA_MODEL


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("entry", ["unregistered", "unshared"])
def test_a_user_is_refused_a_model_they_cannot_read(
    route, entry, providers, admin, make_user, upstream, listener
):
    provider, path, body = ROUTES[route]
    with contextlib.ExitStack() as stack:
        if entry == "unshared":
            stack.enter_context(_workspace_entry(admin, _entry_for(provider), []))
        with make_user().client() as client:
            answered = client.post(path, json=body)

    assert answered.status_code in REFUSED, (
        f"a user reached the {entry} {_entry_for(provider)} over {path}: "
        f"HTTP {answered.status_code} {answered.text[:200]}"
    )
    assert _provider_calls(provider, upstream, listener) == []


@pytest.mark.parametrize("route", ROUTES)
def test_a_user_the_model_is_shared_with_is_served(
    route, providers, admin, make_user, upstream, listener
):
    provider, path, body = ROUTES[route]
    reader = make_user()
    with _workspace_entry(admin, _entry_for(provider), [_read_grant(reader)]):
        with reader.client() as client:
            answered = client.post(path, json=body)

    assert answered.status_code == 200, answered.text
    assert _provider_calls(provider, upstream, listener) == [_entry_for(provider)]


@pytest.mark.parametrize("route", ROUTES)
def test_an_admin_uses_an_unregistered_model(route, providers, admin, upstream, listener):
    provider, path, body = ROUTES[route]
    with admin.client() as client:
        answered = client.post(path, json=body)

    assert answered.status_code == 200, answered.text
    assert _provider_calls(provider, upstream, listener) == [_entry_for(provider)]


# method, path, body, the Ollama path it would reach
MANAGEMENT = [
    ("POST", "/ollama/api/pull", {"name": "qwen3:0.6b"}, "/api/pull"),
    ("POST", "/ollama/api/create", {"model": "brief:latest", "from": OLLAMA_MODEL}, "/api/create"),
    (
        "POST",
        "/ollama/api/copy",
        {"source": OLLAMA_MODEL, "destination": "backup:latest"},
        "/api/copy",
    ),
    ("DELETE", "/ollama/api/delete", {"model": OLLAMA_MODEL}, "/api/delete"),
]


@pytest.mark.parametrize("url_index", ["", "/0"], ids=["any-connection", "connection-0"])
@pytest.mark.parametrize(
    "method, path, body, ollama_path", MANAGEMENT, ids=[row[1] for row in MANAGEMENT]
)
def test_a_user_cannot_manage_ollama_models(
    method, path, body, ollama_path, url_index, providers, make_user, listener
):
    with make_user().client() as client:
        answered = client.request(method, path + url_index, json=body)

    assert answered.status_code in REFUSED, (
        f"a user was not refused {method} {path}{url_index}: HTTP {answered.status_code}"
    )
    assert listener.requests_to(ollama_path) == [], f"Ollama was sent {ollama_path} for a user"
    assert providers.models == [OLLAMA_MODEL]


def test_the_catch_all_proxy_is_off_by_default(make_user, upstream):
    with make_user().client() as client:
        answered = client.post("/openai/embeddings", json={"model": MOCK_MODEL_ID, "input": "hi"})

    assert answered.status_code == 403, answered.text
    assert upstream.requests_to("/embeddings") == []


@pytest.mark.slow
def test_the_catch_all_proxy_forwards_with_the_connections_key(instance_with):
    extra = instance_with(PASSTHROUGH)
    caller = create_user(extra, name=f"Caller {uuid.uuid4().hex[:8]}")

    with caller.client() as client:
        answered = client.post("/openai/embeddings", json={"model": MOCK_MODEL_ID, "input": "hi"})

    assert answered.status_code == 200, answered.text
    [forwarded] = extra.upstream.requests_to("/embeddings")
    assert forwarded.headers.get("Authorization") == "Bearer sk-mock"
    leaked = [name for name, value in forwarded.headers.items() if caller.token in value]
    assert not leaked, f"the caller's token reached the provider in {leaked}"
