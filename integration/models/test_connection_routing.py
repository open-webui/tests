"""Journey: which connection answers a chat, and what a user is told when none can.

Seen through the chat endpoint the web client uses and the stored reply. Two connections serving
one model name answer under their own prefix each, and the provider gets the bare id; without
prefixes the first connection answers the shared name and the second takes over once the first
is switched off. A switched-off OpenAI or Ollama connection's model is refused with "Model not
found", and a model listed by the allowlist of a provider that cannot be reached is stored with
the connection error while the model list still answers. A connection whose URL names
`api.anthropic.com` is listed from every page of Anthropic's model list, by display name, with
the `x-api-key` header (docs: starting-with-anthropic). Verifying an Ollama connection that
refuses the key answers with the server's reason.

Twin of e2e/admin/test_connection_models_for_users.py, test_connection_provider_types.py and
test_connection_verify_and_refresh.py.

Discriminates: passes on dev ebc6add67; in a backend copy, the merge letting the last connection
win a shared name fails the shared-name test, the prefix kept on the id sent upstream fails the
prefix test, the OpenAI listing ignoring `enable` fails both OpenAI switch-off tests and the
Ollama listing ignoring it the Ollama one, and the Anthropic listing reading only the first page
fails the Anthropic test.
"""

from __future__ import annotations

import uuid

import pytest

from harness.chat import ask
from harness.listener import Listener, ReceivedRequest, json_answer, listening
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.second_provider import OPENAI_CONFIG, attach, sse

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

# refused at once, so a connection to it fails without waiting
UNREACHABLE = "http://127.0.0.1:9"
ANTHROPIC_PATH = "/api.anthropic.com/v1"


def add_openai(client, url: str, key: str = "", **config) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    index = str(len(current["OPENAI_API_BASE_URLS"]))
    changed = {
        **current,
        "OPENAI_API_BASE_URLS": [*current["OPENAI_API_BASE_URLS"], url],
        "OPENAI_API_KEYS": [*current["OPENAI_API_KEYS"], key],
        "OPENAI_API_CONFIGS": {**current["OPENAI_API_CONFIGS"], index: {"enable": True, **config}},
    }
    client.post(OPENAI_CONFIG[1], json=changed).raise_for_status()


def switch_off(client, url: str) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    index = str(current["OPENAI_API_BASE_URLS"].index(url))
    current["OPENAI_API_CONFIGS"][index] = {**current["OPENAI_API_CONFIGS"][index], "enable": False}
    client.post(OPENAI_CONFIG[1], json=current).raise_for_status()


def listed_models(client) -> dict[str, dict]:
    listed = client.get("/api/models", params={"refresh": True})
    assert listed.status_code == 200, listed.text
    return {model["id"]: model for model in listed.json()["data"]}


def models_sent(listener: Listener) -> list[str]:
    return [request.json()["model"] for request in listener.requests_to("/v1/chat/completions")]


def answer_with(listener: Listener, text: str) -> None:
    listener.route("POST", "/v1/chat/completions", sse({"content": text}))


def test_two_prefixed_connections_serving_one_name_each_answer_with_the_bare_id(
    admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    north, south = f"north{uuid.uuid4().hex[:4]}", f"south{uuid.uuid4().hex[:4]}"
    with listening() as other, admin.client() as client:
        answer_with(listener, "from the north")
        answer_with(other, "from the south")
        attach(client, listener, "twin", prefix_id=north)
        attach(client, other, "twin", prefix_id=south)

        _, northern = ask(client, "Who are you?", model=f"{north}.twin")
        _, southern = ask(client, "Who are you?", model=f"{south}.twin")

        assert northern["content"] == "from the north"
        assert southern["content"] == "from the south"
        assert models_sent(listener) == ["twin"]
        assert models_sent(other) == ["twin"]


def test_the_first_connection_answers_a_shared_name_until_it_is_switched_off(
    admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    shared_name = f"twin-{uuid.uuid4().hex[:6]}"
    with listening() as second, admin.client() as client:
        answer_with(listener, "the first answered")
        answer_with(second, "the second answered")
        attach(client, listener, shared_name)
        attach(client, second, shared_name)
        assert list(listed_models(client)).count(shared_name) == 1

        _, before = ask(client, "Who answers?", model=shared_name)
        switch_off(client, f"{listener.base_url}/v1")
        _, after = ask(client, "And now?", model=shared_name)

        assert before["content"] == "the first answered"
        assert after["content"] == "the second answered"
        assert len(models_sent(listener)) == 1


def test_a_switched_off_connections_model_is_refused_as_not_found(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    prefix = f"off{uuid.uuid4().hex[:6]}"
    answer_with(listener, "still connected")
    with admin.client() as client:
        attach(client, listener, "alpha", prefix_id=prefix)
        switch_off(client, f"{listener.base_url}/v1")

        assert f"{prefix}.alpha" not in listed_models(client)
        refused = client.post(
            "/api/chat/completions",
            json={"model": f"{prefix}.alpha", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == "Model not found"
    assert listener.requests_to("/v1/chat/completions") == []


def test_a_switched_off_ollama_connection_lists_nothing_and_refuses_its_model(
    admin, preserve, listener
):
    preserve(OLLAMA_CONFIG)
    prefix = f"farm{uuid.uuid4().hex[:6]}"
    server = serve_ollama(listener, "llama3:latest")
    with admin.client() as client:
        connect_ollama(client, listener, prefix_id=prefix, enable=False)
        models = listed_models(client)
        refused = client.post(
            "/api/chat/completions",
            json={
                "model": f"{prefix}.llama3:latest",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert f"{prefix}.llama3:latest" not in models
    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == "Model not found"
    assert server.sent("/api/chat") == []


def test_an_unreachable_allowlisted_model_stores_the_connection_error(admin, preserve):
    preserve(OPENAI_CONFIG)
    prefix = f"gone{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        add_openai(client, f"{UNREACHABLE}/listed/v1")
        add_openai(client, f"{UNREACHABLE}/allowed/v1", prefix_id=prefix, model_ids=["ghost"])
        models = listed_models(client)
        _, failed = ask(client, "Is anyone there?", model=f"{prefix}.ghost")

    assert f"{prefix}.ghost" in models
    assert "mock-model" in models
    assert failed["error"]["content"] == "Open WebUI: Server Connection Error"


def anthropic_models(request: ReceivedRequest):
    if "after_id=claude-harbor-1" in request.path:
        second = [{"id": "claude-lighthouse-1", "display_name": "Claude Lighthouse"}]
        return json_answer({"data": second, "has_more": False, "last_id": "claude-lighthouse-1"})
    first = [{"id": "claude-harbor-1", "display_name": "Claude Harbor"}]
    return json_answer({"data": first, "has_more": True, "last_id": "claude-harbor-1"})


def test_an_anthropic_connection_lists_every_page_by_display_name(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    listener.route("GET", f"{ANTHROPIC_PATH}/models", anthropic_models)
    with admin.client() as client:
        add_openai(client, f"{listener.base_url}{ANTHROPIC_PATH}", key="sk-ant-paged")
        models = listed_models(client)

    assert models["claude-harbor-1"]["name"] == "Claude Harbor"
    assert models["claude-lighthouse-1"]["name"] == "Claude Lighthouse"
    listings = listener.requests_to(f"{ANTHROPIC_PATH}/models")
    assert {request.headers.get("x-api-key") for request in listings} == {"sk-ant-paged"}


def test_a_refused_ollama_verification_answers_with_the_servers_reason(admin, listener):
    listener.route("GET", "/api/version", json_answer({"error": "unauthorized"}, status=401))
    with admin.client() as client:
        refused = client.post("/ollama/verify", json={"url": listener.base_url, "key": ""})

    assert refused.status_code == 500
    assert "unauthorized" in refused.json()["detail"]
