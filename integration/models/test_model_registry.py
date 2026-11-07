"""Regressions in model resolution and provider routing, seen through the HTTP API.

* `5cecb7dbfa`: switching off a model removed it with `list.remove`; an Ollama base name and its
  tagged id resolve to the same entry, so switching both off removed it twice and the
  `ValueError` took the whole model list down.
* `3d630491c` (#28036): `sync_models` passed `user_id` and `updated_at` twice when it updated a
  model that already existed; the broad handler swallowed the `TypeError`, so a second sync
  answered 200 with an empty list and changed nothing.
* `686d8dc54` (#28575): `/openai/responses` serialized the body before the connection was
  resolved, so a connection with a Prefix ID was sent the prefixed model id.
* `9cf1a0796` (#27675, #27595): the Anthropic Messages passthrough read a timeout constant that
  no longer existed, so every passthrough request failed with a 502 before it was sent.
* `20fe43d9da`: the missing-base-model fallback ran after the access check, and that check
  refuses a non-admin whose base model has no workspace row, so only admins got the fallback.
* `eadce55e34` (#28952, #28923): a workspace model whose `base_model_id` is its own id was stored
  as such on create, update and import, and then dropped while models were combined.
* `16f118d77a`: the check that a named Ollama backend serves the model returned early under
  `BYPASS_MODEL_ACCESS_CONTROL`, so a user could aim a request at any backend. Two Ollama
  stand-ins, the model on the first only, on an instance booted with the bypass.
* `051c357f9` (#33759): model meta and params of any depth were stored; settings nested more
  than 32 levels deep are now refused with 400 on create, update and import, and nothing is
  stored. Settings exactly 32 levels deep are still saved.
* `9cf1a0796` (#27595): native Anthropic requests to api.anthropic.com were signed with a bearer
  token, which Anthropic rejects; they now carry `x-api-key` and `anthropic-version`. The check
  matches the host name anywhere in the connection URL, so a listener path of that name stands in.

Twin of unit/models/test_model_registry.py, which keeps the sweep for model pool merges. Its
per-connection listing tests (`16f118d77a`) are already pinned by
integration/security/test_connection_listing_roles.py.

Discriminates: passes on dev `bbfa876af`; each narrow test fails with its fix reverted (the alias
removal, the sync update, the prefix strip, the passthrough timeout, the fallback ordering and the
three self-reference guards, one mutation each). On dev ef67cc3fa, restoring the bypass early
return fails all four backend cases and dropping the api.anthropic.com header rewrite fails the
Anthropic key test. The fallback on the web client path fails on dev a5bc78300 and passes since
PR #31353 (#31345). A re-sync on the default SQLite setup (#31346) passes since PR #31349
(`ac00d40e3`). The nesting tests pass on dev f659c0d12; with 051c357f9 reverted the refused
cases answer 200 and store the model.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.listener import json_answer, listening
from harness.ollama_provider import serve_ollama
from harness.second_provider import OPENAI_CONFIG, attach

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SECOND_MODEL = "second-model"


def _model_form(model_id: str, base_model_id: str | None, **extra) -> dict:
    return {
        "id": model_id,
        "base_model_id": base_model_id,
        "name": f"Preset {model_id}",
        "meta": {},
        "params": {},
        **extra,
    }


def _stored_model(client, model_id: str) -> dict:
    found = client.get("/api/v1/models/model", params={"id": model_id})
    assert found.status_code == 200, found.text
    return found.json()


@pytest.fixture
def delete_after(admin):
    """Workspace model ids to delete once the test is over."""
    model_ids: list[str] = []
    yield model_ids
    with admin.client() as client:
        for model_id in model_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


# --- 5cecb7dbfa: two aliases of one Ollama model can both be switched off ----------------

OLLAMA_CONFIG = ("/ollama/config", "/ollama/config/update")


@pytest.fixture
def ollama_llama(admin, preserve, listener):
    """Ollama switched on with one connection serving `llama3:latest`."""
    tags = {"models": [{"name": "llama3:latest", "model": "llama3:latest"}]}
    listener.route("GET", "/api/tags", json_answer(tags))
    listener.route("GET", "/api/version", json_answer({"version": "0.9.1"}))
    preserve(OLLAMA_CONFIG)
    with admin.client() as client:
        current = client.get(OLLAMA_CONFIG[0]).json()
        enabled = {
            **current,
            "ENABLE_OLLAMA_API": True,
            "OLLAMA_BASE_URLS": [listener.base_url],
            "OLLAMA_API_CONFIGS": {},
        }
        client.post(OLLAMA_CONFIG[1], json=enabled).raise_for_status()


def test_switching_off_both_aliases_of_an_ollama_model_keeps_the_model_list(
    admin, ollama_llama, delete_after
):
    with admin.client() as client:
        for alias in ("llama3", "llama3:latest"):
            delete_after.append(alias)
            created = client.post("/api/v1/models/create", json=_model_form(alias, None))
            assert created.status_code == 200, created.text
            switched = client.post("/api/v1/models/model/toggle", params={"id": alias})
            assert switched.json()["is_active"] is False, switched.text

        listed = client.get("/api/models")

    assert listed.status_code == 200, (
        "both aliases resolve to the same Ollama model, so switching both off removed it twice "
        f"and the second removal took the whole model list down: {listed.text[:300]}"
    )
    model_ids = [model["id"] for model in listed.json()["data"]]
    assert "llama3:latest" not in model_ids
    assert reply.MOCK_MODEL_ID in model_ids


# --- eadce55e34: a model is never stored as based on itself --------------------------------


def test_creating_a_model_based_on_itself_stores_no_base(admin, delete_after):
    model_id = f"self-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=_model_form(model_id, model_id))
        assert created.status_code == 200, created.text

        assert _stored_model(client, model_id)["base_model_id"] is None, (
            "a model based on itself was stored, and combining models then dropped it with all "
            "of its settings (#28952)"
        )


def test_updating_a_model_to_be_based_on_itself_stores_no_base(admin, delete_after):
    model_id = f"self-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    with admin.client() as client:
        form = _model_form(model_id, reply.MOCK_MODEL_ID)
        assert client.post("/api/v1/models/create", json=form).status_code == 200
        updated = client.post("/api/v1/models/model/update", json=_model_form(model_id, model_id))
        assert updated.status_code == 200, updated.text

        assert _stored_model(client, model_id)["base_model_id"] is None, (
            "the edit endpoint stored a model based on itself (#28952)"
        )


def test_importing_a_model_based_on_itself_stores_no_base(admin, delete_after):
    model_id = f"self-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    with admin.client() as client:
        imported = client.post(
            "/api/v1/models/import", json={"models": [_model_form(model_id, model_id)]}
        )
        assert imported.status_code == 200, imported.text

        assert _stored_model(client, model_id)["base_model_id"] is None, (
            "an imported model based on itself was stored as such (#28952)"
        )


def test_an_ordinary_base_model_is_kept(admin, delete_after):
    model_id = f"preset-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    with admin.client() as client:
        form = _model_form(model_id, reply.MOCK_MODEL_ID)
        assert client.post("/api/v1/models/create", json=form).status_code == 200

        assert _stored_model(client, model_id)["base_model_id"] == reply.MOCK_MODEL_ID


# --- 051c357f9: model settings nested too deeply are refused --------------------------------

# meta and params themselves are the first level
MAX_DEPTH = 32


def _nested(levels: int) -> dict:
    """A setting whose value sits `levels` objects deep, alternating dicts and lists."""
    value: object = "bottom"
    for level in range(levels):
        value = {"inner": value} if level % 2 else [value]
    return {"deep": value}


def _deep_form(model_id: str, field: str, levels: int) -> dict:
    return _model_form(model_id, reply.MOCK_MODEL_ID, **{field: _nested(levels)})


def _refused_way(client, way: str, model_id: str, field: str):
    """Save the model with `field` nested too deeply, by `way`; the answer."""
    too_deep = _deep_form(model_id, field, MAX_DEPTH)
    if way == "create":
        return client.post("/api/v1/models/create", json=too_deep)
    if way == "import":
        return client.post("/api/v1/models/import", json={"models": [too_deep]})
    created = client.post("/api/v1/models/create", json=_model_form(model_id, reply.MOCK_MODEL_ID))
    assert created.status_code == 200, created.text
    return client.post("/api/v1/models/model/update", json=too_deep)


@pytest.mark.parametrize("field", ["meta", "params"])
@pytest.mark.parametrize("way", ["create", "update", "import"])
def test_model_settings_nested_too_deeply_are_refused(admin, delete_after, way, field):
    model_id = f"deep-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    with admin.client() as client:
        answer = _refused_way(client, way, model_id, field)
        stored = client.get("/api/v1/models/model", params={"id": model_id})

    assert answer.status_code == 400, (
        f"a model whose {field} is nested {MAX_DEPTH + 1} levels deep was saved by {way}: "
        f"HTTP {answer.status_code}"
    )
    if way == "update":
        assert stored.status_code == 200, stored.text
        assert "deep" not in stored.json()[field], f"the deep {field} was stored anyway"
    else:
        assert stored.status_code != 200, f"the model was stored anyway: {stored.text[:200]}"


@pytest.mark.parametrize("way", ["create", "update"])
def test_model_settings_nested_up_to_the_limit_are_saved(admin, delete_after, way):
    model_id = f"deep-{uuid.uuid4().hex[:8]}"
    delete_after.append(model_id)
    at_limit = {"meta": _nested(MAX_DEPTH - 1), "params": _nested(MAX_DEPTH - 1)}
    form = _model_form(model_id, reply.MOCK_MODEL_ID, **at_limit)
    with admin.client() as client:
        if way == "create":
            saved = client.post("/api/v1/models/create", json=form)
        else:
            plain = _model_form(model_id, reply.MOCK_MODEL_ID)
            assert client.post("/api/v1/models/create", json=plain).status_code == 200
            saved = client.post("/api/v1/models/model/update", json=form)
        assert saved.status_code == 200, saved.text
        stored = _stored_model(client, model_id)

    assert stored["meta"]["deep"] == at_limit["meta"]["deep"]
    assert stored["params"]["deep"] == at_limit["params"]["deep"]


# --- 686d8dc54 and 9cf1a0796: what a second connection is sent ----------------------------


@pytest.fixture
def second_connection(preserve, admin, listener):
    """`second_connection(**config)` adds the listener as a connection serving `SECOND_MODEL`."""
    preserve(OPENAI_CONFIG)

    def add(**config):
        with admin.client() as client:
            attach(client, listener, SECOND_MODEL, **config)
        return listener

    return add


def test_the_responses_api_sends_the_bare_model_id_to_a_prefixed_connection(
    admin, second_connection
):
    provider = second_connection(prefix_id="pfx")
    provider.route("POST", "/v1/responses", json_answer({"id": "resp_1", "output": []}))
    with admin.client() as client:
        answered = client.post(
            "/openai/responses", json={"model": f"pfx.{SECOND_MODEL}", "input": "hi"}
        )

    assert answered.status_code == 200, answered.text
    assert provider.requests_to("/v1/responses")[0].json()["model"] == SECOND_MODEL, (
        "the connection's Prefix ID was sent upstream with the model id, so the provider "
        "answered model not found (#28575)"
    )


def test_the_responses_api_sends_an_unprefixed_model_id_as_is(admin, second_connection):
    provider = second_connection()
    provider.route("POST", "/v1/responses", json_answer({"id": "resp_1", "output": []}))
    with admin.client() as client:
        client.post("/openai/responses", json={"model": SECOND_MODEL, "input": "hi"})

    assert provider.requests_to("/v1/responses")[0].json()["model"] == SECOND_MODEL


def test_the_anthropic_messages_passthrough_reaches_the_provider(admin, second_connection):
    # a LiteLLM connection takes the native passthrough, like api.anthropic.com
    provider = second_connection(provider="litellm")
    message = {"id": "msg_1", "type": "message", "role": "assistant", "content": []}
    provider.route("POST", "/v1/messages", json_answer(message))
    with admin.client() as client:
        answered = client.post(
            "/api/v1/messages",
            json={
                "model": SECOND_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert answered.status_code == 200, (
        f"the Anthropic passthrough failed before it sent anything: HTTP {answered.status_code} "
        f"{answered.text} (#27595)"
    )
    sent = provider.requests_to("/v1/messages")
    assert len(sent) == 1
    assert sent[0].headers.get("Authorization") == "Bearer sk-second"
    assert "x-api-key" not in {name.lower() for name in sent[0].headers}


# --- Instances of their own: the fallback is env-only, and sync replaces every model -----

FALLBACK_ENV = {"ENABLE_CUSTOM_MODEL_FALLBACK": "true"}
SHARED_SESSION_ENV = {"DATABASE_ENABLE_SESSION_SHARING": "true"}
MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")


@pytest.fixture(scope="module")
def fallback_instance(instance_with):
    return instance_with(FALLBACK_ENV)


@pytest.fixture
def orphaned_preset(fallback_instance, preserve):
    """A preset every account can read whose base model is gone, with `mock-model` as default."""
    preserve(MODELS_CONFIG, on=fallback_instance)
    model_id = f"orphan-{uuid.uuid4().hex[:8]}"
    grant = {"principal_type": "user", "principal_id": "*", "permission": "read"}
    with admin_of(fallback_instance).client() as client:
        defaults = client.get(MODELS_CONFIG[0]).json()
        client.post(
            MODELS_CONFIG[1], json={**defaults, "DEFAULT_MODELS": reply.MOCK_MODEL_ID}
        ).raise_for_status()
        form = _model_form(model_id, "gone-model", access_grants=[grant])
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models").raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _completion(actor, model_id: str):
    """A plain API completion: no chat and no socket session, so no per-model fan-out."""
    with actor.client() as client:
        return client.post(
            "/api/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hello"}]},
        )


@pytest.mark.slow
def test_a_regular_user_gets_the_fallback_for_a_preset_whose_base_is_gone(
    fallback_instance, orphaned_preset
):
    fallback_instance.upstream.queue(reply.text("answered by the fallback"))
    answered = _completion(create_user(fallback_instance), orphaned_preset)

    assert answered.status_code == 200, (
        "a regular user was refused a preset whose base model is gone before the fallback "
        f"could apply, so only admins ever reached it: {answered.text}"
    )
    assert answered.json()["choices"][0]["message"]["content"] == "answered by the fallback"
    assert fallback_instance.upstream.chat_requests()[-1]["model"] == reply.MOCK_MODEL_ID


@pytest.mark.slow
def test_an_admin_still_gets_the_fallback(fallback_instance, orphaned_preset):
    fallback_instance.upstream.queue(reply.text("answered by the fallback"))
    answered = _completion(admin_of(fallback_instance), orphaned_preset)

    assert answered.status_code == 200, answered.text
    assert fallback_instance.upstream.chat_requests()[-1]["model"] == reply.MOCK_MODEL_ID


@pytest.mark.slow
def test_the_web_client_path_also_gets_the_fallback(fallback_instance, orphaned_preset):
    """#31345: the per-model fan-out rebuilt the request with the requested model id."""
    fallback_instance.upstream.queue(reply.text("answered by the fallback"))
    with create_user(fallback_instance).client() as client:
        _, message = ask(client, "hello", model=orphaned_preset)

    assert message["content"] == "answered by the fallback", message.get("error")


def _sync_twice(target) -> tuple[list[dict], dict]:
    """Sync a new model, then sync it again renamed; returns the second answer and the row."""
    admin = admin_of(target)
    model_id = f"synced-{uuid.uuid4().hex[:8]}"
    now = int(time.time())
    synced = {**_model_form(model_id, None), "user_id": admin.id, "is_active": True}
    synced.update(updated_at=now, created_at=now)
    with admin.client() as client:
        # sync replaces every model, so the existing ones go back in with it
        existing = client.get("/api/v1/models/all").json()
        first = client.post("/api/v1/models/sync", json={"models": [*existing, synced]})
        assert model_id in [model["id"] for model in first.json()], first.text

        renamed = {**synced, "name": "Renamed on the second sync"}
        second = client.post("/api/v1/models/sync", json={"models": [*existing, renamed]})
        assert second.status_code == 200, second.text
        stored = _stored_model(client, model_id)
        client.post("/api/v1/models/model/delete", json={"id": model_id})
    return second.json(), stored


@pytest.mark.slow
def test_syncing_a_model_that_already_exists_updates_it(instance_with):
    # a shared session keeps the SQLite write lock below from masking this bug
    answered, stored = _sync_twice(instance_with(SHARED_SESSION_ENV))

    assert [model for model in answered if model["id"] == stored["id"]], (
        "the second sync answered with an empty list: updating an existing model raised and "
        "the error was swallowed (#28036)"
    )
    assert stored["name"] == "Renamed on the second sync"


@pytest.mark.slow
def test_syncing_a_model_that_already_exists_updates_it_by_default(fallback_instance):
    """#31346: the update held SQLite's write lock while the grant write opened a second session."""
    answered, stored = _sync_twice(fallback_instance)

    assert stored["name"] == "Renamed on the second sync", answered


# --- 16f118d77a: a named Ollama backend is checked when model access control is bypassed --

BYPASS_ENV = {"BYPASS_MODEL_ACCESS_CONTROL": "true"}
SERVED_MODEL = "llama3:latest"
BACKEND_ROUTES = {
    "chat": ("/ollama/api/chat", {"messages": [{"role": "user", "content": "hi"}]}),
    "generate": ("/ollama/api/generate", {"prompt": "hi"}),
    "embed": ("/ollama/api/embed", {"input": "hi"}),
    "embeddings": ("/ollama/api/embeddings", {"prompt": "hi"}),
}


@pytest.fixture(scope="module")
def two_ollama_backends(instance_with):
    """An instance bypassing model access control, `SERVED_MODEL` on its first backend only."""
    bypassing = instance_with(BYPASS_ENV)
    with listening() as serving_listener, listening() as other_listener:
        serving = serve_ollama(serving_listener, SERVED_MODEL)
        other = serve_ollama(other_listener, "other:latest")
        with admin_of(bypassing).client() as client:
            current = client.get(OLLAMA_CONFIG[0]).json()
            both = {
                **current,
                "ENABLE_OLLAMA_API": True,
                "OLLAMA_BASE_URLS": [serving_listener.base_url, other_listener.base_url],
                "OLLAMA_API_CONFIGS": {},
            }
            client.post(OLLAMA_CONFIG[1], json=both).raise_for_status()
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield bypassing, serving, other


def _ollama_call(actor, route: str, backend: int | None) -> int:
    path, body = BACKEND_ROUTES[route]
    target = path if backend is None else f"{path}/{backend}"
    with actor.client() as client:
        answered = client.post(target, json={"model": SERVED_MODEL, "stream": False, **body})
    return answered.status_code


def _calls_to(server) -> int:
    return len([request for request in server.listener.received if request.method == "POST"])


@pytest.mark.slow
@pytest.mark.parametrize("route", BACKEND_ROUTES)
def test_a_user_cannot_aim_a_model_at_a_backend_that_does_not_serve_it(two_ollama_backends, route):
    bypassing, _serving, other = two_ollama_backends
    calls_before = _calls_to(other)

    status = _ollama_call(create_user(bypassing), route, backend=1)

    assert status == 403, (
        "BYPASS_MODEL_ACCESS_CONTROL waives per-model permissions, not the check that the named "
        f"backend serves the model; a user reached another backend (HTTP {status})"
    )
    assert _calls_to(other) == calls_before, "the request was forwarded to the other backend"


@pytest.mark.slow
def test_the_backend_that_serves_the_model_can_be_named(two_ollama_backends):
    bypassing, serving, _other = two_ollama_backends
    calls_before = _calls_to(serving)

    assert _ollama_call(create_user(bypassing), "generate", backend=0) == 200
    assert _calls_to(serving) == calls_before + 1


@pytest.mark.slow
def test_without_a_named_backend_the_serving_one_is_used(two_ollama_backends):
    bypassing, serving, other = two_ollama_backends
    calls_before = (_calls_to(serving), _calls_to(other))

    assert _ollama_call(create_user(bypassing), "generate", backend=None) == 200
    assert (_calls_to(serving), _calls_to(other)) == (calls_before[0] + 1, calls_before[1])


@pytest.mark.slow
def test_an_admin_may_still_name_any_backend(two_ollama_backends):
    bypassing, _serving, other = two_ollama_backends
    calls_before = _calls_to(other)

    _ollama_call(admin_of(bypassing), "generate", backend=1)

    assert _calls_to(other) == calls_before + 1


# --- 9cf1a0796: api.anthropic.com gets x-api-key and anthropic-version, not a bearer token -

# the check matches the host name anywhere in the URL, so a local path of that name stands in
ANTHROPIC_PATH = "/api.anthropic.com/v1"
CLAUDE_MODEL = "claude-sonnet-4"


@pytest.fixture
def anthropic_connection(preserve, admin, listener):
    """The listener added as an OpenAI connection whose URL names api.anthropic.com."""
    preserve(OPENAI_CONFIG)
    models = {"data": [{"id": CLAUDE_MODEL, "type": "model"}], "has_more": False}
    listener.route("GET", f"{ANTHROPIC_PATH}/models", json_answer(models))
    message = {"id": "msg_1", "type": "message", "role": "assistant", "content": []}
    listener.route("POST", f"{ANTHROPIC_PATH}/messages", json_answer(message))
    with admin.client() as client:
        connections = client.get(OPENAI_CONFIG[0]).json()
        index = str(len(connections["OPENAI_API_BASE_URLS"]))
        updated = {
            **connections,
            "OPENAI_API_BASE_URLS": [
                *connections["OPENAI_API_BASE_URLS"],
                f"{listener.base_url}{ANTHROPIC_PATH}",
            ],
            "OPENAI_API_KEYS": [*connections["OPENAI_API_KEYS"], "sk-ant"],
            "OPENAI_API_CONFIGS": {**connections["OPENAI_API_CONFIGS"], index: {"enable": True}},
        }
        client.post(OPENAI_CONFIG[1], json=updated).raise_for_status()
        listed = client.get("/api/models", params={"refresh": "true"}).json()["data"]
    assert CLAUDE_MODEL in [model["id"] for model in listed], listed
    return listener


def test_a_native_anthropic_request_carries_the_key_anthropic_expects(admin, anthropic_connection):
    with admin.client() as client:
        answered = client.post(
            "/api/v1/messages",
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert answered.status_code == 200, answered.text
    [sent] = anthropic_connection.requests_to(f"{ANTHROPIC_PATH}/messages")
    headers = {name.lower(): value for name, value in sent.headers.items()}
    assert headers.get("x-api-key") == "sk-ant", (
        f"Anthropic's native endpoint was not sent its key as x-api-key (#27595): {headers}"
    )
    assert "authorization" not in headers, "Anthropic's native endpoint rejects a bearer token"
    assert headers.get("anthropic-version")
