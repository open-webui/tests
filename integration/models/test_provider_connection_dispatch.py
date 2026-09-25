"""Provider connection and dispatch regressions fixed in 0.11.0, seen through the API.

* A non-streaming request still carried `stream_options`, which strict OpenAI-compatible
  providers refuse with a 400; the OpenAI router now drops it unless `stream` is on.
* A pipe resolved its base model by writing into the caller's payload, which the tool-call
  continuation re-submits, so the continuation ran as the bare pipe instead of the preset the
  user picked (`f8c0d2fdd`, #26906). The refusal the issue reported is now made earlier by the
  base-model access walk; what still shows is the pipe's own settings reaching the continuation.
* A function or tool row without an owner failed the read model's `user_id: str`, so every
  listing that validated it (the model list, the function and tool lists, startup) answered 500
  (`48f78ca58`, #26850). The test clears the owner in the database, as a legacy row has it.
* A connection's `prefix_id` was removed anywhere in the model name, not only in front
  (`ed663f16e`), and Ollama refused a model pulled after its model list was cached (#27353).
* Saving a connection left the previous model list live until a restart; both config updates
  now clear the cached lists.
* Only the model listing honoured a disabled OpenAI API; a direct chat request still went out.

Twin of unit/models/test_provider_connection_dispatch.py. The pipe's unused `models={}` default
and the disabled `get_all_models` early return (its callee refuses the same way) had no
behaviour of their own to pin.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the `stream_options` pop,
the pipe's payload copy, the nullable `user_id`, `strip_provider_model_prefix` back to
`str.replace`, Ollama's refetch, either config update's cache clear, the 503 guard): the key
reaches the provider, the continuation carries the pipe's seed, the listings answer 500, the
provider gets `llama3.tuned`, the pulled model is a 400, the new model is missing and the
disabled request is not refused.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import httpx
import pytest

from harness.backends import write_rows
from harness.chat import ask
from harness.listener import json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.plugins import installed_function
from harness.python_tools import python_tool
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CONNECTIONS_CONFIG = ("/api/v1/configs/connections", "/api/v1/configs/connections")
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
HELLO = [{"role": "user", "content": "hello"}]
# Starts with the prefix and carries it again inside, where it must survive.
PREFIXED_NAME = "llama3.acme.tuned"

PIPE_WITH_A_TOOL_CALL = """
class Pipe:
    def pipe(self, body):
        if any(message.get("role") == "tool" for message in body["messages"]):
            return f"answered after the tool with seed {body.get('seed')}"
        call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_current_timestamp", "arguments": "{}"},
        }
        return iter([{"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}])
"""

EMPTY_PIPE = """
class Pipe:
    def pipe(self, body):
        return "nothing"
"""

EMPTY_TOOLKIT = '''
class Tools:
    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''


def _completion(model: str) -> dict:
    message = {"role": "assistant", "content": "done"}
    return {
        "id": "second",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _listed_ids(client: httpx.Client) -> set[str]:
    listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return {model["id"] for model in listed.json()["data"]}


def _save_openai(client: httpx.Client, **changes) -> None:
    current = client.get(OPENAI_CONFIG[0])
    current.raise_for_status()
    saved = client.post(OPENAI_CONFIG[1], json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def _add_openai_connection(client: httpx.Client, base_url: str, **config) -> None:
    current = client.get(OPENAI_CONFIG[0]).json()
    index = str(len(current["OPENAI_API_BASE_URLS"]))
    _save_openai(
        client,
        OPENAI_API_BASE_URLS=[*current["OPENAI_API_BASE_URLS"], base_url],
        OPENAI_API_KEYS=[*current["OPENAI_API_KEYS"], "sk-second"],
        OPENAI_API_CONFIGS={**current["OPENAI_API_CONFIGS"], index: {"enable": True, **config}},
    )


@contextmanager
def _ownerless(instance, table: str, row_id: str, owner_id: str) -> Iterator[None]:
    """The row's owner cleared, the way a legacy or orphaned row is stored; restored after."""

    def set_owner(owner: str | None) -> None:
        statement = f"UPDATE {table} SET user_id = :owner WHERE id = :row_id"
        write_rows(instance, statement, [{"owner": owner, "row_id": row_id}])

    set_owner(None)
    try:
        yield
    finally:
        set_owner(owner_id)


# ---------------------------------------------------------------- stream_options


@pytest.mark.parametrize("stream", [False, None], ids=["stream-false", "stream-absent"])
def test_a_non_streaming_request_is_sent_without_stream_options(make_user, upstream, stream):
    body = {"model": MOCK_MODEL_ID, "messages": HELLO, "stream_options": {"include_usage": True}}
    if stream is not None:
        body["stream"] = stream
    with make_user().client() as client:
        answered = client.post("/openai/chat/completions", json=body)

    assert answered.status_code == 200, answered.text
    assert "stream_options" not in upstream.chat_requests()[-1], (
        "a non-streaming request reached the provider with stream_options, which strict "
        "providers refuse with a 400"
    )


def test_a_streaming_request_keeps_its_usage_opt_in(make_user, upstream):
    body = {
        "model": MOCK_MODEL_ID,
        "messages": HELLO,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with make_user().client() as client:
        answered = client.post("/openai/chat/completions", json=body)

    assert answered.status_code == 200, answered.text
    assert upstream.chat_requests()[-1]["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------- the connection prefix


def test_an_openai_prefix_is_stripped_only_in_front(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    listener.route("GET", "/v1/models", json_answer({"data": [{"id": PREFIXED_NAME}]}))
    listener.route("POST", "/v1/chat/completions", json_answer(_completion(PREFIXED_NAME)))
    with admin.client() as client:
        _add_openai_connection(client, f"{listener.base_url}/v1", prefix_id="acme")
        assert f"acme.{PREFIXED_NAME}" in _listed_ids(client)
        answered = client.post(
            "/openai/chat/completions",
            json={"model": f"acme.{PREFIXED_NAME}", "messages": HELLO, "stream": False},
        )

    assert answered.status_code == 200, answered.text
    sent = listener.requests_to("/v1/chat/completions")[-1].json()
    assert sent["model"] == PREFIXED_NAME, (
        f"the provider was sent {sent['model']!r}: the prefix was removed inside the name too"
    )


def test_an_ollama_prefix_is_stripped_only_in_front(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    ollama = serve_ollama(listener, PREFIXED_NAME)
    with admin.client() as client:
        connect_ollama(client, listener, prefix_id="acme")
        assert f"acme.{PREFIXED_NAME}" in _listed_ids(client)
        answered = client.post(
            "/ollama/api/chat",
            json={"model": f"acme.{PREFIXED_NAME}", "messages": HELLO, "stream": False},
        )

    assert answered.status_code == 200, answered.text
    assert ollama.chat_requests()[-1]["model"] == PREFIXED_NAME


# ---------------------------------------------------------------- a model pulled into Ollama


def test_a_model_pulled_after_the_list_was_read_is_found(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    ollama = serve_ollama(listener, "stale:latest")
    with admin.client() as client:
        connect_ollama(client, listener)
        assert "stale:latest" in _listed_ids(client)
        ollama.models.append("fresh:latest")
        answered = client.post(
            "/ollama/api/chat", json={"model": "fresh:latest", "messages": HELLO, "stream": False}
        )

    assert answered.status_code == 200, (
        f"a model pulled after the model list was cached was refused: {answered.text}"
    )
    assert ollama.chat_requests()[-1]["model"] == "fresh:latest"


def test_a_model_ollama_does_not_have_is_still_refused(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    ollama = serve_ollama(listener, "stale:latest")
    with admin.client() as client:
        connect_ollama(client, listener)
        answered = client.post(
            "/ollama/api/chat", json={"model": "ghost:latest", "messages": HELLO, "stream": False}
        )

    assert answered.status_code == 400, answered.text
    assert ollama.chat_requests() == []


# ---------------------------------------------------------------- saved connections are live


@pytest.fixture
def cached_base_models(admin, preserve):
    """The base model list kept between requests, so only a config update can refresh it."""
    preserve(OPENAI_CONFIG, OLLAMA_CONFIG, CONNECTIONS_CONFIG)
    with admin.client() as client:
        current = client.get(CONNECTIONS_CONFIG[0]).json()
        saved = client.post(
            CONNECTIONS_CONFIG[1], json={**current, "ENABLE_BASE_MODELS_CACHE": True}
        )
        assert saved.status_code == 200, saved.text
        assert MOCK_MODEL_ID in _listed_ids(client)
        yield client


def test_a_new_openai_connection_is_listed_at_once(cached_base_models, listener):
    client = cached_base_models
    listener.route("GET", "/v1/models", json_answer({"data": [{"id": "fresh-model"}]}))

    _add_openai_connection(client, f"{listener.base_url}/v1")

    assert "fresh-model" in _listed_ids(client), (
        "the saved OpenAI connection's model is missing: the old model list stayed live"
    )


def test_a_new_ollama_connection_is_listed_at_once(cached_base_models, listener):
    client = cached_base_models
    serve_ollama(listener, "fresh:latest")

    connect_ollama(client, listener)

    assert "fresh:latest" in _listed_ids(client), (
        "the saved Ollama connection's model is missing: the old model list stayed live"
    )


# ---------------------------------------------------------------- a disabled API


def test_a_disabled_openai_api_refuses_a_direct_chat_request(admin, make_user, preserve, upstream):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        _save_openai(client, ENABLE_OPENAI_API=False)

    request = {"model": MOCK_MODEL_ID, "messages": HELLO, "stream": False}
    with make_user().client() as client:
        answered = client.post("/openai/chat/completions", json=request)
        # resolves its connection through the OpenAI model list
        client.post("/api/v1/messages/count_tokens", json=request)

    assert answered.status_code == 503, answered.text
    assert upstream.requests == [], (
        "the disabled OpenAI connection was still contacted: "
        f"{[(sent.method, sent.path) for sent in upstream.requests]}"
    )


def test_a_disabled_ollama_api_refuses_a_chat_request(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    ollama = serve_ollama(listener, "llama3:latest")
    with admin.client() as client:
        connect_ollama(client, listener)
        current = client.get(OLLAMA_CONFIG[0]).json()
        client.post(OLLAMA_CONFIG[1], json={**current, "ENABLE_OLLAMA_API": False})
        answered = client.post(
            "/ollama/api/chat", json={"model": "llama3:latest", "messages": HELLO, "stream": False}
        )

    assert answered.status_code == 503, answered.text
    assert ollama.chat_requests() == []


# ---------------------------------------------------------------- rows without an owner


def test_a_pipe_without_an_owner_leaves_the_listings_working(instance, admin, make_user):
    with (
        installed_function(admin, EMPTY_PIPE) as pipe_id,
        _ownerless(instance, "function", pipe_id, admin.id),
        admin.client() as admin_client,
        make_user().client() as user_client,
    ):
        functions = admin_client.get("/api/v1/functions/")
        models = user_client.get("/api/models")

    assert functions.status_code == 200, (
        f"one ownerless function took the function list down: {functions.text}"
    )
    assert pipe_id in {function["id"] for function in functions.json()}
    assert models.status_code == 200, f"one ownerless pipe took the model list down: {models.text}"


def test_a_tool_without_an_owner_leaves_the_tool_list_working(instance, admin):
    with (
        python_tool(admin, EMPTY_TOOLKIT) as tool_id,
        _ownerless(instance, "tool", tool_id, admin.id),
        admin.client() as client,
    ):
        tools = client.get("/api/v1/tools/")

    assert tools.status_code == 200, f"one ownerless tool took the tool list down: {tools.text}"
    assert tool_id in {tool["id"] for tool in tools.json()}


def test_a_function_with_an_owner_still_reports_it(admin):
    with installed_function(admin, EMPTY_PIPE) as pipe_id:
        with admin.client() as client:
            functions = client.get("/api/v1/functions/").json()

    assert {function["id"]: function["user_id"] for function in functions}[pipe_id] == admin.id


# ---------------------------------------------------------------- a shared preset over a pipe


@contextmanager
def _workspace_model(
    admin, model_id: str, base_model_id: str | None, params: dict
) -> Iterator[str]:
    """A model row every account may read, the way the admin's model editor saves one."""
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": base_model_id,
        "meta": {},
        "params": params,
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        try:
            client.get("/api/models").raise_for_status()
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_the_continuation_after_a_tool_call_runs_as_the_shared_preset(admin, make_user):
    preset_id = f"preset-{uuid.uuid4().hex[:8]}"
    with (
        installed_function(admin, PIPE_WITH_A_TOOL_CALL) as pipe_id,
        # the pipe's own settings, which only a continuation run as the bare pipe picks up
        _workspace_model(admin, pipe_id, None, {"seed": 7}),
        _workspace_model(admin, preset_id, pipe_id, {"function_calling": "native"}),
        make_user().client() as client,
    ):
        _, answer = ask(client, "what time is it?", model=preset_id)

    assert answer["content"] == "answered after the tool with seed None", (
        "the continuation after the tool call was resolved as the bare pipe, whose own "
        f"settings then applied, instead of the preset the user chose: {answer['content']!r}"
    )
