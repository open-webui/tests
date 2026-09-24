"""Regression: a keyless tool server got `Authorization: Bearer ` and a stale cache hid terminals.

Two fixes in `utils/tools.py`, open-webui 0.11.2:

* `26f37426b`: `build_tool_server_headers` formatted `Bearer {key}` unconditionally, so a bearer
  connection saved without a key sent a bare `Authorization: Bearer ` that some servers refuse
  as a malformed credential. Every bearer branch now goes through `bearer_auth_header`, which
  omits the header when the stripped token is empty.
* `81b9afb73`: `get_terminal_servers` took an empty list cached in Redis by another worker as a
  hit, so a configured terminal was "unavailable" to every chat until the cache expired. An
  empty cached list is now only trusted when no enabled connection has a url.

The terminal tests run on an instance of their own backed by a Redis stand-in the test can write
to, which plays the other worker.

Twin of unit/tools/test_tool_server_auth_and_terminal_cache.py.

Discriminates: passes on dev bbfa876af; fails with the bearer branch formatting `Bearer {key}`
again (both keyless calls carry the header) and with an empty cached terminal list trusted again
(the chat on the terminal is refused as unavailable); the nearby tests pass on both.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from integration.stateful_redis import StatefulRedis

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
TERMINAL_CACHE_KEY = "open-webui:terminal_servers"


def _openapi(operation_id: str, path: str) -> dict:
    operation = {"operationId": operation_id, "responses": {"200": {"description": "ok"}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": operation_id, "version": "1"},
        "paths": {path: {"post": operation}},
    }


# --- 26f37426b: a keyless bearer connection sends no Authorization ------------------------


@pytest.fixture
def call_tool_server(make_user, upstream, preserve, listener):
    """`call_tool_server(**auth)` saves a tool server with that auth, has the model call it and
    returns the headers the server received."""
    preserve(TOOL_SERVERS_CONFIG)
    listener.route("GET", "/lookup-srv/openapi.json", json_answer(_openapi("lookup", "/lookup")))
    listener.route("POST", "/lookup-srv/lookup", json_answer({"found": True}))
    caller = make_user(role="admin")

    def call(**auth) -> dict[str, str]:
        connection = {
            "url": f"{listener.base_url}/lookup-srv",
            "path": "openapi.json",
            "config": {"enable": True},
            "info": {"id": "lookup-srv", "name": "Lookup"},
            "auth_type": "bearer",
            **auth,
        }
        with caller.client() as client:
            saved = client.post(
                TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
            )
            assert saved.status_code == 200, saved.text
            upstream.queue(reply.tool_call("lookup", {}), reply.text("looked up"))
            ask(client, "look it up", tool_ids=["server:lookup-srv"])
        (received,) = listener.requests_to("/lookup-srv/lookup")
        return {name.lower(): value for name, value in received.headers.items()}

    return call


@pytest.mark.parametrize("key", ["", "   "], ids=["empty", "blank"])
def test_a_keyless_bearer_connection_sends_no_authorization(call_tool_server, key):
    headers = call_tool_server(key=key)

    assert "authorization" not in headers, (
        f"a tool server saved without a key was sent {headers['authorization']!r}, a malformed "
        "credential some servers refuse"
    )


def test_a_keyed_bearer_connection_still_sends_its_key(call_tool_server):
    assert call_tool_server(key="server-key")["authorization"] == "Bearer server-key"


def test_custom_headers_survive_a_keyless_connection(call_tool_server):
    headers = call_tool_server(key="", headers={"X-Tenant": "acme"})

    assert headers["x-tenant"] == "acme"
    assert "authorization" not in headers


# --- 81b9afb73: an empty cached terminal list does not hide a configured terminal ---------


class SharedRedis(StatefulRedis):
    """The Redis another worker shares with the instance."""

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._values[key] = (value, None)

    def read(self, key: str) -> str | None:
        with self._lock:
            return self._live(key)


@pytest.fixture(scope="module")
def shared_redis():
    store = SharedRedis()
    yield store
    store.close()


@pytest.fixture
def cached_terminal(shared_redis, instance_with, preserve):
    """A terminal saved on an instance backed by `shared_redis`; yields (instance, terminal, id)."""
    launched = instance_with({"REDIS_URL": shared_redis.url})
    preserve(TERMINAL_SERVERS_CONFIG, on=launched)
    with serving_terminal() as terminal:
        terminal.route("GET", "/openapi.json", json_answer(_openapi("run_command", "/execute")))
        connection = terminal.connection()
        with launched.client() as client:
            configure_terminals(client, connection)
        terminal.clear()
        yield launched, terminal, connection["id"]


def _chat_on_terminal(launched, terminal_id: str) -> list[str]:
    """Chat with the terminal selected; returns the names of the tools the model was offered."""
    launched.upstream.queue(reply.text("ready"))
    with launched.client() as client:
        _, message = ask(client, "list my files", terminal_id=terminal_id)
    assert message["content"] == "ready", f"the chat on the terminal failed: {message}"
    offered = launched.upstream.chat_requests()[0].get("tools", [])
    return [tool["function"]["name"] for tool in offered]


@pytest.mark.slow
def test_an_empty_list_cached_by_another_worker_does_not_hide_the_terminal(
    shared_redis, cached_terminal
):
    launched, terminal, terminal_id = cached_terminal
    shared_redis.put(TERMINAL_CACHE_KEY, "[]")

    offered = _chat_on_terminal(launched, terminal_id)

    assert "run_command" in offered, (
        "an empty terminal list cached by another worker made the configured terminal "
        f"unavailable; the model was offered {offered}"
    )
    assert terminal.requests_to("/openapi.json"), "the terminal list was never rebuilt"
    rebuilt = json.loads(shared_redis.read(TERMINAL_CACHE_KEY))
    assert [server["id"] for server in rebuilt] == [terminal_id], "the rebuild was not cached"


@pytest.mark.slow
def test_a_populated_cache_is_used_without_refetching(shared_redis, cached_terminal):
    launched, terminal, terminal_id = cached_terminal

    offered = _chat_on_terminal(launched, terminal_id)

    assert "run_command" in offered
    assert terminal.requests_to("/openapi.json") == []
