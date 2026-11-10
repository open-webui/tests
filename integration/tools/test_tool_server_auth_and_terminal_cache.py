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

A third fix, 5232f9bb2 (PR open-webui/open-webui#33699, issues #30001 and #33579): a terminal
connection with Auth set to Session sent no key on Verify Connection, the orchestrator's
management calls and the tool list fetch, so an orchestrator refused them all and every chat on
it failed with "Terminal server is unavailable". Those calls carry the connection's key, while
the commands a user runs on a plain terminal still go out under the user's own session. Since
dev 6f2fc7bc7 an orchestrator is managed through its templates (the policy, lifecycle and refresh
routes are gone), Verify Connection asks for the template list, and the dialog saves an
orchestrator with Bearer auth; a connection stored with Session before that keeps working, which
the template tests here pin. On dev c15f28649 they fail on a backend copy where Verify Connection
sends the key for Bearer alone (verify) and where the template calls send the key for Bearer
alone (the menu, status, create and command cases).

Twin of unit/tools/test_tool_server_auth_and_terminal_cache.py, which keeps the audit of every
bearer branch and the empty cached list with no terminal able to serve. The system OAuth and
session branches are also called here, with the credentials a request can give them.

Discriminates: passes on dev bbfa876af; fails with the bearer branch formatting `Bearer {key}`
again (both keyless calls carry the header) and with an empty cached terminal list trusted again
(the chat on the terminal is refused as unavailable); the nearby tests pass on both. On dev
ef67cc3fa, formatting the system OAuth header without the token check fails the system OAuth
case. On dev c15f28649, with orchestrators no longer skipped when the tool lists are fetched,
the saving test fails (the orchestrator is asked for `/openapi.json`).
"""

from __future__ import annotations

import json
import uuid
from typing import Callable

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import Answer, json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    TerminalRequest,
    configure_terminals,
    read_grant,
    serving_terminal,
)
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


def test_a_system_oauth_server_called_without_a_sign_in_token_gets_no_authorization(
    call_tool_server,
):
    headers = call_tool_server(auth_type="system_oauth", key="")

    assert "authorization" not in headers, headers["authorization"]


def test_a_session_server_gets_the_callers_own_token(call_tool_server):
    assert call_tool_server(auth_type="session", key="")["authorization"].startswith("Bearer ey")


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


# --- 5232f9bb2: a Session terminal reaches its orchestrator with the connection's key ------

ORCHESTRATOR_KEY = "orchestrator-key-0123"
TEMPLATE = {"id": "tpl-1", "data": {"name": "Workbench", "scope": "user"}}
TEMPLATES = "/api/v1/terminals/templates"


def _requiring(credential: str, answer: Answer) -> Callable[[TerminalRequest], Answer]:
    """Answer only a request carrying `credential`, as an orchestrator does; 401 otherwise."""

    def handler(request: TerminalRequest) -> Answer:
        if request.headers.get("authorization") != credential:
            return json_answer({"detail": "Not authenticated"}, status=401)
        return answer

    return handler


@pytest.fixture
def session_orchestrator():
    """A terminal orchestrator that answers its admin and spec routes only with its key."""
    with serving_terminal() as terminal:
        for method, path, answer in [
            ("GET", TEMPLATES, json_answer([TEMPLATE])),
            ("POST", TEMPLATES, json_answer(TEMPLATE, status=201)),
            ("GET", f"{TEMPLATES}/{TEMPLATE['id']}", json_answer(TEMPLATE)),
            ("GET", "/api/v1/terminals/status", json_answer({"version": "0.4.1"})),
            ("GET", "/files/cwd", json_answer({"cwd": "/home/user"})),
            ("GET", "/openapi.json", json_answer(_openapi("run_command", "/execute"))),
        ]:
            terminal.route(method, path, _requiring(f"Bearer {ORCHESTRATOR_KEY}", answer))
        yield terminal


def test_a_session_terminal_is_verified_with_its_orchestrator_key(admin, session_orchestrator):
    target = {
        "url": session_orchestrator.base_url,
        "key": ORCHESTRATOR_KEY,
        "auth_type": "session",
    }

    with admin.client() as client:
        answered = client.post("/api/v1/configs/terminal_servers/verify", json=target)

    assert (answered.status_code, answered.json()) == (
        200,
        {"status": True, "type": "orchestrator"},
    ), (
        "Verify Connection of a terminal with Session auth went out without the orchestrator's "
        f"key and was refused (#30001, #33579): {answered.status_code} {answered.text}"
    )


def _menu(client, terminal_id: str):
    listed = client.get("/api/v1/terminals/")
    return listed.status_code, [entry["name"] for entry in listed.json()]


def _status(client, terminal_id: str):
    connection_id = terminal_id.split(":")[1]
    answered = client.get(f"{TEMPLATES}/connections/{connection_id}/status")
    return answered.status_code, answered.json()


def _create(client, terminal_id: str):
    form = {"id": str(uuid.uuid4()), "connection_id": terminal_id.split(":")[1], "data": {}}
    answered = client.post(TEMPLATES, json=form)
    return answered.status_code, answered.json().get("terminal_id")


def _command(client, terminal_id: str):
    answered = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")
    return answered.status_code, answered.json()


# what the page does with an orchestrator: (the call, the answer a person with access gets)
TEMPLATE_CALLS = {
    "menu": (_menu, lambda terminal_id: (200, ["Workbench"])),
    "status": (_status, lambda terminal_id: (200, {"version": "0.4.1"})),
    "create": (_create, lambda terminal_id: (201, terminal_id)),
    "command": (_command, lambda terminal_id: (200, {"cwd": "/home/user"})),
}


@pytest.mark.parametrize("call", list(TEMPLATE_CALLS))
def test_a_session_orchestrator_saved_before_templates_is_still_reached_with_its_key(
    admin, preserve, session_orchestrator, call
):
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = session_orchestrator.connection(
        key=ORCHESTRATOR_KEY, auth_type="session", server_type="orchestrator"
    )
    terminal_id = f"template:{connection['id']}:{TEMPLATE['id']}"
    send, expected = TEMPLATE_CALLS[call]

    with admin.client() as client:
        configure_terminals(client, connection)
        answered = send(client, terminal_id)

    assert answered == expected(terminal_id), (
        f"{call} on an orchestrator stored with Session auth went out without the "
        f"orchestrator's key and was refused (#30001, #33579): {answered}"
    )
    credentials = {
        request.headers.get("authorization") for request in session_orchestrator.received
    }
    assert credentials == {f"Bearer {ORCHESTRATOR_KEY}"}, (
        f"the orchestrator was sent {credentials}; a session token of a user never belongs there"
    )


def test_a_session_terminal_offers_its_tools_and_runs_them_as_the_user(
    admin, make_user, preserve, upstream, session_orchestrator
):
    preserve(TERMINAL_SERVERS_CONFIG)
    member = make_user()
    session_orchestrator.route(
        "POST", "/execute", _requiring(f"Bearer {member.token}", json_answer({"output": "ran"}))
    )
    connection = session_orchestrator.connection(
        key=ORCHESTRATOR_KEY,
        auth_type="session",
        config={"access_grants": [read_grant(member.id)]},
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    prompt = f"list my files {uuid.uuid4().hex[:8]}"
    upstream.queue(
        reply.tool_call("run_command", {"command": "ls"}, match=reply.answering(prompt)),
        reply.text("listed", match=reply.answering(prompt)),
    )

    with member.client() as client:
        _, message = ask(client, prompt, terminal_id=connection["id"])

    assert message["content"] == "listed", (
        "a chat on a terminal with Session auth never got its tools, since the tool list was "
        f"fetched without the orchestrator's key (#33579): {message}"
    )
    [executed] = session_orchestrator.requests_to("/execute")
    assert executed.headers.get("authorization") == f"Bearer {member.token}", (
        "the command did not run under the user's own session"
    )


def test_saving_connections_fetches_a_tool_list_from_terminals_and_none_from_an_orchestrator(
    admin, preserve, session_orchestrator
):
    preserve(TERMINAL_SERVERS_CONFIG)
    orchestrator = session_orchestrator.connection(
        key=ORCHESTRATOR_KEY, auth_type="bearer", server_type="orchestrator"
    )
    with serving_terminal() as terminal, admin.client() as client:
        terminal.route("GET", "/openapi.json", json_answer(_openapi("run_command", "/execute")))
        configure_terminals(client, orchestrator, terminal.connection())
        menu = [entry["name"] for entry in client.get("/api/v1/terminals/").json()]
        fetched_from_terminal = len(terminal.requests_to("/openapi.json"))

    assert fetched_from_terminal == 1, (
        f"saving fetched the plain terminal's tool list {fetched_from_terminal} times"
    )
    assert sorted(menu) == ["Fake terminal", "Workbench"], f"the terminal menu shows {menu}"
    asked = [request.path for request in session_orchestrator.received]
    assert asked == [TEMPLATES], (
        "an orchestrator has no tools of its own, its terminals are asked one by one; saving "
        f"and listing asked it for {asked}"
    )
