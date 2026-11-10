"""Journey: OpenAPI tool servers and terminal servers reached by host name, under both resolvers.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` decides which resolver every aiohttp connector Open WebUI opens
uses, and each call an admin or a chat makes to a tool server or a terminal opens its own session.
Each test names the service by `localhost`, a hosts-file name for `::1` alone or a hosts-file name
for another local address and expects the same outcome under both resolvers. An OpenAPI tool server
is verified by the admin, its spec loaded when a chat uses it and its operation called by the
model. A terminal server is verified, an orchestrator has its status read and its templates
listed, created and changed (the policy, lifecycle and refresh routes went with dev 6f2fc7bc7), a
terminal has its files and terminal sessions proxied for the web client, and serves a chat its
spec, working directory, system prompt, AGENTS.md, skills and tool call. A server whose name does
not resolve fails each of those with the same status and message, the resolver's own wording
aside, and a chat that names it still answers. The MCP SDK talks through httpx and the terminal
working directory call of automations is never reached, so neither is covered here.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271).
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from contextlib import contextmanager

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.host_names import (
    FAILS_WITHIN,
    UNRESOLVABLE,
    name_form,
    name_forms,
    serving_by_name,
    timed,
    without_resolver_reason,
)
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    close_of,
    configure_terminals,
    serving_terminal,
    terminal_session,
)
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
VERIFY_TOOL_SERVER = "/api/v1/configs/tool_servers/verify"
SERVER_ID = "named-pets"
TOOL_IDS = [f"server:{SERVER_ID}"]
UNRESOLVABLE_URL = f"http://{UNRESOLVABLE}:8080"
HOME = "/home/ada"
AGENTS_MD = "Answer in British English."
SYSTEM_PROMPT = "You are working in a named terminal."
OK = {"responses": {"200": {"description": "ok"}}}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pets", "version": "1"},
    "paths": {
        "/lookup": {
            "get": {
                "operationId": "lookup",
                **OK,
                "parameters": [{"name": "name", "in": "query", "schema": {"type": "string"}}],
            }
        }
    },
}
TERMINAL_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {"/execute": {"post": {"operationId": "run_command", **OK}}},
}
SKILLS = [
    {
        "id": "terminal:ship-it",
        "name": "ship-it",
        "description": "Ship the app to production",
        "location": f"{HOME}/.skills/ship-it/SKILL.md",
    }
]
CONNECTION_FAILED = "Failed to connect to the tool server"


def _tool_server(url: str, **fields) -> dict:
    return {
        "url": url,
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": SERVER_ID, "name": "Pets"},
        **fields,
    }


@contextmanager
def _terminal_by_name(label: str):
    """The fake terminal server on the name form's address, its `base_url` naming it by host."""
    form = name_form(label)
    with serving_terminal(form.address) as server:
        port = server.base_url.rsplit(":", 1)[1]
        yield dataclasses.replace(server, base_url=f"http://{form.host}:{port}")


def _serve_terminal(server) -> None:
    server.route("GET", "/openapi.json", json_answer(TERMINAL_SPEC))
    server.route("GET", "/api/config", json_answer({"features": {"system": True}}))
    server.route("GET", "/system", json_answer({"prompt": SYSTEM_PROMPT}))
    server.route("GET", "/files/cwd", json_answer({"cwd": f"{HOME}/work", "home": HOME}))
    server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
    server.route("GET", "/skills", json_answer(SKILLS))
    server.route("POST", "/execute", json_answer({"output": "ran by name"}))


@pytest.mark.parametrize("name_form", name_forms())
def test_an_openapi_tool_server_is_verified_loaded_and_called_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(TOOL_SERVERS_CONFIG, on=resolving_instance)
    with serving_by_name(name_form) as server, resolving_admin.client() as client:
        server.route("GET", "/pets/openapi.json", json_answer(SPEC))
        server.route("GET", "/pets/lookup", json_answer({"found": "Rex"}))
        connection = _tool_server(f"{server.base_url}/pets")
        verified = client.post(VERIFY_TOOL_SERVER, json=connection)
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        result = run_tool(
            client, resolving_instance.upstream, "lookup", {"name": "Rex"}, tool_ids=TOOL_IDS
        )

    assert verified.status_code == 200, verified.text
    assert list(verified.json()["paths"]) == ["/lookup"]
    assert saved.status_code == 200, saved.text
    offered = resolving_instance.upstream.chat_requests()[0]["tools"]
    assert "lookup" in [tool["function"]["name"] for tool in offered]
    assert json.loads(result) == {"found": "Rex"}
    [called] = server.requests_to("/pets/lookup")
    assert called.path == "/pets/lookup?name=Rex"
    assert server.requests_to("/pets/openapi.json"), "the spec was never fetched"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_openapi_tool_server_fails_verification_the_same_way(
    resolver, resolving_admin
):
    with resolving_admin.client() as client:
        verified, seconds = timed(
            client.post, VERIFY_TOOL_SERVER, json=_tool_server(UNRESOLVABLE_URL)
        )

    assert (verified.status_code, verified.json()) == (400, {"detail": CONNECTION_FAILED})
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_chat_with_an_unresolvable_openapi_tool_server_still_answers(
    resolver, resolving_instance, resolving_admin, preserve
):
    preserve(TOOL_SERVERS_CONFIG, on=resolving_instance)
    connection = _tool_server(UNRESOLVABLE_URL)
    with resolving_admin.client() as client:
        client.post(
            TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
        ).raise_for_status()
        resolving_instance.upstream.queue(reply.text("no tools today"))
        started_reply, seconds = timed(ask, client, "anyone there?", tool_ids=TOOL_IDS)

    _, message = started_reply
    assert message["content"] == "no tools today", message
    offered = resolving_instance.upstream.chat_requests()[0].get("tools") or []
    assert "lookup" not in [tool["function"]["name"] for tool in offered]
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_tool_call_to_an_unresolvable_server_tells_the_model_the_same_error(
    resolver, resolving_instance, resolving_admin, preserve
):
    preserve(TOOL_SERVERS_CONFIG, on=resolving_instance)
    connection = _tool_server(UNRESOLVABLE_URL, spec_type="json", spec=json.dumps(SPEC))
    with resolving_admin.client() as client:
        client.post(
            TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
        ).raise_for_status()
        result, seconds = timed(
            run_tool, client, resolving_instance.upstream, "lookup", {}, tool_ids=TOOL_IDS
        )

    assert without_resolver_reason(json.loads(result)["error"]) == (
        f"Cannot connect to host {UNRESOLVABLE}:8080 ssl:default [<resolver reason>]"
    )
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.parametrize("name_form", name_forms())
def test_a_terminal_serves_a_chat_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG, on=resolving_instance)
    upstream = resolving_instance.upstream
    with _terminal_by_name(name_form) as server, resolving_admin.client() as client:
        _serve_terminal(server)
        connection = server.connection()
        configure_terminals(client, connection)
        upstream.queue(reply.tool_call("run_command", {}), reply.text("done"))
        _, message = ask(client, "tidy up", terminal_id=connection["id"])

    assert message["content"] == "done", message
    first, second = upstream.chat_requests()
    [run_command] = [
        tool["function"] for tool in first["tools"] if tool["function"]["name"] == "run_command"
    ]
    assert run_command["description"].endswith(f"The current working directory is: {HOME}/work")
    sent = json.dumps(first["messages"])
    assert SYSTEM_PROMPT in sent
    assert AGENTS_MD in sent
    assert "ship-it" in sent
    assert json.loads(second["messages"][-1]["content"]) == {"output": "ran by name"}
    assert len(server.requests_to("/execute")) == 1


@pytest.mark.parametrize("name_form", name_forms())
def test_a_terminal_is_verified_and_its_orchestrator_is_managed_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG, on=resolving_instance)
    templates = "/api/v1/terminals/templates"
    template = {"id": "tpl-1", "data": {"name": "Workbench"}}
    renamed = {"id": "tpl-1", "data": {"name": "Bench"}}
    with _terminal_by_name(name_form) as server, resolving_admin.client() as client:
        server.route("GET", "/api/config", json_answer({"features": {}}))
        target = {"url": server.base_url, "key": "", "auth_type": "none"}
        plain = client.post("/api/v1/configs/terminal_servers/verify", json=target)
        server.route("GET", templates, json_answer([template]))
        server.route("POST", templates, json_answer(template, status=201))
        server.route("GET", f"{templates}/tpl-1", json_answer(template))
        server.route("PATCH", f"{templates}/tpl-1", json_answer(renamed))
        server.route("GET", "/api/v1/terminals/status", json_answer({"version": "0.4.1"}))
        orchestrator = client.post("/api/v1/configs/terminal_servers/verify", json=target)
        connection = server.connection(
            key="orchestrator-key", auth_type="bearer", server_type="orchestrator"
        )
        configure_terminals(client, connection)
        terminal_id = f"template:{connection['id']}:tpl-1"
        status = client.get(f"{templates}/connections/{connection['id']}/status")
        created = client.post(
            templates,
            json={"id": str(uuid.uuid4()), "connection_id": connection["id"], "data": {}},
        )
        changed = client.patch(f"{templates}/{terminal_id}", json={"name": "Bench"})
        listed = client.get("/api/v1/terminals/")

    assert (plain.status_code, plain.json()) == (200, {"status": True, "type": "terminal"})
    assert (orchestrator.status_code, orchestrator.json()) == (
        200,
        {"status": True, "type": "orchestrator"},
    )
    assert (status.status_code, status.json()) == (200, {"version": "0.4.1"})
    assert (created.status_code, created.json()) == (201, {**template, "terminal_id": terminal_id})
    assert (changed.status_code, changed.json()) == (200, renamed)
    assert [(entry["id"], entry["name"]) for entry in listed.json()] == [(terminal_id, "Workbench")]
    [written] = [
        entry for entry in server.requests_to(f"{templates}/tpl-1") if entry.method == "PATCH"
    ]
    assert json.loads(written.body) == {"name": "Bench"}


@pytest.mark.parametrize("name_form", name_forms())
def test_a_terminal_proxies_files_and_sessions_for_the_web_client_by_name(
    resolver, name_form, resolving_instance, resolving_admin, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG, on=resolving_instance)
    with _terminal_by_name(name_form) as server, resolving_admin.client() as client:
        _serve_terminal(server)
        server.route("POST", "/files/upload", json_answer({"path": f"{HOME}/a.txt"}))
        connection = server.connection()
        configure_terminals(client, connection)
        listed = client.get("/api/v1/terminals/")
        fetched = client.get(f"/api/v1/terminals/{connection['id']}/files/cwd")
        uploaded = client.post(
            f"/api/v1/terminals/{connection['id']}/files/upload", content=b"hello"
        )
        with terminal_session(
            resolving_instance.base_url, resolving_admin.token, connection["id"]
        ) as session:
            session.send("ls")
            echoed = session.recv(timeout=10)

    assert [entry["id"] for entry in listed.json()] == [connection["id"]]
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == {"cwd": f"{HOME}/work", "home": HOME}
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json() == {"path": f"{HOME}/a.txt"}
    assert echoed == "ls"
    [terminal] = server.sessions
    assert terminal.auth is not None


@pytest.mark.usefixtures("refuses_unknown_names")
@pytest.mark.parametrize(
    "path,detail",
    [
        pytest.param("verify", "Failed to connect to the terminal server", id="verify"),
        pytest.param("policy", "Failed to access policy on terminal server", id="policy"),
        pytest.param("lifecycle", "Failed to access lifecycle on terminal server", id="lifecycle"),
        pytest.param("refresh", "Failed to refresh terminals", id="refresh"),
    ],
)
def test_an_unresolvable_terminal_fails_every_admin_call_the_same_way(
    resolver, resolving_admin, path, detail
):
    target = {"url": UNRESOLVABLE_URL, "key": "", "auth_type": "none", "policy_id": "dev"}
    with resolving_admin.client() as client:
        answer, seconds = timed(
            client.post, f"/api/v1/configs/terminal_servers/{path}", json=target
        )

    assert (answer.status_code, answer.json()) == (400, {"detail": detail})
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.fixture
def gone_terminal(resolving_instance, resolving_admin, preserve) -> str:
    """A terminal the admin connected under a name that does not resolve; yields its id."""
    preserve(TERMINAL_SERVERS_CONFIG, on=resolving_instance)
    connection = {
        "id": "gone-terminal",
        "name": "Gone",
        "enabled": True,
        "url": UNRESOLVABLE_URL,
        "path": "/openapi.json",
        "key": "",
        "auth_type": "none",
        "config": {"access_grants": []},
    }
    with resolving_admin.client() as client:
        configure_terminals(client, connection)
    return connection["id"]


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_terminal_fails_its_proxy_the_same_way(
    resolver, resolving_admin, gone_terminal
):
    with resolving_admin.client() as client:
        proxied, seconds = timed(client.get, f"/api/v1/terminals/{gone_terminal}/files/cwd")

    assert proxied.status_code == 502, proxied.text
    assert without_resolver_reason(proxied.json()["error"]) == (
        f"Terminal proxy error: Cannot connect to host {UNRESOLVABLE}:8080 "
        "ssl:default [<resolver reason>]"
    )
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_terminal_fails_its_chat_the_same_way(
    resolver, resolving_admin, gone_terminal
):
    with resolving_admin.client() as client:
        answered, seconds = timed(ask, client, "run it", terminal_id=gone_terminal)

    _, message = answered
    assert without_resolver_reason(message["error"]["content"]) == (
        f"Cannot connect to host {UNRESOLVABLE}:8080 ssl:default [<resolver reason>]"
    )
    assert seconds < FAILS_WITHIN, f"the lookup failed only after {seconds:.1f}s"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_unresolvable_terminal_closes_its_session_the_same_way(
    resolver, resolving_instance, resolving_admin, gone_terminal
):
    started = time.monotonic()
    with terminal_session(
        resolving_instance.base_url, resolving_admin.token, gone_terminal
    ) as session:
        closed = close_of(session, timeout=FAILS_WITHIN * 2)
    seconds = time.monotonic() - started

    assert closed is not None, "the session stayed open"
    assert seconds < FAILS_WITHIN, f"the session ended only after {seconds:.1f}s"
