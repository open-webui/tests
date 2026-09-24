"""Regression: four integration-surface defects fixed in Open WebUI 0.11.0.

* `9a6d16849` (PR #27243): `POST /api/chat/actions/{id}` ran the action with no availability
  check, so a disabled action, one the model does not offer and one on a model the caller cannot
  read all ran for anyone who posted the raw action id.
* `301bf519a` (PR #27413, issue #27239): `resolve_schema` dropped its visited set when it
  descended into `properties` and `items`, so two OpenAPI types referring to each other recursed
  until the stack gave out and the whole tool server failed to load.
* `a9a3e5b95` (PR #27423, discussion #27407): `GET /api/v1/users/{id}/preview` listed only the
  models, knowledge bases and tools shared with the user and left out the ones they own.
* `9a772f42c` and `09d4cccb7` (issue #26945): the terminal policy route `/p/<policy>` was built
  per call site and only for orchestrator connections, so a connection with a policy but another
  `server_type` reached the root route and the orchestrator started an unintended container.
  `get_terminal_server_url` now builds it for any policy, as one encoded path segment.

Twin of unit/tools/test_integration_surface.py.

Discriminates: passes on dev bbfa876af; fails with the availability gate of `9a6d16849` removed
(the three refused actions run), with `301bf519a`'s visited set dropped again (the server offers
no tool, not even its ordinary operations), with `a9a3e5b95` reverted (the owned items are
missing), with the orchestrator-only condition of `get_terminal_server_url` restored (the local
and untyped policy connections reach the root route) and with the policy id unquoted (the
escaping tests); every other test passes under each of these.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.listener import json_answer
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_model(client, **fields) -> str:
    model_id = f"surface-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Surface model",
        "meta": {},
        "params": {},
        **fields,
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return model_id


# --- 9a6d16849: an action runs only where the chat offers it -----------------------------

ACTION_SOURCE = """
import urllib.request


class Action:
{sub_actions}
    async def action(self, body: dict, __user__=None):
        urllib.request.urlopen(urllib.request.Request("{url}", data=b"{{}}"), timeout=5)
        return {{"ran": True}}
"""


@pytest.fixture
def action(admin, listener):
    """`action(active=..., sub_action=...)` installs an action that reports each run."""
    listener.route("POST", "/ran", json_answer({}))
    installed = []

    def install(active: bool = True, sub_action: bool = False) -> str:
        sub_actions = '    actions = [{"id": "sub", "name": "Sub"}]' if sub_action else ""
        source = ACTION_SOURCE.format(url=f"{listener.base_url}/ran", sub_actions=sub_actions)
        block = installed_function(admin, source, active=active)
        installed.append(block)
        return block.__enter__()

    yield install
    for block in reversed(installed):
        block.__exit__(None, None, None)


@pytest.fixture
def model_offering(admin):
    """`model_offering(action_id, readable=...)` is a preset whose chat offers that action."""
    created = []

    def create(action_id: str, readable: bool = True) -> str:
        with admin.client() as client:
            model_id = _create_model(
                client,
                meta={"actionIds": [action_id]},
                access_grants=[EVERYONE_READS] if readable else [],
            )
        created.append(model_id)
        return model_id

    yield create
    with admin.client() as client:
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _run_action(actor, action_id: str, model_id: str, **extra):
    body = {
        "model": model_id,
        "chat_id": f"temporary:{uuid.uuid4()}",
        "id": str(uuid.uuid4()),
        "session_id": "harness",
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    with actor.client() as client:
        return client.post(f"/api/chat/actions/{action_id}", json=body)


def test_an_inactive_action_is_refused(make_user, action, model_offering, listener):
    action_id = action(active=False)
    model_id = model_offering(action_id)

    response = _run_action(make_user(), action_id, model_id)

    assert response.status_code == 400, response.text
    assert listener.requests_to("/ran") == [], "a disabled action ran for a raw POST (#27243)"


def test_an_action_the_model_does_not_offer_is_refused(make_user, action, listener):
    action_id = action()

    response = _run_action(make_user(), action_id, MOCK_MODEL_ID)

    assert response.status_code == 400, response.text
    assert listener.requests_to("/ran") == [], (
        "an action the model does not offer ran for a raw POST (#27243)"
    )


def test_an_action_on_a_model_the_user_cannot_read_is_refused(
    make_user, action, model_offering, listener
):
    action_id = action()
    model_id = model_offering(action_id, readable=False)

    response = _run_action(make_user(), action_id, model_id)

    assert response.status_code == 400, response.text
    assert listener.requests_to("/ran") == [], (
        "an action ran on a model its caller cannot read (#27243)"
    )


def test_an_offered_action_on_a_readable_model_runs(make_user, action, model_offering, listener):
    action_id = action()
    model_id = model_offering(action_id)

    response = _run_action(make_user(), action_id, model_id)

    assert response.status_code == 200, response.text
    assert response.json() == {"ran": True}
    assert len(listener.requests_to("/ran")) == 1


def test_an_admin_runs_an_action_the_model_does_not_offer(admin, action, listener):
    action_id = action()

    response = _run_action(admin, action_id, MOCK_MODEL_ID)

    assert response.status_code == 200, response.text
    assert len(listener.requests_to("/ran")) == 1


def test_a_sub_action_of_an_offered_action_runs(make_user, action, model_offering, listener):
    action_id = action(sub_action=True)
    model_id = model_offering(action_id)

    response = _run_action(make_user(), f"{action_id}.sub", model_id)

    assert response.status_code == 200, response.text
    assert len(listener.requests_to("/ran")) == 1


def test_an_action_on_a_direct_connection_model_runs(make_user, action, listener):
    action_id = action()
    direct_model = {"id": "direct-model", "direct": True, "actions": []}

    response = _run_action(make_user(), action_id, "direct-model", model_item=direct_model)

    assert response.status_code == 200, response.text
    assert len(listener.requests_to("/ran")) == 1


@pytest.mark.slow
def test_actions_are_refused_while_plugins_are_off(instance_with):
    without_plugins = instance_with({"ENABLE_PLUGINS": "false"})

    response = _run_action(admin_of(without_plugins), "any_action", MOCK_MODEL_ID)

    assert response.status_code == 400
    assert "ENABLE_PLUGINS" in response.json()["detail"]


# --- 301bf519a: a tool server with mutually referring types still loads ------------------

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
CIRCULAR_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Graph", "version": "1"},
    "paths": {
        "/nodes": {
            "post": {
                "operationId": "create_node",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Node"}}
                    }
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/outer": {
            "post": {
                "operationId": "create_outer",
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Outer"}}
                    }
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/ping": {"get": {"operationId": "ping", "responses": {"200": {"description": "ok"}}}},
    },
    "components": {
        "schemas": {
            "Outer": {
                "type": "object",
                "properties": {"inner": {"$ref": "#/components/schemas/Inner"}},
            },
            "Inner": {"type": "object", "properties": {"name": {"type": "string"}}},
            "Node": {
                "type": "object",
                "properties": {"leaf": {"$ref": "#/components/schemas/Leaf"}},
            },
            "Leaf": {
                "type": "object",
                "properties": {"node": {"$ref": "#/components/schemas/Node"}},
            },
        }
    },
}


@pytest.fixture
def graph_server(admin, preserve, listener) -> str:
    """A saved tool server whose spec has two types referring to each other; returns its id."""
    listener.route("GET", "/graph/openapi.json", json_answer(CIRCULAR_SPEC))
    connection = {
        "url": f"{listener.base_url}/graph",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": "graph", "name": "Graph"},
    }
    preserve(TOOL_SERVERS_CONFIG)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    return "server:graph"


def test_a_tool_server_with_circular_refs_offers_its_tool(admin, upstream, graph_server):
    upstream.queue(reply.text("noted"))
    with admin.client() as client:
        ask(client, "make a node", tool_ids=[graph_server])

    offered = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in upstream.chat_requests()[0].get("tools", [])
    }
    assert "create_node" in offered, (
        f"the tool server with circular $refs offered no tool (#27239): {sorted(offered)}"
    )
    assert "leaf" in offered["create_node"]["properties"]


def test_ordinary_refs_and_empty_schemas_still_resolve(admin, upstream, graph_server):
    upstream.queue(reply.text("noted"))
    with admin.client() as client:
        ask(client, "make a node", tool_ids=[graph_server])

    offered = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in upstream.chat_requests()[0]["tools"]
    }
    inner = offered["create_outer"]["properties"]["inner"]
    assert inner["properties"]["name"]["type"] == "string"
    assert offered["ping"]["properties"] == {}


def test_a_tool_server_with_circular_refs_is_listed(admin, graph_server):
    with admin.client() as client:
        listed = client.get("/api/v1/tools/")

    assert graph_server in [tool["id"] for tool in listed.json()]


# --- a9a3e5b95: the access preview lists what the user owns ------------------------------


@pytest.fixture
def owner_items(make_user):
    """A new admin owning an unshared model, an inactive model, a knowledge base and a tool."""
    owner = make_user(role="admin")
    tool_id = f"owned_{uuid.uuid4().hex[:8]}"
    with owner.client() as client:
        model_id = _create_model(client)
        inactive_model_id = _create_model(client, is_active=False)
        knowledge = client.post(
            "/api/v1/knowledge/create", json={"name": "Owned", "description": ""}
        )
        tool = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": "Owned",
                "content": "class Tools:\n    pass\n",
                "meta": {},
            },
        )
        assert knowledge.status_code == tool.status_code == 200, (knowledge.text, tool.text)
        yield (
            owner,
            {
                "models": model_id,
                "knowledge": knowledge.json()["id"],
                "tools": tool_id,
                "inactive_model": inactive_model_id,
            },
        )
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")
        client.delete(f"/api/v1/knowledge/{knowledge.json()['id']}/delete")
        for created in (model_id, inactive_model_id):
            client.post("/api/v1/models/model/delete", json={"id": created})


def _preview(admin, user_id: str) -> dict:
    with admin.client() as client:
        preview = client.get(f"/api/v1/users/{user_id}/preview")
    assert preview.status_code == 200, preview.text
    return preview.json()


def _listed(preview: dict, kind: str) -> list[str]:
    return [item["id"] for item in preview[kind]["items"]]


@pytest.mark.parametrize("kind", ["models", "knowledge", "tools"])
def test_the_preview_lists_what_the_user_owns(admin, owner_items, kind):
    owner, owned = owner_items

    listed = _listed(_preview(admin, owner.id), kind)

    assert listed.count(owned[kind]) == 1, (
        f"the access preview left out {kind} the user owns (#27407): {listed}"
    )


def test_the_preview_leaves_out_an_owned_inactive_model(admin, owner_items):
    owner, owned = owner_items

    assert owned["inactive_model"] not in _listed(_preview(admin, owner.id), "models")


@pytest.mark.parametrize("kind", ["models", "knowledge", "tools"])
def test_the_preview_leaves_out_what_others_keep_to_themselves(admin, make_user, owner_items, kind):
    _, owned = owner_items

    assert owned[kind] not in _listed(_preview(admin, make_user().id), kind)


# --- 9a772f42c and 09d4cccb7: a policy routes through /p/<policy> ------------------------


@pytest.fixture
def terminal():
    with serving_terminal() as server:
        yield server


@pytest.fixture
def proxied_path(admin, preserve, terminal):
    """`proxied_path(**connection)` is the path a GET for /x arrives at through that connection."""
    preserve(TERMINAL_SERVERS_CONFIG)
    client = admin.client()

    def route(**fields) -> str:
        connection = terminal.connection(**fields)
        configure_terminals(client, connection)
        terminal.clear()
        client.get(f"/api/v1/terminals/{connection['id']}/x")
        return [request.path for request in terminal.received][-1]

    yield route
    client.close()


@pytest.mark.parametrize("server_type", ["orchestrator", "local", None])
def test_a_policy_routes_through_its_policy_path_whatever_the_server_type(
    proxied_path, server_type
):
    arrived_at = proxied_path(policy_id="pol1", server_type=server_type)

    assert arrived_at == "/p/pol1/x", (
        f"a {server_type} connection with a policy reached {arrived_at}; the orchestrator starts "
        "an unintended container on the root route (#26945)"
    )


@pytest.mark.parametrize(
    "policy_id, segment", [("a/b", "a%2Fb"), ("a?b", "a%3Fb"), ("../root", "..%2Froot")]
)
def test_a_policy_id_is_one_encoded_path_segment(proxied_path, policy_id, segment):
    assert proxied_path(policy_id=policy_id, server_type="orchestrator") == f"/p/{segment}/x"


@pytest.mark.parametrize("policy_id", [None, "", "   "])
def test_a_connection_without_a_policy_keeps_the_root_route(proxied_path, policy_id):
    assert proxied_path(policy_id=policy_id, server_type="orchestrator") == "/x"


def test_trailing_slashes_on_the_url_are_not_doubled(proxied_path, terminal):
    arrived_at = proxied_path(
        url=f"{terminal.base_url}///", policy_id="pol1", server_type="orchestrator"
    )
    assert arrived_at == "/p/pol1/x"
