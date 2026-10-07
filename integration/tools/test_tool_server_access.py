"""Journey: who is offered the tool server an admin added, for OpenAPI and MCP servers alike.

A tool server granted to a group is listed in `GET /api/v1/tools/` for its members and, when a
chat names it in `tool_ids`, the provider is offered its tools; someone outside the group gets
neither, even when their chat names the server. A server with no grants is the admins' only, and
one the admin switched off is neither listed nor offered, to anyone. integration/tools/
test_mcp_tool_discovery.py covers the specs route of an MCP server.

Discriminates: in a backend copy whose tools list skips the connection access check, the
outsider and no-grants tests fail on the list; with the check skipped when a chat resolves its
tools, they fail on the offered tools; with the enable check dropped from the list and from the
resolution, the switched-off test fails; with group grants ignored, the member test fails.
"""

from __future__ import annotations

import contextlib
import secrets
from dataclasses import dataclass

import pytest

from harness.access import grant, make_group
from harness.listener import json_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.openapi_server import openapi_connection, serve_openapi
from harness.tool_calls import offered_tools

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@dataclass(frozen=True)
class Server:
    tool_id: str  # the id the tools list shows and a chat names
    offers: str  # the name of one tool it offers the model


@pytest.fixture(params=["openapi", "mcp"])
def add_server(request, admin, preserve, listener):
    """`add(grants, enabled)` saves one server of this kind as the admin's and returns it."""
    preserve(TOOL_SERVERS)
    with contextlib.ExitStack() as stack:

        def add(grants: list[dict], enabled: bool = True) -> Server:
            suffix = secrets.token_hex(3)
            if request.param == "openapi":
                title = f"Tide Table {suffix}"
                serve_openapi(listener, title, {"get_tides": json_answer({"high": "12:04"})})
                connection = openapi_connection(listener.base_url, title, grants)
                server = Server(f"server:{connection['info']['id']}", "get_tides")
            else:
                server_id = f"harbour_{suffix}"
                url = stack.enter_context(serving_mcp())
                connection = mcp_connection(url, server_id, grants)
                server = Server(f"server:mcp:{server_id}", f"{server_id}_echo")
            connection["config"]["enable"] = enabled
            with admin.client() as client:
                saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
            assert saved.status_code == 200, saved.text
            return server

        yield add


def listed_ids(person) -> set[str]:
    with person.client() as client:
        listed = client.get("/api/v1/tools/")
    assert listed.status_code == 200, listed.text
    return {tool["id"] for tool in listed.json()}


def offered_when_named(person, upstream, server: Server) -> set[str]:
    with person.client() as client:
        return offered_tools(client, upstream, tool_ids=[server.tool_id])


def test_a_group_member_is_listed_and_offered_the_servers_tools(
    admin, make_user, upstream, add_server
):
    member = make_user()
    group_id = make_group(admin, [member])
    server = add_server([grant("group", group_id, "read")])

    assert server.tool_id in listed_ids(member)
    assert server.offers in offered_when_named(member, upstream, server)


def test_someone_outside_the_group_is_neither_listed_nor_offered_the_servers_tools(
    admin, make_user, upstream, add_server
):
    outsider = make_user()
    group_id = make_group(admin, [make_user()])
    server = add_server([grant("group", group_id, "read")])

    assert server.tool_id not in listed_ids(outsider)
    assert server.offers not in offered_when_named(outsider, upstream, server)


def test_a_server_without_grants_is_the_admins_only(admin, make_user, upstream, add_server):
    server = add_server([])

    assert server.tool_id in listed_ids(admin)
    assert server.offers in offered_when_named(admin, upstream, server)
    user = make_user()
    assert server.tool_id not in listed_ids(user)
    assert server.offers not in offered_when_named(user, upstream, server)


@pytest.mark.parametrize("who", ["admin", "member"])
def test_a_switched_off_server_is_neither_listed_nor_offered_to_anyone(
    admin, make_user, upstream, add_server, who
):
    member = make_user()
    group_id = make_group(admin, [member])
    server = add_server([grant("group", group_id, "read")], enabled=False)

    person = {"admin": admin, "member": member}[who]
    assert server.tool_id not in listed_ids(person)
    assert server.offers not in offered_when_named(person, upstream, server)
