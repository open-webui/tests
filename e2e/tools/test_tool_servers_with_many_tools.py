"""Journey: people find one tool server among many and an admin filters one with many operations.

With a dozen tool servers the person types part of a name into Integrations > Tools > "Search
tools" and only the servers with that in their name stay listed; picking one turns it on and the
model is offered that server's operations and no other server's. For a server with thirty
operations the admin opens Configure in Admin Settings > Integrations and types a Function Name
Filter List. An entry names the end of an operation name (`get_tide` allows `get_tide` and
`old_get_tide`, not `get_tides`), a `!` before it takes matching operations out, and a list of
exclusions alone allows everything else. A person who picks the server is then offered exactly the
operations the list lets through. The chat's Available Tools dialog under a filter is the
integration discovery test over HTTP plus test_mcp_tool_server_modal.py, so it is not driven here.

Discriminates: in a backend copy whose chat tool resolution ignores the filter list of OpenAPI
servers, every filter test fails (the model is offered all thirty operations). In a frontend build
whose Search tools box leaves the list unfiltered, the search test fails (the other servers stay).
"""

from __future__ import annotations

import contextlib
import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.listener import json_answer, listening
from harness.mcp_server import TOOL_SERVERS
from harness.openapi_server import openapi_connection, serve_openapi
from utils.chat_ui import expect_reply, send
from utils.tool_servers import (
    connection_form,
    connection_row,
    open_admin_integrations,
    open_tools_menu,
    pick_tool,
    save,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE = {"principal_type": "user", "principal_id": "*", "permission": "read"}
NAMED = ["get_tide", "get_tides", "get_wind", "get_winds", "old_get_tide", "old_wind"]
OPERATIONS = NAMED + [f"gauge_{number:02d}" for number in range(24)]


@pytest.fixture
def admin_page(page_for, make_user, preserve):
    preserve(TOOL_SERVERS)
    return page_for(make_user(role="admin"))


def save_connections(admin, *connections: dict) -> None:
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": list(connections)})
    assert saved.status_code == 200, saved.text


def offered_operations(upstream, question: str, among: set[str]) -> set[str]:
    """The operations of `among` the model was sent; the built-in tools are not the servers'."""
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return {tool["function"]["name"] for tool in request["tools"]} & among


def ask(page, upstream) -> str:
    question = f"What can you do here? {secrets.token_hex(3)}"
    upstream.queue(reply.text("Plenty.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Plenty.")
    return question


def test_a_server_among_many_is_found_by_part_of_its_name_and_offered_alone(
    admin, page_for, make_user, upstream, listener
):
    token = secrets.token_hex(3)
    target = f"Tide Table {token}"
    control = f"Wind Rose {token}"
    fillers = [f"Harbour Light {number} {token}" for number in range(10)]
    serve_openapi(listener, target, {"get_tides": json_answer({})})
    with contextlib.ExitStack() as stack:
        other = stack.enter_context(listening())
        serve_openapi(other, control, {"get_gusts": json_answer({})})
        save_connections(
            admin,
            openapi_connection(listener.base_url, target, [EVERYONE]),
            openapi_connection(other.base_url, control, [EVERYONE]),
            *[openapi_connection(other.base_url, name, [EVERYONE]) for name in fillers],
        )
        page = page_for(make_user())
        menu = open_tools_menu(page)
        expect(menu.get_by_role("button", name=control)).to_be_visible()
        expect(menu.get_by_role("button", name=fillers[0])).to_be_visible()

        menu.get_by_placeholder("Search tools").fill("tide tab")

        expect(menu.get_by_role("button", name=control)).to_have_count(0)
        expect(menu.get_by_role("button", name=fillers[0])).to_have_count(0)
        expect(menu.get_by_role("button", name=target)).to_be_visible()
        menu.get_by_role("button", name=target).click()
        expect(menu.get_by_role("button", name=target)).to_have_attribute("aria-pressed", "true")
        page.keyboard.press("Escape")
        question = ask(page, upstream)

    assert offered_operations(upstream, question, {"get_tides", "get_gusts"}) == {"get_tides"}


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ("get_tide, get_wind", {"get_tide", "get_wind", "old_get_tide"}),
        ("!get_tide, !get_wind", set(OPERATIONS) - {"get_tide", "get_wind", "old_get_tide"}),
        ("_wind", {"get_wind", "old_wind"}),
        ("_tide, !old_get_tide", {"get_tide"}),
    ],
    ids=["plain list", "exclusions only", "end of the name", "allowed and excluded"],
)
def test_a_filter_the_admin_types_leaves_a_person_the_operations_it_allows(
    admin_page, admin, page_for, make_user, upstream, listener, entries, expected
):
    server_name = f"Tide Table {secrets.token_hex(3)}"
    serve_openapi(listener, server_name, {name: json_answer({}) for name in OPERATIONS})
    save_connections(admin, openapi_connection(listener.base_url, server_name, [EVERYONE]))
    page = page_for(make_user())
    pick_tool(page, server_name)
    question = ask(page, upstream)
    assert offered_operations(upstream, question, set(OPERATIONS)) == set(OPERATIONS)

    settings = open_admin_integrations(admin_page)
    tooltip_button(connection_row(settings, server_name), "Configure").click()
    form = connection_form(admin_page, "Edit Connection")
    form.get_by_label("Function Name Filter List").fill(entries)
    save(admin_page, form)

    page = page_for(make_user())
    pick_tool(page, server_name)
    question = ask(page, upstream)
    assert offered_operations(upstream, question, set(OPERATIONS)) == expected
