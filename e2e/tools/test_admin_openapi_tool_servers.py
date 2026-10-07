"""Journey: an admin adds an OpenAPI tool server in Admin Settings and people use it in chat.

The admin opens Integrations > External Tool Servers, fills the Add Connection dialog (name, URL,
the auth to send), checks the connection, opens it to everyone or to a group under Access Control
and saves. A person then finds the server under the chat's Tools, the model calls its operation
and the reply shows what the server answered. The server is called with no credentials, the
admin's key or the person's own session token, as the dialog's Auth says. A server the admin
switches off or deletes is gone from the people's Tools; one shared with a group is offered to its
members only. integration/tools/test_openapi_tool_server.py covers what a call sends and returns.

Discriminates: in a backend copy whose tools list skips the connection access check, the group
test fails (the outsider is offered the server); with the enable check dropped from the spec fetch
the switched-off test fails; with the session branch of the tool server headers sending the
connection key, the session case fails. In a frontend build whose dialog leaves the access grants
out of the saved connection, every chat test fails (the server stays the admin's).
"""

from __future__ import annotations

import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.access import make_group
from harness.group_tree import group_of
from harness.listener import json_answer
from harness.mcp_server import TOOL_SERVERS
from harness.openapi_server import authorization, calls_to, openapi_connection, serve_openapi
from utils.chat_ui import conversation, expect_reply, send
from utils.tool_servers import (
    choose_auth,
    connection_form,
    connection_row,
    make_public,
    open_add_connection,
    open_admin_integrations,
    open_tools_menu,
    pick_tool,
    save,
    share_with_group,
    verify,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TIDES = {"high": "12:04", "low": "18:31"}
EVERYONE = {"principal_type": "user", "principal_id": "*", "permission": "read"}


@pytest.fixture
def admin_page(page_for, make_user, preserve):
    preserve(TOOL_SERVERS)
    return page_for(make_user(role="admin"))


@pytest.fixture
def server_name():
    return f"Tide Table {secrets.token_hex(3)}"


def fill_connection(form, name: str, url: str) -> None:
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("URL", exact=True).fill(url)


def save_connections(admin, *connections: dict) -> None:
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": list(connections)})
    assert saved.status_code == 200, saved.text


def ask_for_the_tides(page, upstream) -> None:
    question = f"When are the tides today? {secrets.token_hex(3)}"
    upstream.queue(
        reply.tool_call("get_tides", {}, match=reply.answering(question)),
        reply.text("High water at noon.", match=reply.answering(question)),
    )
    send(page, question)
    expect_reply(page, "High water at noon.")


def test_verify_connection_tells_the_right_key_from_a_wrong_one(admin_page, listener, server_name):
    serve_openapi(listener, server_name, {"get_tides": json_answer(TIDES)}, bearer_key="sea-key")
    form = open_add_connection(admin_page)
    fill_connection(form, server_name, listener.base_url)
    choose_auth(form, "Bearer")

    form.get_by_placeholder("API Key").fill("wrong-key")
    verify(form)
    expect(admin_page.get_by_text("Connection failed")).to_be_visible()

    form.get_by_placeholder("API Key").fill("sea-key")
    verify(form)
    expect(admin_page.get_by_text("Connection successful")).to_be_visible()


@pytest.mark.parametrize("auth", ["None", "Bearer", "Session"])
def test_a_server_the_admin_adds_is_called_in_chat_with_its_auth(
    admin_page, page_for, make_user, upstream, listener, server_name, auth
):
    serve_openapi(listener, server_name, {"get_tides": json_answer(TIDES)})
    form = open_add_connection(admin_page)
    fill_connection(form, server_name, listener.base_url)
    choose_auth(form, auth)
    if auth == "Bearer":
        form.get_by_placeholder("API Key").fill("harbour-master-key")
    make_public(admin_page, form)
    save(admin_page, form)

    person = make_user()
    page = page_for(person)
    pick_tool(page, server_name)
    ask_for_the_tides(page, upstream)

    conversation(page).get_by_text("View Result from get_tides").click()
    expect(conversation(page).get_by_text("18:31")).to_be_visible()
    [call] = calls_to(listener, "get_tides")
    expected = {
        "None": None,
        "Bearer": "Bearer harbour-master-key",
        "Session": f"Bearer {person.token}",
    }
    assert authorization(call) == expected[auth]


def test_a_server_switched_off_is_no_longer_offered(
    admin_page, admin, page_for, make_user, listener, server_name
):
    serve_openapi(listener, server_name, {"get_tides": json_answer(TIDES)})
    control = f"Wind Rose {secrets.token_hex(3)}"
    save_connections(
        admin,
        openapi_connection(listener.base_url, server_name, [EVERYONE]),
        openapi_connection(listener.base_url, control, [EVERYONE]),
    )
    person_page = page_for(make_user())
    expect(open_tools_menu(person_page).get_by_role("button", name=server_name)).to_be_visible()

    settings = open_admin_integrations(admin_page)
    switch = connection_row(settings, server_name).get_by_role("switch")
    expect(switch).to_be_checked()
    switch.click()
    expect(admin_page.get_by_text("Connections saved successfully")).to_be_visible()

    menu = open_tools_menu(person_page)
    expect(menu.get_by_role("button", name=control)).to_be_visible()
    expect(menu.get_by_role("button", name=server_name)).to_have_count(0)


def test_a_deleted_server_is_no_longer_offered(
    admin_page, admin, page_for, make_user, listener, server_name
):
    serve_openapi(listener, server_name, {"get_tides": json_answer(TIDES)})
    control = f"Wind Rose {secrets.token_hex(3)}"
    save_connections(
        admin,
        openapi_connection(listener.base_url, server_name, [EVERYONE]),
        openapi_connection(listener.base_url, control, [EVERYONE]),
    )
    person_page = page_for(make_user())
    expect(open_tools_menu(person_page).get_by_role("button", name=server_name)).to_be_visible()

    settings = open_admin_integrations(admin_page)
    tooltip_button(connection_row(settings, server_name), "Configure").click()
    form = connection_form(admin_page, "Edit Connection")
    form.get_by_role("button", name="Delete").click()
    admin_page.get_by_role("dialog").get_by_role("button", name="Delete").last.click()
    expect(admin_page.get_by_text("Connections saved successfully")).to_be_visible()
    expect(settings.get_by_text(server_name)).to_have_count(0)

    menu = open_tools_menu(person_page)
    expect(menu.get_by_role("button", name=control)).to_be_visible()
    expect(menu.get_by_role("button", name=server_name)).to_have_count(0)


def test_a_server_shared_with_a_group_is_offered_to_its_members_only(
    admin_page, admin, page_for, make_user, upstream, listener, server_name
):
    serve_openapi(listener, server_name, {"get_tides": json_answer(TIDES)})
    member, outsider = make_user(), make_user()
    group_name = group_of(admin, make_group(admin, [member]))["name"]
    control = f"Wind Rose {secrets.token_hex(3)}"
    save_connections(admin, openapi_connection(listener.base_url, control, [EVERYONE]))

    form = open_add_connection(admin_page)
    fill_connection(form, server_name, listener.base_url)
    choose_auth(form, "None")
    share_with_group(admin_page, form, group_name)
    save(admin_page, form)

    outsider_menu = open_tools_menu(page_for(outsider))
    expect(outsider_menu.get_by_role("button", name=control)).to_be_visible()
    expect(outsider_menu.get_by_role("button", name=server_name)).to_have_count(0)

    member_page = page_for(member)
    pick_tool(member_page, server_name)
    ask_for_the_tides(member_page, upstream)
    assert len(calls_to(listener, "get_tides")) == 1
