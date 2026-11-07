"""Journey: the chat's Available Tools dialog lists what an MCP tool server offers.

A person turns on an MCP tool server for the chat and opens Available Tools, shown beside the
+ button with Show Input Selections on (off by default since dev 834c47ee0). The server's entry
folds open to its tools, each with its description and the count beside the name, fetched when the
entry is first opened. A server that cannot be reached shows "Unable to load tools" with a Retry
that fetches again once the server is back. The sign-in states ("Auth required", Reconnect) are
not driven here: a server that needs sign-in cannot be turned on from the menu without leaving
for the sign-in, so integration/tools/test_mcp_tool_discovery.py pins the 401 behind them.

Discriminates: in a frontend build with the dialog's tool discovery removed (the entry listing no
tools, as before the change), the listing and retry tests go red.
"""

from __future__ import annotations

import re
import secrets

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.instance import free_port
from harness.mcp_server import ECHO_DESCRIPTION, TOOL_SERVERS, mcp_connection, serving_mcp
from harness.terminal_server import read_grant
from utils.chat_ui import close_menu, open_integrations_menu, show_input_selections

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def server_name():
    return f"Chart Room {secrets.token_hex(3)}"


def save_connection(admin, preserve, connection: dict) -> None:
    preserve(TOOL_SERVERS)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text


def named(connection: dict, name: str) -> dict:
    return {**connection, "info": {**connection["info"], "name": name}}


def open_tools_dialog(page: Page, server_name: str) -> Locator:
    page.goto("/")
    menu = open_integrations_menu(page)
    menu.get_by_role("menuitem", name=re.compile(r"^Tools")).click()
    menu.get_by_role("menuitemcheckbox", name=re.compile(server_name)).click()
    close_menu(page)
    page.get_by_role("button", name="Available Tools").click()
    dialog = page.get_by_role("dialog").filter(has_text="Available Tools")
    expect(dialog).to_be_visible()
    return dialog


def test_an_mcp_servers_tools_are_listed_with_their_descriptions(
    page_for, make_user, admin, preserve, server_name
):
    person = make_user()
    show_input_selections(person)
    server_id = f"chart_{secrets.token_hex(3)}"
    with serving_mcp(media=True) as url:
        connection = mcp_connection(url, server_id, [read_grant("*")])
        save_connection(admin, preserve, named(connection, server_name))
        dialog = open_tools_dialog(page_for(person), server_name)

        dialog.get_by_text(server_name).click()

        expect(dialog.get_by_text("3 tools")).to_be_visible()
        expect(dialog.get_by_text("echo", exact=True)).to_be_visible()
        expect(dialog.get_by_text(ECHO_DESCRIPTION)).to_be_visible()
        expect(dialog.get_by_text("snapshot", exact=True)).to_be_visible()


def test_a_server_that_is_down_offers_a_retry_that_lists_its_tools_once_it_is_back(
    page_for, make_user, admin, preserve, server_name
):
    person = make_user()
    show_input_selections(person)
    server_id = f"chart_{secrets.token_hex(3)}"
    port = free_port()
    connection = mcp_connection(f"http://127.0.0.1:{port}/mcp", server_id, [read_grant("*")])
    save_connection(admin, preserve, named(connection, server_name))
    dialog = open_tools_dialog(page_for(person), server_name)

    dialog.get_by_text(server_name).click()

    expect(dialog.get_by_text("Unable to load tools")).to_be_visible()
    with serving_mcp(port=port):
        dialog.get_by_role("button", name="Retry").click()
        expect(dialog.get_by_text("echo", exact=True)).to_be_visible()
        expect(dialog.get_by_text("1 tool", exact=True)).to_be_visible()
    expect(dialog.get_by_text("Unable to load tools")).to_have_count(0)
