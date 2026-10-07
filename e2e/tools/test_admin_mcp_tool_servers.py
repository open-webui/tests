"""Journey: an admin adds an MCP tool server over Streamable HTTP and people use it in chat.

In Integrations > External Tool Servers the admin switches the Add Connection dialog to MCP, names
the server (the Connection ID follows the name), enters its URL and the auth it needs, checks the
connection and opens it to everyone. A person picks the server under the chat's Tools, the model
calls its `echo` tool, named after the connection, and the reply shows what the tool returned. A
server keyed by an API key is reached with the key the admin typed, and the check refuses a wrong
one. Sign-in with OAuth 2.1 is in test_mcp_oauth_sign_in.py.

Discriminates: in a backend copy whose MCP connect leaves the bearer header out, the bearer case
and the key check fail; with the MCP branch of the chat's tool resolution skipped, both chat cases
fail (the model is offered no `echo`).
"""

from __future__ import annotations

import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.mcp_server import TOOL_SERVERS, serving_mcp
from utils.chat_ui import expect_reply, send
from utils.tool_servers import (
    choose_auth,
    make_public,
    open_add_connection,
    pick_tool,
    save,
    switch_to_mcp,
    tool_output,
    verify,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KEY = "chart-room-key"


@pytest.fixture
def admin_page(page_for, make_user, preserve):
    preserve(TOOL_SERVERS)
    return page_for(make_user(role="admin"))


def add_mcp_server(page, name: str, url: str, auth: str, key: str = "") -> None:
    form = open_add_connection(page)
    switch_to_mcp(form)
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("URL", exact=True).fill(url)
    choose_auth(form, auth)
    if key:
        form.get_by_placeholder("API Key").fill(key)
    verify(form)
    expect(page.get_by_text("Connection successful")).to_be_visible()
    make_public(page, form)
    save(page, form)


@pytest.mark.parametrize("auth", ["None", "Bearer"])
def test_an_mcp_server_the_admin_adds_is_called_in_chat(
    admin_page, page_for, make_user, upstream, auth
):
    token = secrets.token_hex(3)
    name = f"Chart Room {token}"
    phrase = f"the buoy {token} is green"
    with serving_mcp(bearer_key=KEY if auth == "Bearer" else None) as url:
        add_mcp_server(admin_page, name, url, auth, KEY if auth == "Bearer" else "")

        page = page_for(make_user())
        pick_tool(page, name)
        question = f"say it back {token}"
        upstream.queue(
            reply.tool_call(
                f"chart-room-{token}_echo", {"text": phrase}, match=reply.answering(question)
            ),
            reply.text("Said it back.", match=reply.answering(question)),
        )
        send(page, question)
        expect_reply(page, "Said it back.")

    expect(tool_output(page, f"chart-room-{token}_echo")).to_contain_text(phrase)
    tool_results = [
        entry["content"]
        for entry in upstream.chat_requests()[-1]["messages"]
        if entry["role"] == "tool"
    ]
    assert tool_results == [phrase], tool_results


def test_the_check_refuses_a_wrong_key(admin_page):
    with serving_mcp(bearer_key=KEY) as url:
        form = open_add_connection(admin_page)
        switch_to_mcp(form)
        form.get_by_label("Name", exact=True).fill(f"Chart Room {secrets.token_hex(3)}")
        form.get_by_label("URL", exact=True).fill(url)
        choose_auth(form, "Bearer")
        form.get_by_placeholder("API Key").fill("not-the-key")

        verify(form)

        expect(admin_page.get_by_text("Connection failed")).to_be_visible()
        expect(admin_page.get_by_text("Connection successful")).to_have_count(0)
