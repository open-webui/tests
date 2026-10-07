"""Journey: an MCP tool server behind OAuth 2.1, from the admin's registration to signing out.

The admin adds the server in the Add Connection dialog with Auth set to OAuth 2.1: Check OAuth
Discovery finds its authorization server and Register Client registers Open WebUI there before
the connection is saved. A person presses the server under the chat's Tools, is sent through the
sign-in (the harness's authorization server approves at once) and comes back with the server on
for the chat; the model's tool call carries their token. Later chats reuse that token without a
new sign-in. Disconnect OAuth signs them out: the server needs a sign-in again, and the next one
issues a fresh token. integration/tools/test_mcp_oauth_connection.py covers refresh and revocation.
Sign-ins are counted by the code grants the authorization server answered: Open WebUI's backend
also fetches the authorize URL itself before sending the browser there.

Red on dev ebc6add67: choosing OAuth 2.1 in the dialog leaves its check button labelled and
tooltipped "Verify Connection". f822605b3 meant it to read "Check OAuth Discovery", but the label
is computed by a function the template calls without naming the auth type, so it never updates.

Discriminates: on dev ebc6add67, in a backend copy that never stores the token from the sign-in
callback every sign-in test fails (the server never shows as connected); with the stored token
dropped once a chat call has used it the reuse test fails (the second chat needs a new sign-in);
with the OAuth session delete answering success without deleting the sign-out test fails. In a
frontend build whose dialog saves no access grants the admin's test fails, and in one whose check
button label follows the auth type the label test passes.
"""

from __future__ import annotations

import re
import secrets

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.mcp_oauth import register_oauth_mcp, serving_protected_mcp
from harness.mcp_server import TOOL_SERVERS
from utils.chat_ui import expect_reply, send
from utils.tool_servers import (
    choose_auth,
    make_public,
    open_add_connection,
    open_tools_menu,
    save,
    switch_to_mcp,
    tool_output,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANY = re.compile(".*")
CHECK = re.compile("Check OAuth Discovery|Verify Connection")
EVERYONE = {"principal_type": "user", "principal_id": "*", "permission": "read"}


@pytest.fixture
def mcp(preserve):
    preserve(TOOL_SERVERS)
    with serving_protected_mcp() as protected:
        yield protected


def sign_ins(mcp) -> list[dict]:
    return mcp.auth_server.token_grants("authorization_code")


def sign_in_from_the_tools_menu(page, mcp, server_name: str) -> None:
    """Press the server under Tools and follow the sign-in back to the chat."""
    sign_ins_before = len(sign_ins(mcp))
    open_tools_menu(page).get_by_role("button", name=server_name).click()
    page.wait_for_url(
        lambda url: "error=" in url or len(sign_ins(mcp)) > sign_ins_before,
        timeout=30_000,
    )
    menu = open_tools_menu(page)
    assert "error=" not in page.url, page.url
    # a server that needs a sign-in carries no pressed state at all
    expect(menu.get_by_role("button", name=server_name)).to_have_attribute("aria-pressed", ANY)
    page.keyboard.press("Escape")


def pick_connected_server(page, server_name: str) -> None:
    menu = open_tools_menu(page)
    server = menu.get_by_role("button", name=server_name)
    server.click()
    expect(server).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("Escape")


def echo_through(page, upstream, tool_name: str, phrase: str) -> None:
    question = f"say it back {secrets.token_hex(3)}"
    upstream.queue(
        reply.tool_call(tool_name, {"text": phrase}, match=reply.answering(question)),
        reply.text("Said it back.", match=reply.answering(question)),
    )
    send(page, question)
    expect_reply(page, "Said it back.")
    expect(tool_output(page, tool_name)).to_contain_text(phrase)


def test_the_admin_registers_the_server_and_a_person_signs_in_from_the_chat(
    page_for, make_user, upstream, mcp
):
    token = secrets.token_hex(3)
    name = f"Pilot Desk {token}"
    admin_page = page_for(make_user(role="admin"))
    form = open_add_connection(admin_page)
    switch_to_mcp(form)
    form.get_by_label("Name", exact=True).fill(name)
    form.get_by_label("URL", exact=True).fill(mcp.url)
    choose_auth(form, "OAuth 2.1")
    expect(form.get_by_text("Not Registered")).to_be_visible()

    form.get_by_role("button", name=CHECK).click()
    expect(admin_page.get_by_text("OAuth discovery successful")).to_be_visible()
    form.get_by_role("button", name="Register Client").click()
    expect(form.get_by_text("Registered", exact=True)).to_be_visible()
    make_public(admin_page, form)
    save(admin_page, form)
    assert len(mcp.auth_server.requests_to("/register")) == 1

    page = page_for(make_user())
    sign_in_from_the_tools_menu(page, mcp, name)
    pick_connected_server(page, name)
    echo_through(page, upstream, f"pilot-desk-{token}_echo", "the pilot boards at dawn")

    assert mcp.presented[-1] == mcp.auth_server.issued[-1]["access_token"]


def test_a_new_chat_reuses_the_token_without_signing_in_again(
    page_for, admin, make_user, upstream, mcp
):
    server_id = f"pilot_{secrets.token_hex(3)}"
    register_oauth_mcp(admin, mcp, server_id, [EVERYONE])
    page = page_for(make_user())
    sign_in_from_the_tools_menu(page, mcp, server_id)
    [first_token] = [answer["access_token"] for answer in mcp.auth_server.issued]

    pick_connected_server(page, server_id)
    echo_through(page, upstream, f"{server_id}_echo", "first tide")
    pick_connected_server(page, server_id)
    echo_through(page, upstream, f"{server_id}_echo", "second tide")

    assert len(sign_ins(mcp)) == 1
    assert len(mcp.auth_server.issued) == 1
    assert mcp.presented[-1] == first_token


def test_disconnecting_signs_out_and_the_next_use_signs_in_again(
    page_for, admin, make_user, upstream, mcp
):
    server_id = f"pilot_{secrets.token_hex(3)}"
    register_oauth_mcp(admin, mcp, server_id, [EVERYONE])
    page = page_for(make_user())
    sign_in_from_the_tools_menu(page, mcp, server_id)

    menu = open_tools_menu(page)
    server = menu.get_by_role("button", name=server_id)
    tooltip_button(server, "Disconnect OAuth").click()
    expect(page.get_by_text("OAuth session disconnected")).to_be_visible()
    expect(server).not_to_have_attribute("aria-pressed", ANY)
    page.keyboard.press("Escape")

    sign_in_from_the_tools_menu(page, mcp, server_id)
    pick_connected_server(page, server_id)
    echo_through(page, upstream, f"{server_id}_echo", "after the second sign-in")

    assert len(sign_ins(mcp)) == 2
    assert mcp.presented[-1] == mcp.auth_server.issued[-1]["access_token"]
    assert mcp.presented[-1] != mcp.auth_server.issued[0]["access_token"]


def test_the_check_is_named_for_oauth_discovery_once_oauth_is_chosen(page_for, make_user, mcp):
    form = open_add_connection(page_for(make_user(role="admin")))
    switch_to_mcp(form)
    form.get_by_label("URL", exact=True).fill(mcp.url)

    choose_auth(form, "OAuth 2.1")

    expect(form.get_by_text("Not Registered")).to_be_visible()
    expect(form.get_by_role("button", name="Check OAuth Discovery")).to_be_visible()
    expect(form.get_by_role("button", name="Verify Connection")).to_have_count(0)
