"""Journey: a person adds their own OpenAPI tool server in Settings, once the admin allows it.

Admin Settings > Integrations holds the Direct Integrations switch and the default permissions
hold Direct Tool Servers. With both on, a user's Settings has the Integrations tab; with either
off, it has none, while an admin keeps it. A tool server added there, read by the browser from
its `openapi.json`, is listed among the chat's tools, and picked there it offers its operations
to the model. When the model calls one, the person's browser makes the call itself, with the key
typed in the dialog if there is one, and the reply opens to show what the server answered. The
dialog's Verify Connection says "Connection successful" for a server it reaches and "Connection
failed" for one it does not.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Integrations form sends the
stored switch back and whose Default permissions dialog saves the permissions it opened with,
the tab tests fail; in one that leaves the person's tool servers out of the chat request, the
tool server test fails; in one whose tool call handler never makes the call, or makes it without
the key, the call tests fail; in one whose Verify Connection reports success whatever the server
did, the failure case fails, and in one that reports failure always, the success case fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.instance import free_port
from harness.listener import ReceivedRequest, json_answer
from harness.openapi_server import authorization
from utils.chat_ui import chat_input, conversation, expect_reply, send
from utils.tool_servers import choose_auth, verify

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CONNECTIONS_CONFIG = ("/api/v1/configs/connections", "/api/v1/configs/connections")
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type, x-session-id",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}
TIDE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Tide table", "version": "1.0.0", "description": "Tides of the harbour"},
    "paths": {
        "/tides": {
            "get": {
                "operationId": "get_tides",
                "summary": "Today's high and low tides at the harbour",
                "responses": {"200": {"description": "The tides"}},
            }
        }
    },
}


def save_connections(admin, **changes) -> None:
    with admin.client() as client:
        current = client.get(CONNECTIONS_CONFIG[0]).json()
        saved = client.post(CONNECTIONS_CONFIG[1], json={**current, **changes})
    saved.raise_for_status()


def save_tool_server_permission(admin, allowed: bool) -> None:
    with admin.client() as client:
        current = client.get("/api/v1/users/default/permissions").json()
        current["features"]["direct_tool_servers"] = allowed
        saved = client.post("/api/v1/users/default/permissions", json=current)
    saved.raise_for_status()


@pytest.fixture
def allowed(admin, preserve):
    """Direct Integrations on for the instance and Direct Tool Servers on for every account."""
    preserve(CONNECTIONS_CONFIG, "permissions")
    save_connections(admin, ENABLE_DIRECT_INTEGRATIONS=True)
    save_tool_server_permission(admin, allowed=True)


def settings_tabs(page: Page) -> Locator:
    page.goto("/?settings=general")
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("tab").first).to_be_visible()
    return dialog


def integrations_tab(page: Page) -> Locator:
    # an admin also has the admin Integrations tab, listed after the personal ones
    return settings_tabs(page).get_by_role("tab", name="Integrations", exact=True).first


def test_the_integrations_tab_is_offered_once_both_are_on(page_for, make_user, allowed):
    expect(integrations_tab(page_for(make_user()))).to_be_visible()


def test_the_admins_direct_integrations_switch_takes_the_tab_away(
    page_for, admin, make_user, allowed
):
    admin_page = page_for(admin)
    admin_page.goto("/admin/settings/integrations")
    switch = admin_page.get_by_role("dialog").get_by_role("switch", name="Direct Integrations")
    expect(switch).to_be_checked()
    with admin_page.expect_response(lambda response: "/configs/connections" in response.url):
        switch.click()
    expect(switch).not_to_be_checked()

    expect(integrations_tab(page_for(make_user()))).to_have_count(0)


def test_the_default_permission_takes_the_tab_away_from_users_only(
    page_for, admin, make_user, allowed
):
    admin_page = page_for(admin)
    admin_page.goto("/admin/users/groups")
    admin_page.get_by_role("button", name="Default permissions").click()
    dialog = admin_page.get_by_role("dialog")
    switch = dialog.get_by_role("switch", name="Direct Tool Servers", exact=True)
    expect(switch).to_be_checked()
    switch.click()
    dialog.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Default permissions updated successfully")).to_be_visible()

    expect(integrations_tab(page_for(make_user()))).to_have_count(0)
    expect(integrations_tab(page_for(make_user(role="admin")))).to_be_visible()


def answer_tides(_request: ReceivedRequest):
    status, headers, body = json_answer({"high": "12:04", "low": "18:31"})
    return status, {**headers, **CORS}, body


def serve_tide_table(listener) -> None:
    def spec(_request: ReceivedRequest):
        status, headers, body = json_answer(TIDE_SPEC)
        return status, {**headers, **CORS}, body

    listener.route("GET", "/openapi.json", spec)
    listener.route("OPTIONS", "/*", lambda _request: (204, CORS, b""))


def open_add_connection(page: Page) -> Locator:
    integrations_tab(page).click()
    page.get_by_label("External Tool Servers").get_by_role("button", name="Add Connection").click()
    modal = page.get_by_role("dialog").filter(has=page.get_by_placeholder("API Base URL"))
    expect(modal).to_be_visible()
    return modal


def add_tool_server(page: Page, url: str, key: str | None = None) -> None:
    modal = open_add_connection(page)
    modal.get_by_placeholder("API Base URL").fill(url)
    if key:
        choose_auth(modal, "Bearer")
        modal.get_by_placeholder("API Key").fill(key)
    modal.get_by_role("button", name="Save", exact=True).click()
    expect(page.locator("#tab-tools").get_by_text(url)).to_be_visible()


def pick_added_server(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    page.get_by_role("button", name="Tide table").click()
    page.keyboard.press("Escape")


def test_an_added_tool_server_offers_its_operation_to_the_model(
    page_for, make_user, upstream, listener, allowed
):
    serve_tide_table(listener)
    page = page_for(make_user())
    add_tool_server(page, listener.base_url)

    question = "When is high tide today?"
    upstream.queue(reply.text("At noon.", match=reply.answering(question)))
    pick_added_server(page)
    send(page, question)
    expect_reply(page, "At noon.")

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    offered = {tool["function"]["name"] for tool in sent.get("tools") or []}
    assert "get_tides" in offered, sorted(offered)


@pytest.mark.parametrize("key", [None, "harbour-master-key"], ids=["no-key", "bearer-key"])
def test_the_browser_calls_the_added_tool_server_for_the_model(
    page_for, make_user, upstream, listener, allowed, key
):
    serve_tide_table(listener)
    listener.route("GET", "/tides", answer_tides)
    page = page_for(make_user())
    add_tool_server(page, listener.base_url, key)

    question = "When is low tide today?"
    upstream.queue(
        reply.tool_call("get_tides", {}, match=reply.answering(question)),
        reply.text("Low water at dusk.", match=reply.answering(question)),
    )
    pick_added_server(page)
    send(page, question)
    expect_reply(page, "Low water at dusk.")

    conversation(page).get_by_text("View Result from get_tides").click()
    expect(conversation(page).get_by_text("18:31")).to_be_visible()
    [call] = [entry for entry in listener.requests_to("/tides") if entry.method == "GET"]
    assert authorization(call) == (f"Bearer {key}" if key else None)


def test_verify_connection_tells_a_reachable_server_from_an_unreachable_one(
    page_for, make_user, listener, allowed
):
    serve_tide_table(listener)
    modal = open_add_connection(page_for(make_user()))
    page = modal.page

    modal.get_by_placeholder("API Base URL").fill(f"http://127.0.0.1:{free_port()}")
    verify(modal)
    expect(page.get_by_text("Connection failed")).to_be_visible()

    modal.get_by_placeholder("API Base URL").fill(listener.base_url)
    verify(modal)
    expect(page.get_by_text("Connection successful")).to_be_visible()
