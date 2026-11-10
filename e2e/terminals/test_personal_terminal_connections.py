"""Journey: a person adds, switches, edits and deletes terminals of their own.

With the admin's Direct Integrations on and the Direct Tool Servers permission, a person's
Settings > Integrations has an Open Terminal section. Its Add and Edit Terminal Connection
dialog is the admin's dialog (rebuilt in dev 6f2fc7bc7) without what only applies to a
connection the instance calls: no ID, no Access, no Forward cookies, and only Bearer or None
for Auth. Such a terminal is called by the person's browser, so Verify Connection and the
check before Save are made by the browser itself, and a Terminals service is turned away
("Add Terminals as an administrator-managed connection."). The terminals are stored in the
person's own settings. One of them is active at a time: switching one on in the list switches
the others off. A saved terminal is offered under Direct in the chat input's Terminal menu
without a reload, and an active one that does not answer when the page loads is named in a
notice.

The terminal is the fake from harness/terminal_server.py, answering a page of another origin.
e2e/config/test_settings_tool_servers.py holds who gets the Integrations tab, and
e2e/tools/test_terminal_token_normalization.py a key typed with a space.

The list in Settings shows what the page holds, and the chat page switches every terminal of
the person's own off when it loads unless it is the picked one, so the tests of the list start
from terminals stored switched off.

Discriminates: passes on the dev c15f28649 build. Each test fails in a frontend build with a
change of its own, four builds with several changes each. The first test: the admin's ID
field shown in a person's dialog; the check sent to the instance; Session and OAuth on offer.
The refusal test: a Terminals service accepted; a failed check not shown; a failed check not
stopping the save. The switch test: the others left on when one is switched on. The edit
test: Delete not asking; the edit not stored; Delete removing nothing. The notice test: no
notice for a terminal that does not answer.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.instance import free_port
from harness.listener import json_answer
from harness.terminal_server import FakeTerminalServer, serving_terminal
from utils.terminal_connections import (
    PERSONAL_SETTINGS,
    add_terminal,
    delete_terminal,
    edit_terminal,
    fill_terminal,
    leave_settings,
    personal_terminal_section,
    save_terminal,
    show_advanced,
    terminal_problem,
    verify_terminal,
)
from utils.tool_servers import connection_row
from utils.user_terminals import listed_terminal, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KEY = "harbour-key"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
NO_ANSWER = "Failed to connect to the terminal server"
NOT_FOR_TERMINALS = "Add Terminals as an administrator-managed connection."
EMPTY = "No terminal connections configured."


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    """A plain Open Terminal that answers the browser from another origin."""
    with serving_terminal() as server:
        server.cors = True
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {}}))
        yield server


@pytest.fixture
def terminals_service() -> Iterator[FakeTerminalServer]:
    """A Terminals service that answers the browser from another origin: it lists templates."""
    with serving_terminal() as server:
        server.cors = True
        server.route("GET", "/api/v1/terminals/templates", json_answer([]))
        yield server


@pytest.fixture
def person(admin, make_user, direct_integrations, preserve) -> Actor:
    """An account that may add terminals of its own."""
    preserve("permissions")
    direct_integrations(True)
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["direct_tool_servers"] = True
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    return make_user()


def _tagged(name: str) -> str:
    return f"{name} {uuid.uuid4().hex[:6]}"


def _own_terminal(server_url: str, name: str, enabled: bool) -> dict:
    """A terminal as a person's dialog stores it."""
    return {
        "url": server_url,
        "key": KEY,
        "name": name,
        "path": "/openapi.json",
        "auth_type": "bearer",
        "enabled": enabled,
        "forward_cookies": False,
        "server_type": "terminal",
        "config": {
            "chat_uploads": "filesystem",
            "working_directory_context": True,
            "user_shell_tools": "auto",
        },
    }


def _store(account: Actor, *terminals: dict) -> None:
    with account.client() as client:
        saved = client.post(
            PERSONAL_SETTINGS,
            json={"ui": {"showChangelog": False, "terminalServers": list(terminals)}},
        )
    assert saved.status_code == 200, saved.text


def _stored(account: Actor) -> list[dict]:
    with account.client() as client:
        answer = client.get("/api/v1/users/user/settings")
    assert answer.status_code == 200, answer.text
    return answer.json()["ui"].get("terminalServers") or []


def _switched_on(account: Actor) -> list[str]:
    return [entry["name"] for entry in _stored(account) if entry["enabled"]]


def _page_with(page_for, account: Actor, *terminals: dict) -> Page:
    """A page of `account` that loaded with those terminals in the account's settings."""
    page = page_for(account)
    _store(account, *terminals)
    page.reload()
    return page


def _asked_by_the_browser(server: FakeTerminalServer, page: Page) -> list[str]:
    """The paths the page itself asked the terminal for."""
    origin = page.evaluate("location.origin")
    return [
        request.path
        for request in server.received
        if request.method == "GET" and request.headers.get("origin") == origin
    ]


def _switch(section: Locator, name: str) -> Locator:
    return connection_row(section, name).get_by_role("switch")


def _settings_stored(page: Page):
    return page.expect_response(
        lambda response: (
            response.request.method == "POST" and response.url.endswith(PERSONAL_SETTINGS)
        )
    )


def test_a_persons_own_terminal_is_checked_by_their_browser_and_offered_in_the_chat_once_saved(
    page_for, person, terminal
):
    page = page_for(person)
    name = _tagged("My laptop")
    checks_by_the_instance = []
    page.on(
        "request",
        lambda request: (
            checks_by_the_instance.append(request.url)
            if request.url.endswith("/terminal_servers/verify")
            else None
        ),
    )
    section = personal_terminal_section(page)
    expect(section.get_by_text(EMPTY)).to_be_visible()
    expect(section.get_by_role("link", name="Learn more about Open Terminal")).to_be_visible()

    dialog = add_terminal(section)
    expect(dialog.get_by_label("Name", exact=True)).to_be_visible()
    expect(dialog.get_by_label("ID", exact=False)).to_have_count(0)
    expect(dialog.get_by_role("button", name="Access", exact=True)).to_have_count(0)
    expect(dialog.get_by_label("Auth", exact=True).locator("option")).to_have_text(
        ["Bearer", "None"]
    )
    show_advanced(dialog)
    expect(dialog.get_by_label("OpenAPI Spec", exact=True)).to_be_visible()
    expect(dialog.get_by_role("switch", name="Forward cookies")).to_have_count(0)

    fill_terminal(dialog, name, terminal.base_url, key=KEY)
    verify_terminal(dialog)
    expect(terminal_problem(dialog)).to_have_count(0)
    asked = _asked_by_the_browser(terminal, page)
    assert asked == ["/api/v1/terminals/templates", "/api/config"], (
        f"the check of a person's own terminal reached it from their browser as {asked}"
    )
    assert not checks_by_the_instance, "the instance was asked to check a person's own terminal"

    save_terminal(page, dialog, saved_to=PERSONAL_SETTINGS)

    expect(connection_row(section, name)).to_be_visible()
    expect(_switch(section, name)).to_be_checked()
    expect(section.get_by_text(EMPTY)).to_have_count(0)
    assert _stored(person) == [_own_terminal(terminal.base_url, name, enabled=True)], (
        f"the terminal was stored in the person's settings as {_stored(person)}"
    )
    leave_settings(page)
    menu = open_terminal_menu(page)
    expect(menu.get_by_text("Direct", exact=True)).to_be_visible()
    expect(listed_terminal(page, name)).to_be_visible()


def test_a_terminals_service_or_an_address_nothing_answers_at_is_not_added_as_a_persons_own(
    page_for, person, terminal, terminals_service
):
    page = page_for(person)
    name = _tagged("My laptop")
    section = personal_terminal_section(page)
    dialog = add_terminal(section)
    save = dialog.get_by_role("button", name="Save", exact=True)

    fill_terminal(dialog, name, terminals_service.base_url, key=KEY)
    verify_terminal(dialog)
    expect(terminal_problem(dialog)).to_have_text(NOT_FOR_TERMINALS)
    assert "/api/v1/terminals/templates" in _asked_by_the_browser(terminals_service, page), (
        "the Terminals service was not asked by the browser"
    )
    save.click()
    expect(terminal_problem(dialog)).to_have_text(NOT_FOR_TERMINALS)
    expect(save).to_be_enabled()
    assert _stored(person) == [], "a Terminals service was stored as a person's own terminal"

    dialog.get_by_label("URL", exact=True).fill(f"http://127.0.0.1:{free_port()}")
    save.click()
    expect(terminal_problem(dialog)).to_have_text(NO_ANSWER)
    expect(save).to_be_enabled()
    assert _stored(person) == [], "a terminal nothing answers at was stored"

    dialog.get_by_label("URL", exact=True).fill(terminal.base_url)
    save_terminal(page, dialog, saved_to=PERSONAL_SETTINGS)
    assert [entry["name"] for entry in _stored(person)] == [name], _stored(person)


def test_switching_one_of_a_persons_terminals_on_switches_the_others_off(
    page_for, person, terminal
):
    first, second, third = _tagged("My laptop"), _tagged("Workshop"), _tagged("Shed")
    page = _page_with(
        page_for,
        person,
        _own_terminal(terminal.base_url, first, enabled=False),
        _own_terminal(f"{terminal.base_url}/workshop", second, enabled=False),
        _own_terminal(f"{terminal.base_url}/shed", third, enabled=False),
    )
    section = personal_terminal_section(page)

    def switch_over(name: str) -> None:
        with _settings_stored(page) as stored:
            _switch(section, name).click()
        assert stored.value.ok, stored.value.text()

    def shown_on() -> list[str]:
        return [name for name in (first, second, third) if _switch(section, name).is_checked()]

    expect(_switch(section, first)).not_to_be_checked()
    switch_over(first)
    expect(_switch(section, first)).to_be_checked()
    assert _switched_on(person) == [first], f"switched on are {_switched_on(person)}"

    switch_over(second)
    expect(_switch(section, second)).to_be_checked()
    expect(_switch(section, first)).not_to_be_checked()
    assert shown_on() == [second], f"the list shows as switched on: {shown_on()}"
    assert _switched_on(person) == [second], f"switched on are {_switched_on(person)}"

    switch_over(second)
    expect(_switch(section, second)).not_to_be_checked()
    assert _switched_on(person) == [], f"switched on are {_switched_on(person)}"
    assert [entry["name"] for entry in _stored(person)] == [first, second, third], _stored(person)


def test_a_persons_terminal_is_edited_and_deleted_from_its_row(page_for, person, terminal):
    name, renamed, other = _tagged("My laptop"), _tagged("Old laptop"), _tagged("Workshop")
    page = _page_with(
        page_for,
        person,
        _own_terminal(terminal.base_url, name, enabled=False),
        _own_terminal(f"{terminal.base_url}/workshop", other, enabled=False),
    )
    section = personal_terminal_section(page)

    dialog = edit_terminal(section, name)
    expect(dialog.get_by_label("URL", exact=True)).to_have_value(terminal.base_url)
    expect(dialog.get_by_placeholder("API Key")).to_have_value(KEY)
    dialog.get_by_label("Name", exact=True).fill(renamed)
    save_terminal(page, dialog, saved_to=PERSONAL_SETTINGS)
    expect(connection_row(section, renamed)).to_be_visible()
    expect(_switch(section, renamed)).not_to_be_checked()
    assert _stored(person) == [
        _own_terminal(terminal.base_url, renamed, enabled=False),
        _own_terminal(f"{terminal.base_url}/workshop", other, enabled=False),
    ], f"after the rename the person's terminals are {_stored(person)}"

    dialog = edit_terminal(section, renamed)
    delete_terminal(page, dialog, confirm=False)
    expect(dialog).to_be_visible()
    assert len(_stored(person)) == 2, f"an unanswered Delete left {_stored(person)}"
    with _settings_stored(page) as stored:
        delete_terminal(page, dialog)
    assert stored.value.ok, stored.value.text()
    expect(dialog).to_be_hidden()
    expect(connection_row(section, other)).to_be_visible()
    expect(section.get_by_text(renamed)).to_have_count(0)
    assert [entry["name"] for entry in _stored(person)] == [other], _stored(person)

    leave_settings(page)
    open_terminal_menu(page)
    expect(listed_terminal(page, other)).to_be_visible()
    expect(listed_terminal(page, renamed)).to_have_count(0)


def test_an_active_terminal_that_does_not_answer_is_named_in_a_notice_when_the_page_loads(
    page_for, person, terminal
):
    name = _tagged("My laptop")
    gone = f"http://127.0.0.1:{free_port()}"
    page = page_for(person)
    _store(person, _own_terminal(terminal.base_url, name, enabled=True))
    with page.expect_response(f"{terminal.base_url}/openapi.json") as read:
        page.reload()
    assert read.value.ok, f"the page did not read its terminal's tool list: {read.value.status}"
    open_terminal_menu(page)
    expect(listed_terminal(page, name)).to_be_visible()
    expect(page.get_by_text("terminal server", exact=False)).to_have_count(0)

    _store(person, _own_terminal(gone, name, enabled=True))
    page.reload()

    expect(page.get_by_text(f"Failed to connect to {gone} terminal server")).to_be_visible()
