"""Journey: an admin adds, checks, edits and deletes a terminal connection.

Admin Settings > Integrations > Terminal lists the connections and opens the Add and the Edit
Terminal Connection dialog, rebuilt in dev 6f2fc7bc7. A connection needs a name and an address;
its ID is made up when none is typed and cannot be changed later. Verify Connection asks the
instance what answers at the address: a plain Open Terminal, a Terminals service (the dialog
then says that terminals are created from the terminal selector) or nothing (the dialog says
so). Save runs the same check first when the kind is not known yet and stores nothing when it
fails. The address is stored without a trailing slash, the key as typed (it is trimmed where it
is sent), the auth choice as picked, except that a Terminals service is always stored with
Bearer. A saved connection is listed at once and offered in the chat input's Terminal menu
without a reload. The Edit dialog starts from what is stored each time it opens and keeps the
connection's switch and the entries it does not show. Access gives the terminal to people and
groups, Advanced sets where the tool list is read from and whether cookies are forwarded, and
Delete asks first. A save the instance refuses is reported as "Failed to save terminal servers"
and stores nothing.

The terminal is the fake from harness/terminal_server.py; a Terminals service is the same fake
answering the template list. e2e/admin/test_terminal_connection_settings.py holds the two
Advanced settings that shape the model's tools, the untouched save and the red test for Auth
None (the API Key field must be filled in whatever Auth says, so every test here types a key).

Discriminates: passes on the dev c15f28649 build. Each test fails in a frontend build with a
change of its own, four builds with several changes each. The new connection test: Name not
required; no ID made up; a new connection started on Default uploads; the list behind the
Terminal menu not refreshed after a save (also the delete test). The verify test: the kind
kept when the address changes; a failed check not shown (also the save check test). The save
check test: a trailing slash kept; a failed check not stopping the save. The Terminals test:
the picked auth stored. The auth test: the key trimmed when stored; Session and OAuth taken
off the select (also the Terminals test); a typed ID replaced by a made-up one. The edit test:
the dialog started from its last state; the switch reset on Save; a stored connection without
an upload setting shown as Filesystem; the ID of a stored connection not locked. The refused
save test: the failure not reported. The access test: the access list not stored. The
Advanced test: Forward cookies not stored; the path not stored; the preview joining address
and path whatever the path is. The delete test: Delete not asking. The list test: the address
and New Terminal fallbacks dropped.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.instance import free_port
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    serving_terminal,
)
from utils.access_control import access_row
from utils.member_picker import pick_member
from utils.terminal_connections import (
    ADMIN_INTEGRATIONS,
    add_terminal,
    choose_terminal_auth,
    delete_terminal,
    edit_terminal,
    field_the_browser_refuses,
    fill_terminal,
    leave_settings,
    save_terminal,
    show_advanced,
    shown_admin_terminal_section,
    terminal_problem,
    verify_terminal,
)
from utils.tool_servers import connection_row
from utils.user_terminals import listed_terminal, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

KEY = "harbour-key"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
NO_ANSWER = "Failed to connect to the terminal server"
TERMINALS_HINT = "Create and configure terminals from the terminal selector."
AUTH_HINTS = {
    "Bearer": None,
    "None": "No authentication",
    "Session": "Forwards system user session credentials to authenticate",
    "OAuth": "Forwards system user OAuth access token to authenticate",
}


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    """A plain Open Terminal, as far as the connection dialog asks."""
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {}}))
        yield server


@pytest.fixture
def terminals_service() -> Iterator[FakeTerminalServer]:
    """A Terminals service, as far as the connection dialog asks: it lists templates."""
    with serving_terminal() as server:
        server.route("GET", "/api/v1/terminals/templates", json_answer([]))
        yield server


@pytest.fixture
def admin_account(make_user, preserve):
    preserve(TERMINAL_SERVERS_CONFIG)
    return make_user(role="admin")


@pytest.fixture
def open_terminals(page_for, admin_account):
    """Opens the admin's page on Admin Settings > Integrations; returns it and the Terminal
    section. Call it once the connections the test starts from are stored."""

    def open_them() -> tuple[Page, Locator]:
        page = page_for(admin_account, path=ADMIN_INTEGRATIONS)
        return page, shown_admin_terminal_section(page)

    return open_them


def _tagged(name: str) -> str:
    return f"{name} {uuid.uuid4().hex[:6]}"


def _stored_connections(admin_account) -> list[dict]:
    with admin_account.client() as client:
        answer = client.get(TERMINAL_SERVERS_CONFIG[0])
    assert answer.status_code == 200, answer.text
    return answer.json()["TERMINAL_SERVER_CONNECTIONS"]


def _stored(admin_account, name: str) -> dict | None:
    return next(
        (entry for entry in _stored_connections(admin_account) if entry.get("name") == name),
        None,
    )


def _store(admin_account, *connections: dict) -> None:
    with admin_account.client() as client:
        configure_terminals(client, *connections)


def _terminal_menu_of(account) -> list[str]:
    with account.client() as client:
        answer = client.get("/api/v1/terminals/")
    assert answer.status_code == 200, answer.text
    return [entry["name"] for entry in answer.json()]


def _unreachable_address() -> str:
    return f"http://127.0.0.1:{free_port()}"


def _row_switch(section: Locator, name: str) -> Locator:
    return connection_row(section, name).get_by_role("switch")


def test_a_new_connection_needs_a_name_and_an_address_and_is_offered_in_the_chat_once_saved(
    open_terminals, admin_account, terminal
):
    page, section = open_terminals()
    name = _tagged("Build box")
    dialog = add_terminal(section)
    save = dialog.get_by_role("button", name="Save", exact=True)
    check = dialog.get_by_role("button", name="Verify Connection", exact=True)
    expect(dialog.get_by_label("Chat Uploads", exact=True).locator("option:checked")).to_have_text(
        "Filesystem"
    )
    expect(dialog.get_by_role("button", name="Delete", exact=True)).to_have_count(0)
    expect(save).to_be_disabled()
    expect(check).to_be_disabled()

    fill_terminal(dialog, "", terminal.base_url, key=KEY)
    expect(save).to_be_enabled()
    expect(check).to_be_enabled()
    refused = field_the_browser_refuses(dialog)
    assert refused and refused.startswith('"My Terminal"'), (
        f"a connection without a name is not held back at the Name field: {refused}"
    )
    save.click()
    expect(dialog).to_be_visible()
    assert _stored_connections(admin_account) == [], "a connection without a name was stored"

    dialog.get_by_label("Name", exact=True).fill(f"  {name} ")
    save_terminal(page, dialog)

    expect(page.get_by_text("Terminal servers saved")).to_be_visible()
    expect(connection_row(page.get_by_role("dialog"), name)).to_be_visible()
    [stored] = _stored_connections(admin_account)
    made_up_id = stored.pop("id")
    assert re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", made_up_id), (
        f"a connection added without an ID was stored with the ID {made_up_id!r}"
    )
    assert stored == {
        "name": name,
        "enabled": True,
        "url": terminal.base_url,
        "path": "/openapi.json",
        "key": KEY,
        "auth_type": "bearer",
        "forward_cookies": False,
        "server_type": "terminal",
        "config": {
            "access_grants": [],
            "chat_uploads": "filesystem",
            "working_directory_context": True,
            "user_shell_tools": "auto",
        },
    }, f"the new connection was stored as {stored}"

    leave_settings(page)
    open_terminal_menu(page)
    expect(listed_terminal(page, name)).to_be_visible()


def test_verify_connection_tells_a_terminal_from_terminals_and_from_an_address_nothing_answers_at(
    open_terminals, admin_account, terminal, terminals_service
):
    page, section = open_terminals()
    dialog = add_terminal(section)

    fill_terminal(dialog, _tagged("Build box"), _unreachable_address(), key=KEY)
    verify_terminal(dialog)
    expect(terminal_problem(dialog)).to_have_text(NO_ANSWER)

    dialog.get_by_label("URL", exact=True).fill(terminal.base_url)
    verify_terminal(dialog)
    expect(terminal_problem(dialog)).to_have_count(0)
    expect(dialog.get_by_text(TERMINALS_HINT)).to_have_count(0)
    asked = [(request.method, request.path) for request in terminal.received]
    assert asked == [("GET", "/api/v1/terminals/templates"), ("GET", "/api/config")], (
        f"the check asked the terminal for {asked}"
    )

    dialog.get_by_label("URL", exact=True).fill(terminals_service.base_url)
    verify_terminal(dialog)
    expect(dialog.get_by_text(TERMINALS_HINT)).to_be_visible()
    expect(terminal_problem(dialog)).to_have_count(0)

    # another address is another server: what the last one was no longer counts
    dialog.get_by_label("URL", exact=True).fill(terminal.base_url)
    expect(dialog.get_by_text(TERMINALS_HINT)).to_have_count(0)


def test_save_checks_the_address_first_and_stores_nothing_while_nothing_answers_there(
    open_terminals, admin_account, terminal
):
    page, section = open_terminals()
    name = _tagged("Build box")
    dialog = add_terminal(section)
    fill_terminal(dialog, name, _unreachable_address(), key=KEY)
    save = dialog.get_by_role("button", name="Save", exact=True)

    save.click()

    expect(terminal_problem(dialog)).to_have_text(NO_ANSWER)
    expect(save).to_be_enabled()
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)
    assert _stored_connections(admin_account) == [], "a connection nothing answers at was stored"

    dialog.get_by_label("URL", exact=True).fill(f"{terminal.base_url}/")
    save_terminal(page, dialog)
    stored = _stored(admin_account, name)
    assert stored and (stored["server_type"], stored["url"]) == ("terminal", terminal.base_url), (
        f"the connection saved once its terminal answered was stored as {stored}"
    )


def test_a_terminals_service_is_stored_as_one_and_with_bearer_whatever_auth_was_picked(
    open_terminals, admin_account, terminals_service
):
    page, section = open_terminals()
    name = _tagged("Terminals")
    dialog = add_terminal(section)
    choose_terminal_auth(dialog, "Session")
    fill_terminal(dialog, name, terminals_service.base_url, key=KEY)

    save_terminal(page, dialog)

    stored = _stored(admin_account, name)
    assert stored and (stored["server_type"], stored["auth_type"], stored["key"]) == (
        "orchestrator",
        "bearer",
        KEY,
    ), f"the Terminals service was stored as {stored}"
    reopened = edit_terminal(section, name)
    expect(reopened.get_by_text(TERMINALS_HINT)).to_be_visible()
    expect(reopened.get_by_label("Auth", exact=True).locator("option:checked")).to_have_text(
        "Bearer"
    )


def test_each_auth_choice_says_what_it_sends_and_auth_key_and_a_typed_id_are_stored_as_entered(
    open_terminals, admin_account, terminal
):
    page, section = open_terminals()
    name = _tagged("Build box")
    typed_id = f"build-box-{uuid.uuid4().hex[:6]}"
    dialog = add_terminal(section)
    auth = dialog.get_by_label("Auth", exact=True)
    expect(auth.locator("option")).to_have_text(list(AUTH_HINTS))
    expect(auth.locator("option:checked")).to_have_text("Bearer")
    for choice, hint in AUTH_HINTS.items():
        choose_terminal_auth(dialog, choice)
        for text in filter(None, AUTH_HINTS.values()):
            expect(dialog.get_by_text(text, exact=True)).to_have_count(1 if text == hint else 0)

    choose_terminal_auth(dialog, "Session")
    fill_terminal(dialog, name, terminal.base_url, key=f" {KEY} ")
    expect(dialog.get_by_label("ID", exact=False)).to_be_enabled()
    dialog.get_by_label("ID", exact=False).fill(typed_id)
    save_terminal(page, dialog)

    stored = _stored(admin_account, name)
    entered = ("session", f" {KEY} ", typed_id)
    assert stored and (stored["auth_type"], stored["key"], stored["id"]) == entered, (
        f"Session, a key typed between spaces and the ID {typed_id} were stored as {stored}"
    )
    first_asked = {}
    for request in terminal.received:
        first_asked.setdefault(request.path, request.headers.get("authorization"))
    checked_and_read = ("/api/v1/terminals/templates", "/api/config", "/openapi.json")
    assert {path: first_asked.get(path) for path in checked_and_read} == dict.fromkeys(
        checked_and_read, f"Bearer {KEY}"
    ), f"the check before the save and the read of the tool list sent the key as {first_asked}"
    reopened = edit_terminal(section, name)
    expect(reopened.get_by_label("Auth", exact=True).locator("option:checked")).to_have_text(
        "Session"
    )
    expect(reopened.get_by_text(AUTH_HINTS["Session"])).to_be_visible()
    expect(reopened.get_by_placeholder("API Key")).to_have_value(f" {KEY} ")


def test_the_edit_dialog_starts_from_what_is_stored_and_save_keeps_what_it_does_not_show(
    open_terminals, admin_account, terminal
):
    name, renamed = _tagged("Build box"), _tagged("Dry dock")
    connection_id = f"build-box-{uuid.uuid4().hex[:6]}"
    # switched off, without an upload setting and with entries the dialog does not know
    stored = terminal.connection(
        id=connection_id,
        name=name,
        enabled=False,
        key=KEY,
        auth_type="bearer",
        label="set over the API",
        config={"access_grants": [], "note": "set over the API"},
    )
    _store(admin_account, stored)
    before = _stored_connections(admin_account)
    page, section = open_terminals()

    dialog = edit_terminal(section, name)
    expect(dialog.get_by_label("ID", exact=False)).to_have_value(connection_id)
    expect(dialog.get_by_label("ID", exact=False)).to_be_disabled()
    expect(dialog.get_by_label("URL", exact=True)).to_have_value(terminal.base_url)
    expect(dialog.get_by_placeholder("API Key")).to_have_value(KEY)
    expect(dialog.get_by_label("Chat Uploads", exact=True).locator("option:checked")).to_have_text(
        "Default"
    )
    expect(dialog.get_by_role("button", name="Delete", exact=True)).to_be_visible()

    dialog.get_by_label("Name", exact=True).fill("Never saved")
    dialog.get_by_label("Chat Uploads", exact=True).select_option(label="Filesystem")
    dialog.get_by_role("button", name="Close", exact=True).click()
    expect(dialog).to_be_hidden()
    after = _stored_connections(admin_account)
    assert after == before, f"closing the dialog changed the stored connection to {after}"

    dialog = edit_terminal(section, name)
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)
    expect(dialog.get_by_label("Chat Uploads", exact=True).locator("option:checked")).to_have_text(
        "Default"
    )
    dialog.get_by_label("Name", exact=True).fill(renamed)
    save_terminal(page, dialog)

    settings = page.get_by_role("dialog")
    expect(connection_row(settings, renamed)).to_be_visible()
    expect(_row_switch(settings, renamed)).not_to_be_checked()
    [saved] = _stored_connections(admin_account)
    assert saved == {
        "id": connection_id,
        "name": renamed,
        "enabled": False,
        "url": terminal.base_url,
        "path": "/openapi.json",
        "key": KEY,
        "auth_type": "bearer",
        "forward_cookies": False,
        "server_type": "terminal",
        "label": "set over the API",
        "config": {
            "access_grants": [],
            "note": "set over the API",
            "chat_uploads": "default",
            "working_directory_context": True,
            "user_shell_tools": "auto",
        },
    }, f"the renamed connection was stored as {saved}"


def _make_a_user(admin, account) -> None:
    with admin.client() as client:
        changed = client.post(
            f"/api/v1/users/{account.id}/update",
            json={"role": "user", "name": account.name, "email": account.email},
        )
    assert changed.status_code == 200, changed.text


def test_a_save_the_instance_refuses_is_reported_and_stores_nothing(
    open_terminals, admin, admin_account, terminal
):
    name = _tagged("Build box")
    _store(
        admin_account,
        terminal.connection(name=name, key=KEY, auth_type="bearer", server_type="terminal"),
    )
    before = _stored_connections(admin_account)
    page, section = open_terminals()
    dialog = edit_terminal(section, name)
    dialog.get_by_label("Name", exact=True).fill(_tagged("Dry dock"))

    # another admin takes the admin role away while the dialog is open
    _make_a_user(admin, admin_account)
    assert field_the_browser_refuses(dialog) is None, "the browser holds the save back itself"
    dialog.get_by_role("button", name="Save", exact=True).click()

    expect(page.get_by_text("Failed to save terminal servers")).to_be_visible()
    expect(page.get_by_text("Terminal servers saved")).to_have_count(0)
    with admin.client() as client:
        after = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
    assert after == before, f"a refused save changed the stored connection to {after}"


def _named_group(admin, name: str, member) -> str:
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [member.id]}
        )
    assert added.status_code == 200, added.text
    return group_id


def _give_access(page: Page, dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    # the picker lists one page of users; searching reaches any account
    picker.get_by_placeholder("Search").fill(name)
    pick_member(picker, re.compile(re.escape(name)))
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(access_row(dialog, name)).to_be_visible()


def test_people_and_groups_given_access_in_the_dialog_are_offered_the_terminal(
    open_terminals, admin, admin_account, make_user, terminal
):
    name = _tagged("Build box")
    reader, member, outsider = make_user(), make_user(), make_user()
    group_name = _tagged("Riggers")
    group_id = _named_group(admin, group_name, member)
    page, section = open_terminals()
    dialog = add_terminal(section)
    fill_terminal(dialog, name, terminal.base_url, key=KEY)

    dialog.get_by_role("button", name="Access", exact=True).click()
    expect(dialog.get_by_text("No access grants. Private to you.")).to_be_visible()
    _give_access(page, dialog, reader.name)
    _give_access(page, dialog, group_name)
    level = access_row(dialog, group_name).get_by_role("combobox", name="Access level")
    expect(level.locator("option")).to_have_text(["Read", "Write"])
    level.select_option(label="Write")
    save_terminal(page, dialog)

    granted = sorted(
        (grant["principal_type"], grant["principal_id"], grant["permission"])
        for grant in _stored(admin_account, name)["config"]["access_grants"]
    )
    # write includes read, and is stored beside it
    expected = sorted(
        [("user", reader.id, "read"), ("group", group_id, "read"), ("group", group_id, "write")]
    )
    assert granted == expected, f"the access list was stored as {granted}"
    for person in (reader, member):
        assert name in _terminal_menu_of(person), (
            f"{person.name} was given access and is not offered the terminal"
        )
    assert name not in _terminal_menu_of(outsider), (
        "a person outside the access list is offered the terminal"
    )


def test_advanced_shows_where_the_tool_list_is_read_and_stores_that_and_forwarded_cookies(
    open_terminals, admin_account, terminal
):
    terminal.route("GET", "/spec/tools.json", json_answer(SPEC))
    page, section = open_terminals()
    name = _tagged("Build box")
    dialog = add_terminal(section)
    fill_terminal(dialog, name, terminal.base_url, key=KEY)
    show_advanced(dialog)
    spec_path = dialog.get_by_label("OpenAPI Spec", exact=True)
    forward_cookies = dialog.get_by_role("switch", name="Forward cookies")

    def read_from(address: str) -> Locator:
        return dialog.get_by_text(f'WebUI will make requests to "{address}"', exact=True)

    expect(spec_path).to_have_value("/openapi.json")
    expect(read_from(f"{terminal.base_url}/openapi.json")).to_be_visible()
    spec_path.fill("https://specs.example.com/tools.json")
    expect(read_from("https://specs.example.com/tools.json")).to_be_visible()
    spec_path.fill("spec/tools.json")
    expect(read_from(f"{terminal.base_url}/spec/tools.json")).to_be_visible()
    expect(forward_cookies).not_to_be_checked()
    forward_cookies.click()
    expect(forward_cookies).to_be_checked()
    save_terminal(page, dialog)

    stored = _stored(admin_account, name)
    assert stored and (stored["path"], stored["forward_cookies"]) == ("spec/tools.json", True), (
        f"the Advanced settings were stored as {stored}"
    )
    assert terminal.requests_to("/spec/tools.json"), (
        "the tool list was not read from the path set under Advanced: "
        f"{[request.path for request in terminal.received]}"
    )
    reopened = edit_terminal(section, name)
    show_advanced(reopened)
    expect(reopened.get_by_label("OpenAPI Spec", exact=True)).to_have_value("spec/tools.json")
    expect(reopened.get_by_role("switch", name="Forward cookies")).to_be_checked()


def test_a_connection_is_deleted_only_once_the_question_is_answered_with_delete(
    open_terminals, admin_account, terminal
):
    going, staying = _tagged("Old box"), _tagged("Build box")
    _store(
        admin_account,
        terminal.connection(name=going, server_type="terminal"),
        terminal.connection(name=staying, server_type="terminal"),
    )
    page, section = open_terminals()

    dialog = edit_terminal(section, going)
    delete_terminal(page, dialog, confirm=False)
    expect(dialog).to_be_visible()
    listed = [entry["name"] for entry in _stored_connections(admin_account)]
    assert listed == [going, staying], f"an unanswered Delete left {listed}"

    with page.expect_response(
        lambda response: (
            response.request.method == "POST" and response.url.endswith(TERMINAL_SERVERS_CONFIG[1])
        )
    ) as saved:
        delete_terminal(page, dialog)
    assert saved.value.ok, saved.value.text()
    expect(dialog).to_be_hidden()
    settings = page.get_by_role("dialog")
    expect(connection_row(settings, staying)).to_be_visible()
    expect(settings.get_by_text(going)).to_have_count(0)
    listed = [entry["name"] for entry in _stored_connections(admin_account)]
    assert listed == [staying], f"after the delete the connections are {listed}"

    leave_settings(page)
    open_terminal_menu(page)
    expect(listed_terminal(page, staying)).to_be_visible()
    expect(listed_terminal(page, going)).to_have_count(0)


def test_a_connection_is_listed_by_its_name_else_its_address_else_as_new_terminal(
    open_terminals, admin_account, terminal
):
    name = _tagged("Build box")
    _store(
        admin_account,
        terminal.connection(name=name),
        terminal.connection(name=""),
        terminal.connection(name="", url=""),
    )

    page, section = open_terminals()

    for shown in (name, terminal.base_url, "New Terminal"):
        expect(connection_row(section, shown)).to_be_visible()
    expect(section.get_by_role("switch")).to_have_count(3)
    expect(section.get_by_text("No terminal connections configured.")).to_have_count(0)
