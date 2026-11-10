"""Regression: an admin sets Working Directory Context and User Shell Tools on a terminal.

Commit `a20b622ba` added both to the Advanced part of the Add/Edit Terminal Connection dialog
under Admin Settings > Integrations > Terminal. Working Directory Context is a switch, on by
default; User Shell Tools is a select of Automatic (the default) and Always Include. Saving
stores them in the connection's `config` as `working_directory_context` and `user_shell_tools`,
and reopening the connection shows them as saved. A connection saved without touching them
stores the defaults. integration/tools/test_terminal_connection_settings.py covers what each
setting changes for the model.

A dialog shows only some of what is stored, and Save must keep the rest. For a Terminals
orchestrator the connection dialog used to edit the connection's policy and dropped the policy
fields it did not show (restricted mode, security contexts), so terminals started without them
(open-webui/open-webui#33580). Dev 6f2fc7bc7 rebuilt the dialog and replaced policies with
templates: the connection dialog edits the connection alone, and each terminal of an
orchestrator has a Terminal settings dialog of its own in the chat input's Terminal menu, which
shows its name, scope, image, limits, network and environment. The same must hold for both: a
connection saved untouched is stored as it was, fields the dialog does not show included, and a
terminal saved from its dialog, untouched or with a shown field edited, keeps the fields its
dialog does not show (idle timeout, maximum lifetime). The terminal tests run on the real
Terminals service (`harness.terminals`) and skip without Docker or its images.

The rebuilt connection dialog asks for an API key whatever Auth is set to, so a terminal that
needs none (Auth set to None) cannot be saved without typing one; 0.13.3 showed no key field
there. The test of that stays red until it is fixed (finding: `issue: [regression] [dev] A
terminal connection with Auth set to None cannot be saved without an API key`).

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1, the Auth None test aside.
The two settings tests and the untouched connection test fail in a backend copy that stores a
connection without its `config` (the settings tests also failed on a build without a20b622ba,
where the dialog has neither control). The two terminal tests fail in a backend copy whose
template update clears every stored field the request does not name, as a whole replacement
would (the idle timeout and lifetime are gone).
"""

from __future__ import annotations

import secrets
from typing import Iterator

import pytest
from playwright.sync_api import expect

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    serving_terminal,
)
from harness.terminals import create_template
from utils.terminal_connections import (
    ADMIN_INTEGRATIONS,
    add_terminal,
    choose_terminal_auth,
    edit_terminal,
    fill_terminal,
    save_terminal,
    show_advanced,
    shown_admin_terminal_section,
)
from utils.user_terminals import open_terminal_settings, save_terminal_settings

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

KEY = "build-box-key"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
# The Terminal settings dialog shows none of these; they are set in Terminals itself.
UNSHOWN_TEMPLATE_FIELDS = {"idle_timeout_minutes": 45, "max_lifetime_minutes": 600}
SHOWN_TEMPLATE_FIELDS = {
    "scope": "user",
    "cpu_limit": "1",
    "memory_limit": "512Mi",
    "env": {"TZ": "UTC", "OPEN_TERMINAL_START_DIR": "work"},
}


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        # Save asks the server what it is before it stores the connection
        server.route("GET", "/api/config", json_answer({"features": {}}))
        yield server


@pytest.fixture
def admin_account(make_user, preserve):
    preserve(TERMINAL_SERVERS_CONFIG)
    return make_user(role="admin")


def stored_connection(admin_account, name: str) -> dict:
    with admin_account.client() as client:
        connections = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
    [connection] = [entry for entry in connections if entry.get("name") == name]
    return connection


def test_both_settings_are_saved_and_shown_again(page_for, admin_account, terminal):
    page = page_for(admin_account, path=ADMIN_INTEGRATIONS)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = add_terminal(shown_admin_terminal_section(page))
    show_advanced(dialog)
    working_directory = dialog.get_by_role("switch", name="Working Directory Context")
    expect(working_directory).to_be_checked()
    working_directory.click()
    expect(working_directory).not_to_be_checked()
    dialog.get_by_label("User Shell Tools").select_option(label="Always Include")
    expect(dialog.get_by_text("Keep tools listed even while your shell is closed.")).to_be_visible()

    fill_terminal(dialog, name, terminal.base_url, key=KEY)
    save_terminal(page, dialog)

    config = stored_connection(admin_account, name).get("config") or {}
    assert config.get("working_directory_context") is False, config
    assert config.get("user_shell_tools") == "always", config

    # a page of its own reads the connection from the instance again
    again = page_for(admin_account, path=ADMIN_INTEGRATIONS)
    reopened = edit_terminal(shown_admin_terminal_section(again), name)
    show_advanced(reopened)
    expect(reopened.get_by_role("switch", name="Working Directory Context")).not_to_be_checked()
    expect(reopened.get_by_label("User Shell Tools")).to_have_value("always")


def test_a_connection_saved_untouched_stores_the_defaults(page_for, admin_account, terminal):
    page = page_for(admin_account, path=ADMIN_INTEGRATIONS)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = add_terminal(shown_admin_terminal_section(page))
    show_advanced(dialog)
    expect(dialog.get_by_role("switch", name="Working Directory Context")).to_be_checked()
    expect(dialog.get_by_label("User Shell Tools")).to_have_value("auto")

    fill_terminal(dialog, name, terminal.base_url, key=KEY)
    save_terminal(page, dialog)

    config = stored_connection(admin_account, name).get("config") or {}
    assert config.get("working_directory_context") is True, config
    assert config.get("user_shell_tools") == "auto", config


def test_a_terminal_without_authentication_is_saved_without_a_key(
    page_for, admin_account, terminal
):
    """Red on dev c15f28649, where the API Key field must be filled in whatever Auth is set to:
    `issue: [regression] [dev] A terminal connection with Auth set to None cannot be saved
    without an API key`."""
    page = page_for(admin_account, path=ADMIN_INTEGRATIONS)
    name = f"Open box {secrets.token_hex(3)}"
    dialog = add_terminal(shown_admin_terminal_section(page))
    choose_terminal_auth(dialog, "None")
    expect(dialog.get_by_text("No authentication", exact=True)).to_be_visible()

    fill_terminal(dialog, name, terminal.base_url)
    save_terminal(page, dialog)

    stored = stored_connection(admin_account, name)
    assert (stored["auth_type"], stored["key"]) == ("none", ""), stored


def test_saving_a_stored_connection_untouched_keeps_everything_stored(
    page_for, admin_account, terminal
):
    name = f"Build box {secrets.token_hex(3)}"
    # the dialog shows neither the list's switch nor a config entry it does not know
    stored = terminal.connection(
        name=name,
        enabled=False,
        key=KEY,
        auth_type="session",
        forward_cookies=True,
        server_type="terminal",
        config={
            "access_grants": [],
            "chat_uploads": "default",
            "working_directory_context": False,
            "user_shell_tools": "always",
            "label": "set over the API",
        },
    )
    with admin_account.client() as client:
        configure_terminals(client, stored)
    page = page_for(admin_account, path=ADMIN_INTEGRATIONS)
    dialog = edit_terminal(shown_admin_terminal_section(page), name)
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)

    save_terminal(page, dialog)

    saved = stored_connection(admin_account, name)
    changed = {
        field: (stored.get(field), saved.get(field))
        for field in stored.keys() | saved.keys()
        if stored.get(field) != saved.get(field)
    }
    assert not changed, f"saving the untouched dialog changed (stored, saved): {changed}"


@pytest.fixture
def terminal_settings(page_for, admin_account, connect_terminals, terminals):
    """A terminal of the orchestrator holding fields its dialog does not show, opened in its
    Terminal settings dialog; yields (page, dialog, terminal id, what the orchestrator stores)."""
    name = f"Workbench {secrets.token_hex(3)}"
    connection = connect_terminals()
    with admin_account.client() as client:
        created = create_template(
            client, connection, name=name, **SHOWN_TEMPLATE_FIELDS, **UNSHOWN_TEMPLATE_FIELDS
        )
    [template] = terminals.templates()
    assert template["data"].items() >= UNSHOWN_TEMPLATE_FIELDS.items(), (
        f"the orchestrator did not store the fields the dialog does not show: {template}"
    )
    page = page_for(admin_account)
    return page, open_terminal_settings(page, name), created["terminal_id"], template["data"]


def test_saving_a_terminal_untouched_keeps_everything_stored(terminal_settings, terminals):
    page, dialog, terminal_id, stored = terminal_settings

    save_terminal_settings(page, dialog, terminal_id)

    [template] = terminals.templates()
    assert template["data"] == stored, (
        f"saving the untouched Terminal settings changed the terminal to {template['data']}"
    )


def test_editing_a_terminal_keeps_the_fields_its_dialog_does_not_show(terminal_settings, terminals):
    page, dialog, terminal_id, stored = terminal_settings
    dialog.get_by_role("button", name="Advanced").click()
    dialog.get_by_label("CPU", exact=True).fill("0.5")

    save_terminal_settings(page, dialog, terminal_id)

    [template] = terminals.templates()
    assert template["data"] == {**stored, "cpu_limit": "0.5"}, (
        f"the edited terminal was saved as {template['data']}"
    )
