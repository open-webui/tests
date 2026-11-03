"""Regression: an admin sets Working Directory Context and User Shell Tools on a terminal.

Commit `a20b622ba` added both to the Advanced part of the Add/Edit Terminal Connection dialog
under Admin Settings > Integrations > Terminal. Working Directory Context is a switch, on by
default; User Shell Tools is a select of Automatic (the default) and Always Include. Saving
stores them in the connection's `config` as `working_directory_context` and `user_shell_tools`,
and reopening the connection shows them as saved. A connection saved without touching them
stores the defaults. integration/tools/test_terminal_connection_settings.py covers what each
setting changes for the model.

For a Terminals orchestrator the dialog also edits the connection's policy, and Save sends the
policy back to the orchestrator, which replaces it whole. The dialog shows only some policy
fields, so saving used to drop the rest (restricted mode, security contexts) and terminals then
started without them (open-webui/open-webui#33580). Saving must keep every field the dialog does
not show, untouched or with shown fields edited.

Discriminates: passes on dev 1c010b438 with its built frontend; on a build with a20b622ba (and the
unrelated 7d205a86d) reverted the first two tests fail (the dialog has neither control and the
connection is stored without them). The two policy tests pass with the #33580 fix and fail on dev
f3ef8b949 (the saved policy lacks the fields the dialog does not show).
"""

from __future__ import annotations

import json
import secrets
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    TerminalRequest,
    configure_terminals,
    serving_terminal,
)
from utils.tool_servers import connection_row
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
# The dialog shows the first six fields; the rest are set in Terminals itself.
STORED_POLICY = {
    "image": "ghcr.io/open-webui/open-terminal:latest",
    "cpu_limit": "2",
    "memory_limit": "2Gi",
    "storage": "10Gi",
    "idle_timeout_minutes": 45,
    "env": {"TZ": "Europe/Vienna"},
    "restricted": True,
    "storage_mode": "per-user",
    "pod_security_context": {"runAsNonRoot": True, "runAsUser": 1000},
    "container_security_context": {"allowPrivilegeEscalation": False},
}
STORED_LIFECYCLE = {"max_lifetime_minutes": 600}


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        yield server


@pytest.fixture
def admin_account(make_user, preserve):
    preserve(TERMINAL_SERVERS_CONFIG)
    return make_user(role="admin")


def terminal_section(page: Page) -> Locator:
    page.goto("/admin/settings/integrations")
    heading = page.get_by_role("dialog").get_by_text("Open Terminal", exact=True)
    expect(heading).to_be_visible()
    return heading.locator("xpath=../..")


def open_add_dialog(page: Page) -> Locator:
    tooltip_button(terminal_section(page), "Add Connection").click()
    return terminal_dialog(page, "Add Terminal Connection")


def terminal_dialog(page: Page, title: str) -> Locator:
    dialog = page.get_by_role("dialog").filter(has_text=title)
    expect(dialog).to_be_visible()
    return dialog


def show_advanced(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Advanced", exact=True).click()
    expect(dialog.get_by_label("User Shell Tools")).to_be_visible()


def fill_and_save(page: Page, dialog: Locator, name: str, url: str) -> None:
    dialog.get_by_label("Name", exact=True).fill(name)
    dialog.get_by_label("URL", exact=True).fill(url)
    save(page, dialog)


def save(page: Page, dialog: Locator) -> None:
    with page.expect_response(
        lambda response: (
            response.url.endswith(TERMINAL_SERVERS_CONFIG[1]) and response.request.method == "POST"
        )
    ) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()


def stored_config(admin_account, name: str) -> dict:
    with admin_account.client() as client:
        connections = client.get(TERMINAL_SERVERS_CONFIG[0]).json()["TERMINAL_SERVER_CONNECTIONS"]
    [connection] = [entry for entry in connections if entry.get("name") == name]
    return connection.get("config") or {}


def test_both_settings_are_saved_and_shown_again(page_for, admin_account, terminal):
    page = page_for(admin_account)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = open_add_dialog(page)
    show_advanced(dialog)
    working_directory = dialog.get_by_role("switch", name="Working Directory Context")
    expect(working_directory).to_be_checked()
    working_directory.click()
    expect(working_directory).not_to_be_checked()
    dialog.get_by_label("User Shell Tools").select_option(label="Always Include")
    expect(dialog.get_by_text("Keep tools listed even while your shell is closed.")).to_be_visible()

    fill_and_save(page, dialog, name, terminal.base_url)

    config = stored_config(admin_account, name)
    assert config.get("working_directory_context") is False, config
    assert config.get("user_shell_tools") == "always", config

    section = terminal_section(page)
    tooltip_button(connection_row(section, name), "Configure").click()
    reopened = terminal_dialog(page, "Edit Terminal Connection")
    show_advanced(reopened)
    expect(reopened.get_by_role("switch", name="Working Directory Context")).not_to_be_checked()
    expect(reopened.get_by_label("User Shell Tools")).to_have_value("always")


def test_a_connection_saved_untouched_stores_the_defaults(page_for, admin_account, terminal):
    page = page_for(admin_account)
    name = f"Build box {secrets.token_hex(3)}"
    dialog = open_add_dialog(page)
    show_advanced(dialog)
    expect(dialog.get_by_role("switch", name="Working Directory Context")).to_be_checked()
    expect(dialog.get_by_label("User Shell Tools")).to_have_value("auto")

    fill_and_save(page, dialog, name, terminal.base_url)

    config = stored_config(admin_account, name)
    assert config.get("working_directory_context") is True, config
    assert config.get("user_shell_tools") == "auto", config


def serve_policy(terminal: FakeTerminalServer, policy_id: str) -> str:
    """Serve `policy_id`'s policy and lifecycle as an orchestrator does; returns the policy path."""
    path = f"/api/v1/policies/{policy_id}"

    def echo(request: TerminalRequest):
        return json_answer({"id": policy_id, "data": json.loads(request.body)})

    terminal.route("GET", path, json_answer({"id": policy_id, "data": STORED_POLICY}))
    terminal.route("PUT", path, echo)
    terminal.route("GET", f"{path}/lifecycle", json_answer({"data": STORED_LIFECYCLE}))
    terminal.route("PUT", f"{path}/lifecycle", echo)
    return path


def open_orchestrator(page_for, admin_account, terminal) -> tuple[Page, Locator, str]:
    """Save an orchestrator connection and open it in the Edit dialog with its policy loaded."""
    name = f"Orchestrator {secrets.token_hex(3)}"
    policy_id = f"policy-{secrets.token_hex(3)}"
    policy_path = serve_policy(terminal, policy_id)
    with admin_account.client() as client:
        configure_terminals(
            client, terminal.connection(name=name, server_type="orchestrator", policy_id=policy_id)
        )
    page = page_for(admin_account)
    tooltip_button(connection_row(terminal_section(page), name), "Configure").click()
    dialog = terminal_dialog(page, "Edit Terminal Connection")
    dialog.get_by_role("button", name="Orchestrator", exact=True).click()
    expect(dialog.get_by_label("Image")).to_have_value(STORED_POLICY["image"])
    return page, dialog, policy_path


def sent_to(terminal: FakeTerminalServer, path: str) -> dict:
    [put] = [request for request in terminal.requests_to(path) if request.method == "PUT"]
    return json.loads(put.body)


def test_saving_an_orchestrator_untouched_keeps_its_whole_policy(page_for, admin_account, terminal):
    page, dialog, policy_path = open_orchestrator(page_for, admin_account, terminal)

    save(page, dialog)

    policy = sent_to(terminal, policy_path)
    assert policy == STORED_POLICY, f"saving the untouched dialog changed the policy to {policy}"
    lifecycle = sent_to(terminal, f"{policy_path}/lifecycle")
    assert lifecycle == STORED_LIFECYCLE, f"saving changed the lifecycle to {lifecycle}"


def test_editing_policy_fields_keeps_the_fields_the_dialog_does_not_show(
    page_for, admin_account, terminal
):
    page, dialog, policy_path = open_orchestrator(page_for, admin_account, terminal)
    dialog.get_by_label("CPU", exact=True).fill("4")
    dialog.get_by_label("Storage", exact=True).select_option(label="Ephemeral")
    expect(dialog.get_by_label("Size", exact=True)).to_be_hidden()

    save(page, dialog)

    policy = sent_to(terminal, policy_path)
    expected = {key: value for key, value in STORED_POLICY.items() if key != "storage"}
    assert policy == {**expected, "cpu_limit": "4"}, f"the edited policy was saved as {policy}"
