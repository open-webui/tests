"""Journey: a person gives an automation a terminal in its Create dialog and on its page.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24). The dialog has a Terminal
dropdown once the person has a terminal. Picking one shows a Working Directory field; both are
saved with the automation and shown again on its page, which is its editor since dev 9723b3a6a,
and when it is cloned. Picking the picked
terminal again takes it off, and its working directory with it. While a terminal is picked for
an automation that writes to a chat, the coding agents installed in that terminal join the
dialog's models; taking the terminal off drops a picked agent again.

The terminal is the fake of `harness.terminal_server`, and a real Open Terminal with the
stand-in coding agent (`harness.coding_agent`) for the agents; that test skips without
`open-terminal`. `integration/terminals/test_terminal_automations.py` has what the server
does with the saved terminal.

Discriminates: passes on the dev c15f28649 build; one frontend copy with four edits turns each
test red, with e2e/models/test_automation_runs.py green on it as the control. The dropdown
shown without a terminal fails the offers test; a clone not taking over the working directory
fails the saved-and-shown-again test at the clone; the working directory kept when the terminal
is taken off fails the taken-off test; the agents left out of the dialog's models fail the
agents test. On dev 2fdf4ce6b, where the page replaced the Edit dialog, a frontend copy keeping the
stored terminal when it is taken off on the page turns the taken-off test red.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.coding_agent import FAST, connect_agent_terminal, model_id
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID
from utils.automations_page import (
    automation_page,
    automation_row,
    close_terminal_menu,
    instructions,
    open_terminal_menu,
    save_changes,
    terminal_button,
    toggle_terminal,
    wait_until_loaded,
    working_directory,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0"
AGENT_NAME = f"opencode/{FAST}"
DIRECTORY = "/srv/nets"


@pytest.fixture
def scheduler(admin, preserve, make_user):
    """A person who may use automations; their automations go afterwards."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    account = make_user()
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


@pytest.fixture
def terminal(preserve) -> Iterator[FakeTerminalServer]:
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        yield server


def _connect(admin, server: FakeTerminalServer, *people) -> dict:
    """The admin saves a connection to the terminal under a name of its own, for those people."""
    connection = server.connection(
        name=f"Harbour box {uuid.uuid4().hex[:6]}",
        config={"access_grants": [read_grant(person.id) for person in people]},
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection


def _stored(owner, automation_id: str) -> dict:
    with owner.client() as client:
        read = client.get(f"/api/v1/automations/{automation_id}")
    assert read.status_code == 200, read.text
    return read.json()["data"]


def _open_create_dialog(page: Page) -> Locator:
    page.goto("/automations")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    dialog = page.get_by_role("dialog")
    wait_until_loaded(dialog)
    return dialog


def _model_menu(page: Page, dialog: Locator, shown: str, search: str) -> Locator:
    """The dialog's model dropdown, opened and narrowed to `search`."""
    dialog.get_by_role("button", name=shown).first.click()
    menu = page.get_by_role("menu")
    menu.get_by_role("textbox", name="Search a model").fill(search)
    return menu


def _pick_model(page: Page, dialog: Locator, name: str, shown: str = "Select model") -> None:
    _model_menu(page, dialog, shown, name).get_by_role("menuitem", name=name).click()
    expect(page.get_by_role("menu")).to_have_count(0)
    expect(dialog.get_by_role("button", name=name).first).to_be_visible()


def _create(page: Page, dialog: Locator, prompt: str, title: str) -> str:
    """Fill the prompt and title, press Create and return the new automation's id."""
    dialog.get_by_role("textbox", name="Instructions").fill(prompt)
    dialog.get_by_role("textbox", name="Automation title").fill(title)
    dialog.get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/automations/[0-9a-f-]+$"))
    expect(instructions(page)).to_have_value(prompt)
    return page.url.rsplit("/", 1)[-1]


def test_the_automation_dialog_offers_a_terminal_only_once_the_person_has_one(
    page_for, admin, scheduler, terminal
):
    page = page_for(scheduler, path="/automations")

    without = _open_create_dialog(page)
    expect(without.get_by_role("button", name="Select model").first).to_be_visible()
    expect(terminal_button(without)).to_have_count(0)

    connection = _connect(admin, terminal, scheduler)
    with_one = _open_create_dialog(page)
    menu = open_terminal_menu(page, with_one)
    expect(menu.get_by_role("menuitem", name=connection["name"])).to_be_visible()
    expect(working_directory(menu)).to_have_count(0)


def test_a_terminal_and_working_directory_picked_for_an_automation_are_saved_and_shown_again(
    page_for, admin, scheduler, terminal
):
    connection = _connect(admin, terminal, scheduler)
    name = connection["name"]
    tag = uuid.uuid4().hex[:6]
    prompt, title = f"count the nets {tag}", f"Net watch {tag}"
    page = page_for(scheduler, path="/automations")

    creating = _open_create_dialog(page)
    _pick_model(page, creating, MOCK_MODEL_ID)
    toggle_terminal(page, open_terminal_menu(page, creating), name)
    working_directory(open_terminal_menu(page, creating, name)).fill(DIRECTORY)
    close_terminal_menu(page, creating, name)
    automation_id = _create(page, creating, prompt, title)

    stored = _stored(scheduler, automation_id)
    assert stored["terminal"] == {"server_id": connection["id"], "cwd": DIRECTORY}, (
        f"the automation was saved with the terminal {stored.get('terminal')}"
    )

    details = automation_page(page)
    expect(working_directory(open_terminal_menu(page, details, name))).to_have_value(DIRECTORY)
    close_terminal_menu(page, details, name)

    page.goto("/automations")
    page.get_by_role("textbox", name="Search Automations").fill(title)
    row = automation_row(page, title)
    row.get_by_role("button", name="Automation Menu").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Clone").click()
    cloning = page.get_by_role("dialog")
    wait_until_loaded(cloning)
    expect(cloning.get_by_role("textbox", name="Automation title")).to_have_value(
        f"{title} (Clone)"
    )
    expect(working_directory(open_terminal_menu(page, cloning, name))).to_have_value(DIRECTORY)


def test_picking_the_picked_terminal_again_takes_it_and_its_working_directory_off(
    page_for, admin, scheduler, terminal
):
    connection = _connect(admin, terminal, scheduler)
    name = connection["name"]
    form = {
        "name": f"Watch {uuid.uuid4().hex[:6]}",
        "is_active": False,
        "data": {
            "prompt": f"count the nets {uuid.uuid4().hex[:6]}",
            "model_id": MOCK_MODEL_ID,
            "rrule": SCHEDULE,
            "terminal": {"server_id": connection["id"], "cwd": DIRECTORY},
            "target": {"type": "chat"},
        },
    }
    with scheduler.client() as client:
        created = client.post("/api/v1/automations/create", json=form)
    assert created.status_code == 200, created.text
    automation_id = created.json()["id"]
    page = page_for(scheduler, path=f"/automations/{automation_id}")
    expect(instructions(page)).to_have_value(form["data"]["prompt"])

    editing = automation_page(page)
    menu = open_terminal_menu(page, editing, name)
    expect(working_directory(menu)).to_have_value(DIRECTORY)
    toggle_terminal(page, menu, name)
    expect(terminal_button(editing)).to_be_visible()

    menu = open_terminal_menu(page, editing)
    expect(working_directory(menu)).to_have_count(0)
    toggle_terminal(page, menu, name)
    menu = open_terminal_menu(page, editing, name)
    expect(working_directory(menu)).to_have_value("")
    toggle_terminal(page, menu, name)
    save_changes(page)

    stored = _stored(scheduler, automation_id)
    assert stored.get("terminal") is None, (
        f"the terminal taken off in the dialog is still saved: {stored['terminal']}"
    )


def test_the_agents_of_a_picked_terminal_join_the_models_of_an_automation(
    page_for, admin, scheduler, preserve, agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    name = f"Agent box {uuid.uuid4().hex[:6]}"
    connection = connect_agent_terminal(admin, agent_terminal, scheduler, name=name)
    tag = uuid.uuid4().hex[:6]
    prompt, title = f"tidy the repository {tag}", f"Tidy up {tag}"
    page = page_for(scheduler, path="/automations")

    dialog = _open_create_dialog(page)
    models = _model_menu(page, dialog, "Select model", MOCK_MODEL_ID)
    expect(models.get_by_role("menuitem", name=MOCK_MODEL_ID)).to_be_visible()
    models.get_by_role("textbox", name="Search a model").fill(AGENT_NAME)
    expect(models.get_by_role("menuitem")).to_have_count(0)
    dialog.get_by_role("button", name="Select model").first.click()
    expect(page.get_by_role("menu")).to_have_count(0)

    toggle_terminal(page, open_terminal_menu(page, dialog), name)
    _pick_model(page, dialog, AGENT_NAME)
    toggle_terminal(page, open_terminal_menu(page, dialog, name), name)
    expect(dialog.get_by_role("button", name="Select model").first).to_be_visible()

    toggle_terminal(page, open_terminal_menu(page, dialog), name)
    _pick_model(page, dialog, AGENT_NAME)
    automation_id = _create(page, dialog, prompt, title)

    stored = _stored(scheduler, automation_id)
    assert stored["model_id"] == model_id(connection["id"], FAST), (
        f"the automation was saved with the model {stored['model_id']}"
    )
    assert stored["terminal"]["server_id"] == connection["id"], (
        f"the agent's automation was saved with the terminal {stored['terminal']}"
    )
