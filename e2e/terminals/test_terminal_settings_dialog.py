"""Journey: a person manages a terminal of a Terminals connection from its Terminal settings.

New on dev after 0.13.3 (6f2fc7bc7, titled `refac`, changed eight times up to c15f28649): a
person with write access to a Terminals connection opens Terminal settings from a terminal's
More options in the chat input's Terminal menu. The dialog shows what the terminal is stored
with and saves changes, a new name included. Its Access button reads Private or Shared and
opens the sharing: people and groups, for reading, never everyone. Machines lists the
terminal's machines by whose they are (the person, the chat and its owner, or the terminal's
name for one shared machine) with Running or Stopped; a machine's menu restarts, stops and
deletes it after asking, starts it without asking, and opens System info and the files of a
machine the person can reach. What gets a machine of its own is locked while machines exist.
Delete terminal says how many machines go with it. A rename, a share, an unshare and a delete
reach the open pages of everyone who has the terminal without a reload. A person who may only
use the terminal is offered none of this.

The real Terminals service runs on Docker (`harness.terminals`); without its images or Docker
the tests skip. The two tests that kept fields the dialog does not show are in
e2e/admin/test_terminal_connection_settings.py.

`test_system_info_and_files_are_not_offered_for_another_persons_chat_machine` is red on dev
c15f28649: both are offered there and fail with "Chat access denied" (finding
`open-webui-terminal-settings-offers-info-and-files-for-others-chat-machine`).

Discriminates: the other nine pass on the dev c15f28649 build with Terminals 0.4.1 and Open
Terminal 0.17.7. In a backend copy: with a terminal's changes no longer pushed to the people
who have it, the rename, sharing and deleting tests fail (the other person's menu keeps the old
state); with the owner's name left out of a chat's machine, the per chat test fails; with the
person's name left out of a user's machine, the machines, locked and refused tests fail (the row
reads Deleted user); with everyone taken for a writer of the connection, the offered test fails
(the reader gets New terminal and their calls are answered). On a build of a frontend copy with
one change to the dialog each: the start directory not read from the stored terminal (the shows
what is stored test) and CPU left out of the Advanced line (the same test); the lock left out
(the locked test: the choice stays open with a machine); the Access button always Private and
sharing with everyone on offer (the sharing test); the machine count of Delete terminal fixed
at 0 (the deleting test); another person's machine treated as one's own (the machines test: it
offers System info and View files) and Stop no longer asking (the same test); System info and
View files never offered (the own machine test); New terminal offered to everyone and Terminal
settings on every terminal (the offered test); the error of a failed request not shown (the
refused test).
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import unquote

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.chat_history import seed_chat
from harness.terminal_server import configure_terminals, read_grant, write_grant
from harness.terminals import (
    create_template,
    execute,
    instance_of,
    output_of,
    share_template,
    wait_until_addresses_settle,
)
from utils.access_control import visibility_choices
from utils.cached_chat import open_terminal_options
from utils.member_picker import pick_member
from utils.user_terminals import (
    TEMPLATES,
    advanced,
    confirmation,
    listed_terminal,
    machine_actions,
    machine_row,
    machines_heading,
    open_machine_menu,
    open_terminal_menu,
    open_terminal_settings,
    save_terminal_settings,
    show_advanced,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NO_MACHINES = "No machines yet. One starts the first time the terminal is used."


class Managed:
    """A terminal on the connected orchestrator, the person who manages it (write access to the
    connection) and a person it is shared with (read access)."""

    def __init__(self, admin, writer, reader, connection, terminal_id, name):
        self.admin, self.writer, self.reader = admin, writer, reader
        self.connection, self.terminal_id, self.name = connection, terminal_id, name

    def run(self, person, command: str, chat_id: str | None = None) -> str:
        """Run `command` in the machine the terminal gives `person`, starting it if needed."""
        with person.client() as client:
            printed = output_of(execute(client, self.terminal_id, command, chat_id=chat_id))
        # a machine that just started would cost an open page the requests it has under way
        wait_until_addresses_settle()
        return printed

    def machine_state(self, person, **context) -> str:
        with person.client() as client:
            return instance_of(client, self.terminal_id, **context)["status"]


@pytest.fixture
def managed_terminal(admin, make_user, connect_terminals):
    """`managed_terminal(**data)`: the admin creates a terminal (one machine per person unless
    `scope` says otherwise) and shares it with a reader; returns the `Managed` terminal."""

    def manage(**data) -> Managed:
        writer, reader = make_user(), make_user()
        name = data.pop("name", f"Slipway {uuid.uuid4().hex[:8]}")
        connection = connect_terminals(write_grant(writer.id), read_grant(reader.id))
        with admin.client() as client:
            terminal_id = create_template(client, connection, name=name, **data)["terminal_id"]
            share_template(client, terminal_id, read_grant(reader.id))
        return Managed(admin, writer, reader, connection, terminal_id, name)

    return manage


def _stored(terminals, terminal_id: str) -> dict | None:
    """What the orchestrator keeps of the terminal; None once it is gone."""
    template_id = terminal_id.rsplit(":", 1)[1]
    found = [row["data"] for row in terminals.templates() if row["id"] == template_id]
    return found[0] if found else None


def _machines(terminals, terminal_id: str) -> dict[str, str]:
    """The state of each of the terminal's machines on the orchestrator, by whose it is."""
    template_id = terminal_id.rsplit(":", 1)[1]
    return {
        machine["context_id"]: machine["status"]
        for machine in terminals.instances()
        if machine["template_id"] == template_id
    }


def _acts_on_machine(page: Page, terminal_id: str, action: str):
    """Waits for the answer to the machine action (`start`, `stop`, `restart`, `instance`)."""

    def is_the_action(response) -> bool:
        return response.request.method in ("POST", "DELETE") and unquote(response.url).endswith(
            f"{TEMPLATES}/{terminal_id}/{action}"
        )

    return page.expect_response(is_the_action)


def _confirm(page: Page, title: str, label: str, message: str) -> None:
    asked = confirmation(page, title)
    expect(asked.get_by_text(message, exact=True)).to_be_visible()
    asked.get_by_role("button", name=label, exact=True).click()
    expect(asked).to_be_hidden()


def _access_dialog(page: Page, settings: Locator, name: str, shows: str) -> Locator:
    """Open the terminal's sharing from the button that reads `shows` (Private or Shared)."""
    settings.get_by_role("button", name=shows, exact=True).click()
    access = page.get_by_role("dialog", name=f"Access: {name}", exact=True)
    expect(access.get_by_role("button", name="Add Access")).to_be_visible()
    return access


def _share_with(page: Page, access: Locator, name: str) -> None:
    access.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    # the picker lists one page of users; searching reaches any account
    picker.get_by_placeholder("Search").fill(name)
    pick_member(picker, re.compile(re.escape(name)))
    with page.expect_response(lambda response: response.url.endswith("/access")) as stored:
        picker.get_by_role("button", name="Add", exact=True).click()
    assert stored.value.ok, f"the share was refused: {stored.value.text()}"
    expect(access.get_by_text(name)).to_be_visible()


def _saved_chat(person, title: str) -> str:
    """A saved chat of `person` with that title; returns its id."""
    with person.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": title}])
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert titled.status_code == 200, f"naming the chat failed: {titled.text}"
    return chat_id


def _close(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Close", exact=True).click()
    expect(dialog).to_be_hidden()


def test_terminal_settings_shows_what_is_stored_and_a_rename_reaches_everyone_who_has_it(
    page_for, managed_terminal, terminals
):
    tag = uuid.uuid4().hex[:8]
    variables = {
        "OPEN_TERMINAL_START_DIR": f"/home/user/dock-{tag}",
        "OPEN_TERMINAL_SETUP_SCRIPT": f"echo rigged-{tag} > rigging.txt",
        f"TIDE_{tag.upper()}": f"ebb-{tag}",
    }
    terminal = managed_terminal(env=variables, cpu_limit="0.5", memory_limit="256Mi")
    renamed = f"Dry dock {tag}"
    reader_page = page_for(terminal.reader)
    open_terminal_menu(reader_page)
    expect(listed_terminal(reader_page, terminal.name)).to_be_visible()
    page = page_for(terminal.writer)

    dialog = open_terminal_settings(page, terminal.name)
    expect(dialog.get_by_role("heading", name=terminal.name, exact=True)).to_be_visible()
    expect(
        dialog.get_by_label("Separate machine per", exact=True).locator("option:checked")
    ).to_have_text("User")
    expect(advanced(dialog)).to_contain_text(
        f"/home/user/dock-{tag} · Setup script · 0.5 CPU · 256Mi · 1 variables"
    )
    show_advanced(dialog)
    shown = {
        label: dialog.get_by_label(label, exact=True).input_value()
        for label in ("Start directory", "Setup script", "CPU", "Memory", "Variable name")
    }
    assert shown == {
        "Start directory": f"/home/user/dock-{tag}",
        "Setup script": f"echo rigged-{tag} > rigging.txt",
        "CPU": "0.5",
        "Memory": "256Mi",
        "Variable name": f"TIDE_{tag.upper()}",
    }, f"Terminal settings shows {shown}"
    expect(dialog.get_by_label("Variable value", exact=True)).to_have_value(f"ebb-{tag}")

    dialog.get_by_label("Name", exact=True).fill(renamed)
    save_terminal_settings(page, dialog, terminal.terminal_id)

    stored = _stored(terminals, terminal.terminal_id)
    assert (stored["name"], stored["env"]) == (renamed, variables), (
        f"the renamed terminal is stored as {stored}"
    )
    open_terminal_menu(page)
    expect(listed_terminal(page, renamed)).to_be_visible()
    expect(listed_terminal(reader_page, renamed)).to_be_visible()
    expect(listed_terminal(reader_page, terminal.name)).to_have_count(0)


def test_separate_machine_per_is_locked_until_the_terminals_machines_are_deleted(
    page_for, managed_terminal, terminals
):
    terminal = managed_terminal()
    writer = terminal.writer
    tag = uuid.uuid4().hex[:8]
    terminal.run(writer, f"echo keel-{tag} > keel.txt")
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)
    choice = dialog.get_by_label("Separate machine per", exact=True)

    expect(machines_heading(dialog, 1)).to_be_visible()
    expect(machine_row(dialog, writer.name)).to_contain_text("Running")
    expect(choice).to_be_disabled()
    expect(dialog.get_by_text("Delete all machines to change this.", exact=True)).to_be_visible()
    expect(dialog.get_by_text(NO_MACHINES, exact=True)).to_have_count(0)

    open_machine_menu(dialog, writer.name).get_by_role("menuitem", name="Delete machine").click()
    asked = confirmation(page, f"Delete {writer.name}?")
    asked.get_by_role("button", name="Cancel", exact=True).click()
    expect(asked).to_be_hidden()
    assert terminals.holding("keel.txt") == {f"user:{writer.id}": f"keel-{tag}\n"}, (
        f"a delete that was cancelled left {terminals.holding('keel.txt')}"
    )
    open_machine_menu(dialog, writer.name).get_by_role("menuitem", name="Delete machine").click()
    with _acts_on_machine(page, terminal.terminal_id, "instance") as deleted:
        _confirm(page, f"Delete {writer.name}?", "Delete", "The machine and its files are deleted.")
    assert deleted.value.ok, f"deleting the machine was refused: {deleted.value.text()}"

    expect(dialog.get_by_text(NO_MACHINES, exact=True)).to_be_visible()
    expect(machines_heading(dialog, 0)).to_be_visible()
    expect(choice).to_be_enabled()
    expect(dialog.get_by_text("Each person gets their own machine.", exact=True)).to_be_visible()
    assert _machines(terminals, terminal.terminal_id) == {}, (
        f"the orchestrator still has {_machines(terminals, terminal.terminal_id)}"
    )
    assert terminals.holding("keel.txt") == {}, "the deleted machine's file is still on a machine"
    choice.select_option(label="Chat")
    save_terminal_settings(page, dialog, terminal.terminal_id)
    assert _stored(terminals, terminal.terminal_id).get("scope") == "chat", (
        f"the terminal is stored as {_stored(terminals, terminal.terminal_id)}"
    )


def test_a_terminal_shared_from_its_settings_shows_up_at_once_and_goes_when_unshared(
    page_for, make_user, managed_terminal
):
    terminal = managed_terminal()
    admin, reader = terminal.admin, terminal.reader
    member = make_user()
    # a person shares with the groups they belong to
    team = make_group(admin, [member, terminal.writer])
    tag = uuid.uuid4().hex[:8]
    private = f"Lighthouse {tag}"
    with admin.client() as client:
        connection = {
            **terminal.connection,
            "config": {
                "access_grants": [
                    write_grant(terminal.writer.id),
                    read_grant(reader.id),
                    read_grant(member.id),
                ]
            },
        }
        configure_terminals(client, connection)
        private_id = create_template(client, connection, name=private)["terminal_id"]
        # a new terminal is shared with its creator; this one is shared with nobody
        share_template(client, private_id)
        share_template(client, terminal.terminal_id, read_grant(reader.id), read_grant(member.id))
        team_name = client.get(f"/api/v1/groups/id/{team}").json()["name"]
    reader_page, member_page = page_for(reader), page_for(member)
    for other_page in (reader_page, member_page):
        open_terminal_menu(other_page)
        expect(listed_terminal(other_page, terminal.name)).to_be_visible()
        expect(listed_terminal(other_page, private)).to_have_count(0)
    page = page_for(terminal.writer)
    dialog = open_terminal_settings(page, private)

    access = _access_dialog(page, dialog, private, shows="Private")
    expect(visibility_choices(access)).to_have_text(["Restricted"])
    page.keyboard.press("Escape")
    _share_with(page, access, reader.name)
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()
    expect(access.get_by_text("Read", exact=True)).to_be_visible()
    expect(access.get_by_role("combobox", name="Access level")).to_have_count(0)

    expect(listed_terminal(reader_page, private)).to_be_visible()
    expect(listed_terminal(member_page, terminal.name)).to_be_visible()
    expect(listed_terminal(member_page, private)).to_have_count(0)
    _close(access)
    expect(dialog.get_by_role("button", name="Shared", exact=True)).to_be_visible()

    access = _access_dialog(page, dialog, private, shows="Shared")
    with page.expect_response(lambda response: response.url.endswith("/access")) as stored:
        access.get_by_role("button", name=f"Remove {reader.name}", exact=True).click()
    assert stored.value.ok, f"the unshare was refused: {stored.value.text()}"
    _close(access)
    access = _access_dialog(page, dialog, private, shows="Private")
    expect(listed_terminal(reader_page, private)).to_have_count(0)
    expect(listed_terminal(reader_page, terminal.name)).to_be_visible()
    _share_with(page, access, team_name)
    expect(listed_terminal(member_page, private)).to_be_visible()
    expect(listed_terminal(reader_page, private)).to_have_count(0)
    with member.client() as client:
        printed = output_of(execute(client, private_id, f"echo beam-{tag}"))
    assert printed.strip() == f"beam-{tag}", (
        f"the group's member ran a command that printed {printed!r}"
    )


def test_deleting_a_terminal_says_how_many_machines_go_and_removes_it_for_everyone(
    page_for, managed_terminal, terminals
):
    terminal = managed_terminal()
    writer, reader = terminal.writer, terminal.reader
    kept = f"Breakwater {uuid.uuid4().hex[:8]}"
    with terminal.admin.client() as client:
        kept_id = create_template(client, terminal.connection, name=kept)["terminal_id"]
        share_template(client, kept_id, read_grant(reader.id))
    for person in (writer, reader):
        terminal.run(person, "echo ballast > ballast.txt")
    reader_page = page_for(reader)
    open_terminal_menu(reader_page)
    expect(listed_terminal(reader_page, terminal.name)).to_be_visible()
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)
    expect(machines_heading(dialog, 2)).to_be_visible()

    dialog.get_by_role("button", name="Delete terminal", exact=True).click()
    asked = confirmation(page, f"Delete {terminal.name}?")
    expect(
        asked.get_by_text("This deletes 2 machines and their files.", exact=True)
    ).to_be_visible()
    asked.get_by_role("button", name="Cancel", exact=True).click()
    expect(asked).to_be_hidden()
    assert len(terminals.holding("ballast.txt")) == 2, "a delete that was cancelled took machines"
    dialog.get_by_role("button", name="Delete terminal", exact=True).click()
    confirmation(page, f"Delete {terminal.name}?").get_by_role(
        "button", name="Delete", exact=True
    ).click()

    expect(dialog).to_be_hidden()
    open_terminal_menu(page)
    expect(listed_terminal(page, kept)).to_be_visible()
    expect(listed_terminal(page, terminal.name)).to_have_count(0)
    expect(listed_terminal(reader_page, kept)).to_be_visible()
    expect(listed_terminal(reader_page, terminal.name)).to_have_count(0)
    assert _stored(terminals, terminal.terminal_id) is None, "the orchestrator kept the terminal"
    assert terminals.holding("ballast.txt") == {}, (
        f"machines of the deleted terminal are left: {terminals.holding('ballast.txt')}"
    )
    assert _stored(terminals, kept_id) is not None, "the other terminal went with it"

    page.keyboard.press("Escape")
    dialog = open_terminal_settings(page, kept)
    expect(dialog.get_by_text(NO_MACHINES, exact=True)).to_be_visible()
    dialog.get_by_role("button", name="Delete terminal", exact=True).click()
    asked = confirmation(page, f"Delete {kept}?")
    expect(asked.get_by_text("This terminal is deleted.", exact=True)).to_be_visible()


def test_machines_are_listed_by_owner_and_stopped_started_and_restarted_from_their_menu(
    page_for, managed_terminal, terminals
):
    terminal = managed_terminal()
    writer, reader = terminal.writer, terminal.reader
    tag = uuid.uuid4().hex[:8]
    for person in (writer, reader):
        terminal.run(person, f"echo kept-{tag} > ~/kept.txt && echo lost-{tag} > /tmp/lost.txt")
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)

    expect(machines_heading(dialog, 2)).to_be_visible()
    expect(machine_row(dialog, writer.name)).to_contain_text("Running")
    expect(machine_row(dialog, reader.name)).to_contain_text("Running")
    assert machine_actions(open_machine_menu(dialog, reader.name)) == [
        "Restart",
        "Stop",
        "Delete machine",
    ]

    page.get_by_role("menu").get_by_role("menuitem", name="Stop", exact=True).click()
    _confirm(
        page,
        f"Stop {reader.name}?",
        "Stop",
        "Running work stops. Files are kept, and the machine starts again when it is next used.",
    )
    expect(machine_row(dialog, reader.name)).to_contain_text("Stopped")
    expect(machine_row(dialog, writer.name)).to_contain_text("Running")
    assert terminal.machine_state(reader) == "stopped", "the reader's machine was not stopped"
    assert terminal.machine_state(writer) == "running", "the writer's own machine was stopped too"

    assert machine_actions(open_machine_menu(dialog, reader.name)) == ["Start", "Delete machine"]
    with _acts_on_machine(page, terminal.terminal_id, "start") as started:
        page.get_by_role("menu").get_by_role("menuitem", name="Start", exact=True).click()
    assert started.value.ok, f"starting the machine was refused: {started.value.text()}"
    wait_until_addresses_settle()
    expect(machine_row(dialog, reader.name)).to_contain_text("Running")
    assert terminal.machine_state(reader) == "running", "the reader's machine was not started"

    open_machine_menu(dialog, writer.name).get_by_role(
        "menuitem", name="Restart", exact=True
    ).click()
    with _acts_on_machine(page, terminal.terminal_id, "restart") as restarted:
        _confirm(
            page,
            f"Restart {writer.name}?",
            "Restart",
            "Running work stops and the machine starts again. Files are kept.",
        )
    assert restarted.value.ok, f"restarting the machine was refused: {restarted.value.text()}"
    wait_until_addresses_settle()
    expect(machine_row(dialog, writer.name)).to_contain_text("Running")
    after = {
        person.name: terminal.run(person, "cat ~/kept.txt; cat /tmp/lost.txt 2>/dev/null; true")
        for person in (writer, reader)
    }
    assert after[writer.name].split() == [f"kept-{tag}"], (
        f"after its restart the writer's machine holds {after[writer.name]!r}"
    )
    assert _machines(terminals, terminal.terminal_id) == {
        f"user:{writer.id}": "running",
        f"user:{reader.id}": "running",
    }


def test_system_info_and_files_open_for_the_persons_own_running_machine_alone(
    page_for, managed_terminal, terminals
):
    terminal = managed_terminal(scope=None)
    writer, reader = terminal.writer, terminal.reader
    tag = uuid.uuid4().hex[:8]
    terminal.run(reader, f"echo chart > ~/chart-{tag}.txt")
    [machine] = [m for m in terminals.instances() if m["context_id"] == "default"]
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)

    expect(machines_heading(dialog, 1)).to_be_visible()
    expect(machine_row(dialog, terminal.name)).to_contain_text("Running")
    menu = open_machine_menu(dialog, terminal.name)
    assert machine_actions(menu) == [
        "System info",
        "View files",
        "Restart",
        "Stop",
        "Delete machine",
    ]
    menu.get_by_role("menuitem", name="View files", exact=True).click()
    files = page.get_by_role("dialog", name=f"Files on {terminal.name}", exact=True)
    expect(files.get_by_text(f"chart-{tag}.txt")).to_be_visible()
    _close(files)

    open_machine_menu(dialog, terminal.name).get_by_role(
        "menuitem", name="System info", exact=True
    ).click()
    info = page.get_by_role("dialog", name=f"System info for {terminal.name}", exact=True)
    expect(info.get_by_text(terminals.hostname(machine["id"]))).to_be_visible()
    _close(info)

    open_machine_menu(dialog, terminal.name).get_by_role(
        "menuitem", name="Stop", exact=True
    ).click()
    with _acts_on_machine(page, terminal.terminal_id, "stop"):
        confirmation(page, f"Stop {terminal.name}?").get_by_role(
            "button", name="Stop", exact=True
        ).click()
    expect(machine_row(dialog, terminal.name)).to_contain_text("Stopped")
    assert machine_actions(open_machine_menu(dialog, terminal.name)) == ["Start", "Delete machine"]


def test_a_machine_of_a_terminal_per_chat_is_named_after_its_chat_and_its_owner(
    page_for, managed_terminal
):
    terminal = managed_terminal(scope="chat")
    writer, reader = terminal.writer, terminal.reader
    tag = uuid.uuid4().hex[:8]
    chat_id = _saved_chat(reader, f"Catch log {tag}")
    terminal.run(reader, "echo net > net.txt", chat_id=chat_id)
    page = page_for(writer)

    dialog = open_terminal_settings(page, terminal.name)
    expect(machines_heading(dialog, 1)).to_be_visible()
    row = machine_row(dialog, f"Catch log {tag}")
    expect(row).to_contain_text(re.compile(rf"Catch log {tag}\s*·\s*{re.escape(reader.name)}"))
    expect(row).to_contain_text("Running")
    _close(dialog)

    with reader.client() as client:
        deleted = client.delete(f"/api/v1/chats/{chat_id}")
        assert deleted.status_code == 200, deleted.text
    dialog = open_terminal_settings(page, terminal.name)
    expect(machines_heading(dialog, 1)).to_be_visible()
    expect(machine_row(dialog, "Deleted chat")).to_contain_text("Running")


def test_system_info_and_files_are_not_offered_for_another_persons_chat_machine(
    page_for, managed_terminal
):
    """Red on dev c15f28649: both are offered there and can only fail."""
    terminal = managed_terminal(scope="chat")
    writer, reader = terminal.writer, terminal.reader
    tag = uuid.uuid4().hex[:8]
    own, theirs = f"Own log {tag}", f"Their log {tag}"
    terminal.run(writer, "true", chat_id=_saved_chat(writer, own))
    terminal.run(reader, "true", chat_id=_saved_chat(reader, theirs))
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)
    expect(machines_heading(dialog, 2)).to_be_visible()

    open_machine_menu(dialog, own).get_by_role("menuitem", name="System info", exact=True).click()
    info = page.get_by_role("dialog", name=f"System info for {own}", exact=True)
    expect(info.get_by_text("Memory", exact=True).first).to_be_visible()
    _close(info)

    offered = machine_actions(open_machine_menu(dialog, theirs))
    assert offered == ["Restart", "Stop", "Delete machine"], (
        f"the machine of another person's chat offers {offered}"
    )


def test_only_a_person_who_may_write_to_the_connection_is_offered_new_terminal_and_settings(
    page_for, managed_terminal
):
    terminal = managed_terminal()
    writer, reader = terminal.writer, terminal.reader
    terminal.run(reader, "true")
    terminal.run(writer, "true")

    page = page_for(writer)
    menu = open_terminal_menu(page)
    expect(menu.get_by_role("menuitem", name="New terminal", exact=True)).to_be_visible()
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)
    options = open_terminal_options(page, terminal.name)
    expect(options.get_by_role("menuitem", name="Terminal settings", exact=True)).to_be_visible()
    expect(options.get_by_role("menuitem", name="Stop", exact=True)).to_be_visible()

    reader_page = page_for(reader)
    menu = open_terminal_menu(reader_page)
    expect(listed_terminal(reader_page, terminal.name)).to_be_visible()
    expect(menu.get_by_role("menuitem", name="New terminal", exact=True)).to_have_count(0)
    reader_page.keyboard.press("Escape")
    expect(menu).to_have_count(0)
    options = open_terminal_options(reader_page, terminal.name)
    expect(options.get_by_role("menuitem", name="View files", exact=True)).to_be_visible()
    offered = machine_actions(reader_page.get_by_role("menu").last)
    assert offered == ["System info", "View files"], f"a reader is offered {offered}"
    with reader.client() as client:
        refused = {
            "settings": client.get(f"{TEMPLATES}/{terminal.terminal_id}"),
            "machines": client.get(f"{TEMPLATES}/{terminal.terminal_id}/instances"),
            "create": client.post(
                TEMPLATES,
                json={
                    "id": str(uuid.uuid4()),
                    "connection_id": terminal.connection["id"],
                    "data": {"name": "Jetty"},
                },
            ),
        }
    answers = {call: answer.status_code for call, answer in refused.items()}
    assert answers == {"settings": 403, "machines": 403, "create": 403}, (
        f"the reader's own calls were answered {answers}"
    )


def test_a_refused_save_or_machine_action_says_why_in_the_dialog_and_changes_nothing(
    page_for, managed_terminal, terminals
):
    terminal = managed_terminal()
    writer = terminal.writer
    tag = uuid.uuid4().hex[:8]
    terminal.run(writer, f"echo anchor-{tag} > anchor.txt")
    page = page_for(writer)
    dialog = open_terminal_settings(page, terminal.name)
    expect(dialog.get_by_role("alert")).to_have_count(0)

    show_advanced(dialog)
    dialog.get_by_label("Network", exact=True).fill(f"nowhere-{tag}")
    dialog.get_by_role("button", name="Save", exact=True).click()
    expect(dialog.get_by_role("alert")).to_have_text(f"Unknown network profile: 'nowhere-{tag}'")
    assert not _stored(terminals, terminal.terminal_id).get("network_profile"), (
        f"the refused network was stored: {_stored(terminals, terminal.terminal_id)}"
    )

    with terminal.admin.client() as client:
        only_reading = {"access_grants": [read_grant(writer.id), read_grant(terminal.reader.id)]}
        configure_terminals(client, {**terminal.connection, "config": only_reading})
    open_machine_menu(dialog, writer.name).get_by_role("menuitem", name="Delete machine").click()
    with _acts_on_machine(page, terminal.terminal_id, "instance") as deleted:
        confirmation(page, f"Delete {writer.name}?").get_by_role(
            "button", name="Delete", exact=True
        ).click()
    assert deleted.value.status == 403, f"the delete was answered {deleted.value.status}"

    expect(dialog.get_by_role("alert")).to_have_text("Terminal connection write access required")
    expect(machine_row(dialog, writer.name)).to_contain_text("Running")
    assert terminals.holding("anchor.txt") == {f"user:{writer.id}": f"anchor-{tag}\n"}, (
        f"the refused delete left {terminals.holding('anchor.txt')}"
    )
