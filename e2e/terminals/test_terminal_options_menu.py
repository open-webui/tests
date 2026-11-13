"""Journey: a terminal's More options in the chat input's Terminal menu, by the kind of terminal.

New on dev after 0.13.3 (f4cf2ca0c, titled `refac`). Every terminal in the Terminal menu has a More
options button. For a terminal the person only uses (a plain Open Terminal behind an admin
connection, a terminal on a Terminals connection that is only shared with them) it offers System
info and View files. View files opens "Files on <name>", a file browser of that terminal. For a
terminal the person manages (on a Terminals connection they may write to) the menu first reads the
machine the open chat would use: without a running machine it offers Start and Terminal settings;
with one it offers Restart and Stop, which ask first, System info, View files and Terminal
settings. Each action says in a toast what it does and that it is done. When the machine's state
cannot be read the menu offers what it offers for a machine that is not running. A terminal with a
machine per chat has no machine before the chat is saved, so a new chat offers Terminal settings
alone, and in a saved chat the entries are about that chat's machine. An open System info window
closes when the person loses the terminal.

A local fake terminal stands for the plain terminal. The real Terminals service
runs on Docker (`harness.terminals`); without its images or Docker those tests skip. What a
person who may only use a terminal on a Terminals connection is offered is in
test_terminal_settings_dialog.py, and a per chat terminal a reader cannot reach before the chat
is saved in e2e/tools/test_terminal_system_info.py. A person's own terminals from Settings were
removed from Open WebUI in 44d55bd0f, and their test with them.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7. On
builds of a frontend copy with one change each: every terminal taken for one the person manages
(the plain test: Restart, Stop and Terminal settings are offered); System info left open (the loses
the terminal test); Stop not asking, Restart not asking, and Start reporting something else when
done (the manager test); a state that cannot be read taken for running (the cannot be read test); a
per chat terminal taken for reachable before the chat is saved, and the machine actions never
naming the chat (the per chat test). The toasts of Stop and Restart, built as the one of Start is,
have no mutation of their own.
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import parse_qs, unquote, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.chat_history import seed_chat
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
    write_grant,
)
from harness.terminals import (
    create_template,
    execute,
    instance_of,
    output_of,
    share_template,
    wait_until_addresses_settle,
)
from utils.cached_chat import open_terminal_options
from utils.chat_ui import close_menu, expect_reply
from utils.user_terminals import TEMPLATES, confirmation, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOME = "/home/user/"
LOGS = "/home/user/logs/"
LISTINGS = {
    HOME: [{"name": "logs", "type": "directory"}, {"name": "readme.md", "type": "file"}],
    LOGS: [{"name": "tide.txt", "type": "file", "size": 16}],
}
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
USES = ["System info", "View files"]
NOT_RUNNING = ["Start", "Terminal settings"]
RUNNING = ["Restart", "Stop", "System info", "View files", "Terminal settings"]
STOP_MESSAGE = (
    "Running work stops. Files are kept, and the machine starts again when it is next used."
)
RESTART_MESSAGE = "Running work stops and the machine starts again. Files are kept."


def _listing(request):
    directory = parse_qs(urlparse(request.path).query).get("directory", [HOME])[0]
    return json_answer({"entries": LISTINGS.get(directory, []), "writable": True})


def _system_info(hostname: str) -> dict:
    return {
        "hostname": hostname,
        "os": "Linux",
        "arch": "x86_64",
        "environment": "host",
        "cpu": {"percent": 5.0, "cores": 2, "scope": "host"},
        "memory": {"total": 2**31, "used": 2**30, "scope": "host"},
        "disk": {"path": HOME, "total": 2**33, "used": 2**31, "free": 3 * 2**31},
        "network": [],
        "processes": [],
        "process_scope": "environment",
        "uptime_seconds": 60,
        "service_uptime_seconds": 60,
        "sampled_at": 1_800_000_000,
    }


@pytest.fixture
def fake_terminal():
    """A terminal with two files to browse, which also answers a browser on another origin."""
    with serving_terminal() as server:
        server.cors = True
        server.hostname = f"box-{uuid.uuid4().hex[:6]}"
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/api/config", json_answer({"features": {}}))
        server.route("GET", "/system/info", json_answer(_system_info(server.hostname)))
        server.route(
            "GET",
            "/files/cwd",
            json_answer({"cwd": HOME, "home": HOME, "root": {"path": HOME, "label": "user"}}),
        )
        server.route("POST", "/files/cwd", json_answer({"cwd": HOME}))
        server.route("GET", "/files/list", _listing)
        server.route("GET", "/files/read", json_answer({"content": "high at six"}))
        server.route("GET", "/skills", json_answer([]))
        yield server


def _connect(admin, preserve, server, *people) -> tuple[str, dict]:
    """The admin connects the fake terminal for those people; returns its name and connection."""
    preserve(TERMINAL_SERVERS_CONFIG)
    name = f"Lab box {uuid.uuid4().hex[:6]}"
    grants = [read_grant(person.id) for person in people]
    connection = server.connection(name=name, config={"access_grants": grants})
    with admin.client() as client:
        configure_terminals(client, connection)
    return name, connection


def _offered(options: Locator, entries: list[str]) -> None:
    """The More options read exactly `entries`, once the machine's state has been read."""
    # the Terminal menu behind it has entries without a text of their own
    expect(options.get_by_role("menuitem").filter(has_text=re.compile(r"\S"))).to_have_text(entries)


def _choose(page: Page, name: str, entry: str, offered: list[str]) -> None:
    """Open the More options of the terminal `name`, check what they offer and click `entry`."""
    options = open_terminal_options(page, name)
    _offered(options, offered)
    options.get_by_role("menuitem", name=entry, exact=True).click()


def _files_dialog(page: Page, name: str) -> Locator:
    dialog = page.get_by_role("dialog", name=f"Files on {name}", exact=True)
    expect(dialog.get_by_role("region", name="File browser")).to_be_visible()
    return dialog


def _system_info_dialog(page: Page, name: str) -> Locator:
    return page.get_by_role("dialog", name=f"System info for {name}")


def _close(dialog: Locator) -> None:
    dialog.get_by_role("button", name="Close", exact=True).click()
    expect(dialog).to_be_hidden()


def _saved_chat(person, title: str) -> str:
    """A saved chat of `person` with that title; returns its id."""
    with person.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": title}, {"role": "assistant", "content": "Noted."}],
        )
        titled = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": title}})
    assert titled.status_code == 200, f"naming the chat failed: {titled.text}"
    return chat_id


def _acts_on_machine(page: Page, terminal_id: str, action: str):
    """Waits for the answer to the machine action (`start`, `stop`, `restart`)."""

    def is_the_action(response) -> bool:
        return response.request.method == "POST" and unquote(response.url).endswith(
            f"{TEMPLATES}/{terminal_id}/{action}"
        )

    return page.expect_response(is_the_action)


def _state(person, terminal_id: str, **context) -> str:
    """The state of the machine the terminal gives `person`, as Open WebUI reports it."""
    with person.client() as client:
        return instance_of(client, terminal_id, **context)["status"]


def _run(person, terminal_id: str, command: str) -> str:
    with person.client() as client:
        printed = output_of(execute(client, terminal_id, command))
    # a machine that just started would cost an open page the requests it has under way
    wait_until_addresses_settle()
    return printed.strip()


def test_a_plain_terminal_offers_its_system_info_and_files_and_nothing_to_manage(
    page_for, make_user, admin, preserve, fake_terminal
):
    person = make_user(role="admin")
    name, _ = _connect(admin, preserve, fake_terminal)
    page = page_for(person)

    _choose(page, name, "View files", offered=USES)

    dialog = _files_dialog(page, name)
    expect(dialog.get_by_role("heading", name=name, exact=True)).to_be_visible()
    browser = dialog.get_by_role("region", name="File browser")
    expect(browser.get_by_text("readme.md", exact=True)).to_be_visible()
    browser.get_by_text("logs", exact=True).click()
    expect(browser.get_by_text("tide.txt", exact=True)).to_be_visible()
    expect(browser.get_by_text("readme.md", exact=True)).to_have_count(0)
    browser.get_by_text("tide.txt", exact=True).click()
    expect(browser.get_by_text("high at six")).to_be_visible()
    _close(dialog)

    _choose(page, name, "System info", offered=USES)
    expect(_system_info_dialog(page, name)).to_contain_text(fake_terminal.hostname)


def test_an_open_system_info_closes_when_the_person_loses_the_terminal(
    page_for, make_user, admin, preserve, fake_terminal
):
    person = make_user()
    team = make_group(admin, [person])
    preserve(TERMINAL_SERVERS_CONFIG)
    tag = uuid.uuid4().hex[:6]
    lost, kept = f"Lab box {tag}", f"Spare box {tag}"
    through_the_team = {"principal_type": "group", "principal_id": team, "permission": "read"}
    with admin.client() as client:
        configure_terminals(
            client,
            fake_terminal.connection(name=lost, config={"access_grants": [through_the_team]}),
            fake_terminal.connection(name=kept, config={"access_grants": [read_grant(person.id)]}),
        )
    page = page_for(person)
    _choose(page, lost, "System info", offered=USES)
    info = _system_info_dialog(page, lost)
    expect(info).to_contain_text(fake_terminal.hostname)

    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{team}/users/remove", json={"user_ids": [person.id]}
        )
    assert removed.status_code == 200, removed.text

    expect(info).to_be_hidden()
    menu = open_terminal_menu(page)
    expect(menu.get_by_role("menuitemradio")).to_have_text([kept])


def test_a_manager_starts_stops_and_restarts_their_machine_from_the_terminals_options(
    page_for, make_user, admin, connect_terminals, terminals
):
    manager, other = make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Slipway {tag}"
    connection = connect_terminals(write_grant(manager.id), read_grant(other.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
        share_template(client, terminal_id, read_grant(other.id))
    _run(other, terminal_id, "true")
    page = page_for(manager)

    with _acts_on_machine(page, terminal_id, "start") as started:
        _choose(page, name, "Start", offered=NOT_RUNNING)
    assert started.value.ok, f"starting the machine was refused: {started.value.text()}"
    expect(page.get_by_text(f"{name} started", exact=True)).to_be_visible()
    # Start leaves the Terminal menu open
    close_menu(page)
    assert _state(manager, terminal_id) == "running", "Start did not start the manager's machine"
    wait_until_addresses_settle()
    kept = _run(manager, terminal_id, f"echo kept-{tag} > ~/kept-{tag}.txt; touch /tmp/lost; ls")
    assert kept == f"kept-{tag}.txt", f"the machine's home holds {kept!r}"

    _choose(page, name, "View files", offered=RUNNING)
    dialog = _files_dialog(page, name)
    expect(dialog.get_by_text(f"kept-{tag}.txt", exact=True)).to_be_visible()
    _close(dialog)

    _choose(page, name, "Stop", offered=RUNNING)
    asked = confirmation(page, f"Stop {name}?")
    expect(asked.get_by_text(STOP_MESSAGE, exact=True)).to_be_visible()
    asked.get_by_role("button", name="Cancel", exact=True).click()
    expect(asked).to_be_hidden()
    assert _state(manager, terminal_id) == "running", "a Stop that was cancelled stopped it"
    _choose(page, name, "Stop", offered=RUNNING)
    with _acts_on_machine(page, terminal_id, "stop") as stopped:
        confirmation(page, f"Stop {name}?").get_by_role("button", name="Stop", exact=True).click()
    assert stopped.value.ok, f"stopping the machine was refused: {stopped.value.text()}"
    expect(page.get_by_text(f"{name} stopped", exact=True)).to_be_visible()
    assert _state(manager, terminal_id) == "stopped", "Stop did not stop the manager's machine"
    assert _state(other, terminal_id) == "running", "Stop also stopped another person's machine"
    wait_until_addresses_settle()

    with _acts_on_machine(page, terminal_id, "start") as started:
        _choose(page, name, "Start", offered=NOT_RUNNING)
    assert started.value.ok, f"starting the machine again was refused: {started.value.text()}"
    close_menu(page)
    wait_until_addresses_settle()
    _run(manager, terminal_id, "touch /tmp/lost")

    _choose(page, name, "Restart", offered=RUNNING)
    asked = confirmation(page, f"Restart {name}?")
    expect(asked.get_by_text(RESTART_MESSAGE, exact=True)).to_be_visible()
    with _acts_on_machine(page, terminal_id, "restart") as restarted:
        asked.get_by_role("button", name="Restart", exact=True).click()
    assert restarted.value.ok, f"restarting the machine was refused: {restarted.value.text()}"
    expect(page.get_by_text(f"{name} restarted", exact=True)).to_be_visible()
    wait_until_addresses_settle()
    left = _run(manager, terminal_id, "ls ~ /tmp/lost 2>&1")
    assert f"kept-{tag}.txt" in left and "No such file" in left, (
        f"after the restart the machine holds {left!r}"
    )
    machines = {machine["context_id"]: machine["status"] for machine in terminals.instances()}
    assert machines == {f"user:{manager.id}": "running", f"user:{other.id}": "running"}, (
        f"the orchestrator holds {machines}"
    )


def test_a_machine_whose_state_cannot_be_read_is_offered_as_one_that_is_not_running(
    page_for, make_user, admin, connect_terminals
):
    manager = make_user()
    name = f"Slipway {uuid.uuid4().hex[:8]}"
    connection = connect_terminals(write_grant(manager.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
    _run(manager, terminal_id, "true")
    page = page_for(manager)
    _offered(open_terminal_options(page, name), RUNNING)
    close_menu(page)

    page.route("**/instance?*", lambda route: route.fulfill(status=500, json={"detail": "down"}))

    _offered(open_terminal_options(page, name), NOT_RUNNING)
    assert _state(manager, terminal_id) == "running"


def test_on_a_terminal_with_a_machine_per_chat_the_options_are_about_the_open_chats_machine(
    page_for, make_user, admin, connect_terminals, terminals
):
    manager = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Per chat berth {tag}"
    connection = connect_terminals(write_grant(manager.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name, scope="chat")["terminal_id"]
    first = _saved_chat(manager, f"first {tag}")
    second = _saved_chat(manager, f"second {tag}")
    page = page_for(manager)

    # a chat that is not saved yet has no machine to act on
    _offered(open_terminal_options(page, name), ["Terminal settings"])
    close_menu(page)

    page.goto(f"/c/{first}")
    expect_reply(page, "Noted.")
    with _acts_on_machine(page, terminal_id, "start") as started:
        _choose(page, name, "Start", offered=NOT_RUNNING)
    assert started.value.ok, f"starting the chat's machine was refused: {started.value.text()}"
    expect(page.get_by_text(f"{name} started", exact=True)).to_be_visible()
    # Start leaves the Terminal menu open
    close_menu(page)
    states = {
        chat: _state(manager, terminal_id, chat_id=chat_id)
        for chat, chat_id in (("first", first), ("second", second))
    }
    assert states == {"first": "running", "second": "not_created"}, (
        f"Start in the first chat left the machines {states}"
    )
    machines = [machine["context_id"] for machine in terminals.instances()]
    assert machines == [f"chat:{first}"], f"the orchestrator holds {machines}"
    wait_until_addresses_settle()
    _offered(open_terminal_options(page, name), RUNNING)

    page.goto(f"/c/{second}")
    expect_reply(page, "Noted.")
    _offered(open_terminal_options(page, name), NOT_RUNNING)
