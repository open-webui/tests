"""Journey: the chat's Terminal menu opens a terminal's System Info with what the terminal reports.

New on dev for 0.13.1 (7d4dda270, 36d049490, 8a8e42d17, d49756cf1, titled `refac`). Each terminal
in the Terminal menu has a More options button whose menu holds System info (a button of its own
on the row until f4cf2ca0c), which opens a window reading the terminal's `system/info` through
the instance: host name, CPU, memory and disk use, cores and the busiest
processes, its uptime and, folded under Network, each interface with its addresses. When the
terminal cannot answer, the window says System info unavailable with the reason. A terminal
that gives every chat its own environment offers no System info in a chat that is not saved yet,
as it has nothing to report there, and offers it once a message has saved the chat. A local fake
terminal answers with fixed numbers; the per-chat terminal is one scoped to the chat on a real
Terminals orchestrator on Docker (`harness.terminals`, since 6f2fc7bc7), and its test skips
without it.

Discriminates: passes on dev f659c0d12; with the System info button removed from the menu in a
frontend build every test fails; on the integration side, test_terminal_system_info.py pins the
proxy's refusal for an unsaved per-chat terminal.
The uptime and Network checks pass on dev e8c0ad85e and fail on a copy of the dev e8c0ad85e build
whose window labels the uptime otherwise and lists no interface addresses.
The per-chat test passes on the dev c15f28649 build and fails with the terminal list no longer
telling the browser a terminal's scope, in a backend copy (More options, with System info, is
offered before the chat is saved).
Retargeted at the More options menu on dev c15f28649: both window tests were red there on the old
button and pass again; with the instance answering 404 for `system/info` in a backend copy the
reports test fails (the window says System info unavailable). The cannot-answer test has no
backend mutation, a refusal being what it expects; the frontend proofs above still hold for it,
the window being reached through the one System info entry.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.terminals import create_template, share_template
from utils.cached_chat import open_terminal_options
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

GIB = 1024**3
SYSTEM_INFO = {
    "hostname": "lab-host-7",
    "os": "Linux",
    "arch": "x86_64",
    "environment": "host",
    "cpu": {"percent": 12.5, "cores": 6, "scope": "host"},
    "memory": {"total": 16 * GIB, "used": 4 * GIB, "scope": "host"},
    "disk": {"path": "/workspace", "total": 100 * GIB, "used": 25 * GIB, "free": 75 * GIB},
    "network": [{"name": "eth0", "up": True, "addresses": ["10.1.2.3"]}],
    "processes": [
        {"pid": 4242, "name": "busy-worker", "cpu_percent": 87.5, "memory_bytes": 512 * 1024**2}
    ],
    "process_scope": "environment",
    "uptime_seconds": 7200,
    "service_uptime_seconds": 600,
    "sampled_at": 1_800_000_000,
}


@pytest.fixture
def fake_terminal():
    with serving_terminal() as server:
        server.route("GET", "/api/config", json_answer({"features": {}}))
        yield server


def _connect(admin, preserve, server, person, **fields) -> str:
    preserve(TERMINAL_SERVERS_CONFIG)
    name = f"Lab box {uuid.uuid4().hex[:6]}"
    connection = server.connection(name=name, **fields)
    connection["config"]["access_grants"] = [read_grant(person.id)]
    with admin.client() as client:
        configure_terminals(client, connection)
    return name


def _terminal_menu(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def _open_system_info(page: Page, name: str) -> Locator:
    options = open_terminal_options(page, name)
    options.get_by_role("menuitem", name="System info", exact=True).click()
    dialog = page.get_by_role("dialog", name=f"System info for {name}")
    expect(dialog).to_be_visible()
    return dialog


def test_system_info_shows_what_the_terminal_reports(
    page_for, make_user, admin, preserve, fake_terminal
):
    fake_terminal.route("GET", "/system/info", json_answer(SYSTEM_INFO))
    person = make_user()
    name = _connect(admin, preserve, fake_terminal, person)
    page = page_for(person)

    dialog = _open_system_info(page, name)

    expect(dialog).to_contain_text("lab-host-7")
    expect(dialog.get_by_role("progressbar", name="CPU")).to_have_attribute(
        "aria-valuetext", "12.5%"
    )
    expect(dialog.get_by_role("progressbar", name="Memory")).to_have_attribute(
        "aria-valuetext", "4.0 GiB / 16.0 GiB"
    )
    expect(dialog.get_by_role("progressbar", name="Disk")).to_have_attribute(
        "aria-valuetext", "25.0 GiB / 100.0 GiB"
    )
    expect(dialog).to_contain_text("6 cores")
    expect(dialog.get_by_role("table")).to_contain_text("busy-worker")
    expect(dialog).to_contain_text("Uptime: 2h 0m")
    dialog.get_by_text("Network", exact=True).click()
    network = dialog.get_by_role("definition").filter(has_text="10.1.2.3")
    expect(network).to_be_visible()
    expect(dialog.get_by_role("term").filter(has_text="eth0")).to_be_visible()
    assert fake_terminal.requests_to("/system/info"), "the window never asked the terminal"


def test_system_info_says_when_the_terminal_cannot_answer(
    page_for, make_user, admin, preserve, fake_terminal
):
    fake_terminal.route("GET", "/system/info", (500, {"Content-Type": "text/plain"}, b"down"))
    person = make_user()
    name = _connect(admin, preserve, fake_terminal, person)
    page = page_for(person)

    dialog = _open_system_info(page, name)

    expect(dialog.get_by_role("alert")).to_contain_text("System info unavailable")
    expect(dialog.get_by_role("progressbar")).to_have_count(0)


def test_a_per_chat_terminal_offers_no_system_info_before_the_chat_is_saved(
    page_for, make_user, admin, connect_terminals, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:6]
    per_user, per_chat = f"Own box {tag}", f"Per chat box {tag}"
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        for name, scope in ((per_user, "user"), (per_chat, "chat")):
            terminal_id = create_template(client, connection, name=name, scope=scope)["terminal_id"]
            share_template(client, terminal_id, read_grant(person.id))
    page = page_for(person)

    menu = _terminal_menu(page)

    expect(menu.get_by_role("menuitemradio", name=per_chat)).to_be_visible()
    expect(menu.get_by_role("menuitem", name=f"More options for {per_user}")).to_have_count(1)
    expect(menu.get_by_role("menuitem", name=f"More options for {per_chat}")).to_have_count(0)

    # once a message has saved the chat, the chat has a machine to report on
    menu.get_by_role("menuitemradio", name=per_chat).click()
    expect(page.get_by_role("menu")).to_have_count(0)
    prompt = f"which box is this {tag}"
    upstream.queue(reply.text("Yours.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Yours.")
    page.get_by_role("main").get_by_role("button", name=per_chat, exact=True).click()
    # a terminal's More options show once its row is pointed at
    page.get_by_role("menu").get_by_role("menuitemradio", name=per_chat).hover()
    page.get_by_role("menuitem", name=f"More options for {per_chat}").click()
    expect(page.get_by_role("menuitem", name="System info", exact=True)).to_be_visible()
