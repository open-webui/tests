"""Journey: a person's own device shows under "Your devices" and the model works on it.

A device connected through Tether is a terminal of its owner alone. The chat input's Terminal
menu lists it under the heading "Your devices", below the terminals an admin connected. The
person picks it and asks the model to run a command: the call travels to the device and its
output goes back to the model. The heading and the device appear in the open page when the
device connects, follow a new name, and go when the device disconnects; a person without a
device sees no such heading.

The device is the fake of `harness/tether_device.py`; the admin's terminal is the fake terminal
server.

Discriminates: passes on the dev c15f28649 build. In a backend copy, with every relayed call
sent to `/` the device has no `run_command` to offer and the first test fails; with no
`terminals:changed` sent when a device connects, changes or goes, the second test fails (the
open page never shows the device).
"""

from __future__ import annotations

import json
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
from harness.tether_device import DeviceRequest, TetherDevice
from harness.upstream import conversation
from utils.cached_chat import ask, calling
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DEVICES = "Your devices"
COMMAND = {"type": "object", "properties": {"command": {"type": "string"}}}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {
                "operationId": "run_command",
                "requestBody": {"content": {"application/json": {"schema": COMMAND}}},
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


def _serve_terminal(device: TetherDevice) -> None:
    def ran(request: DeviceRequest):
        return json_answer({"output": f"ran {request.json()['command']} on the device"})

    device.route("GET", "/openapi.json", json_answer(SPEC))
    device.route("GET", "/files/cwd", json_answer({"cwd": "/home/user", "home": "/home/user"}))
    device.route("POST", "/execute", ran)


def _open_terminal_menu(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Terminals", exact=True)).to_be_visible()
    return menu


def _close_terminal_menu(page: Page) -> None:
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)


def _under_your_devices(menu: Locator) -> list[str]:
    """What the menu shows below the heading "Your devices", as a person reads it."""
    lines = [line.strip() for line in menu.inner_text().splitlines() if line.strip()]
    headings = [index for index, line in enumerate(lines) if line.lower() == DEVICES.lower()]
    assert len(headings) == 1, f"the menu does not show the heading {DEVICES!r} once: {lines}"
    return lines[headings[0] + 1 :]


def test_a_person_picks_their_device_under_your_devices_and_the_model_runs_a_command_on_it(
    page_for, make_user, connect_device, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Laptop {tag}"
    device = connect_device(person, name=name)
    _serve_terminal(device)
    prompt = f"build the docs {tag}"

    page = page_for(person)
    menu = _open_terminal_menu(page)
    expect(menu.get_by_text(DEVICES, exact=True)).to_be_visible()
    assert _under_your_devices(menu) == [name], (
        f"under {DEVICES!r} the menu shows {_under_your_devices(menu)}"
    )
    menu.get_by_role("menuitemradio", name=name).click()
    expect(page.get_by_role("main").get_by_role("button", name=name, exact=True)).to_be_visible()
    expect(page.get_by_role("menu")).to_have_count(0)
    ask(
        page,
        upstream,
        prompt,
        calling("run_command", {"command": f"make {tag}"}),
        reply.text(f"Built {tag}."),
    )

    answered = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    results = [entry["content"] for entry in conversation(answered[-1]) if entry["role"] == "tool"]
    assert results, f"run_command never ran; the model was sent {conversation(answered[-1])}"
    assert json.loads(results[-1]) == {"output": f"ran make {tag} on the device"}, (
        f"the model was told {results[-1]} of its command"
    )
    [executed] = device.requests_to("/execute")
    assert executed.json() == {"command": f"make {tag}"}, executed.body
    assert executed.headers["x-user-id"] == person.id, executed.headers


def test_your_devices_follows_a_device_as_it_connects_takes_a_new_name_and_leaves(
    page_for, admin, make_user, connect_device, preserve
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    shared, first_name, new_name = f"Server {tag}", f"Laptop {tag}", f"Desk {tag}"
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        connection = server.connection(
            name=shared, config={"access_grants": [read_grant(person.id)]}
        )
        with admin.client() as client:
            configure_terminals(client, connection)

        page = page_for(person)
        menu = _open_terminal_menu(page)
        expect(menu.get_by_role("menuitemradio", name=shared)).to_be_visible()
        expect(menu.get_by_text(DEVICES, exact=True)).to_have_count(0)
        _close_terminal_menu(page)

        device = connect_device(person, name=first_name)
        menu = _open_terminal_menu(page)
        expect(menu.get_by_role("menuitemradio", name=first_name)).to_be_visible()
        assert _under_your_devices(menu) == [first_name], (
            f"under {DEVICES!r} the menu shows {_under_your_devices(menu)}"
        )
        _close_terminal_menu(page)

        device.announce(["open_terminal"], name=new_name)
        menu = _open_terminal_menu(page)
        expect(menu.get_by_role("menuitemradio", name=new_name)).to_be_visible()
        expect(menu.get_by_role("menuitemradio", name=first_name)).to_have_count(0)
        _close_terminal_menu(page)

        device.drop()
        menu = _open_terminal_menu(page)
        expect(menu.get_by_role("menuitemradio", name=shared)).to_be_visible()
        expect(menu.get_by_text(DEVICES, exact=True)).to_have_count(0)
        expect(menu.get_by_role("menuitemradio", name=new_name)).to_have_count(0)
