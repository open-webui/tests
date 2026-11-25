"""Journey: a person's own device shows under "Your devices" and the model works on it.

A device connected through Tether is a terminal of its owner alone. The chat input's Terminal
menu lists it under the heading "Your devices", below the terminals an admin connected. The
person picks it and asks the model to run a command: the call travels to the device and its
output goes back to the model. The heading and the device appear in the open page when the
device connects, follow a new name, and go when the device disconnects; a person without a
device sees no such heading. Connections the person added in the Tether app show as theirs: the
provider's model in the model selector, which answers through the device, the tool server
under "Your devices" in the Tools list and the terminal under "Your devices" in the Terminal
menu, each named "<device> · <connection>". MCP servers added in Tether, a local command and
one by its address, show under "Your devices" in the Tools list too, and the one the person
picks there gives the model its tool, which runs on the device. They show under "Tool Servers"
on dev f80a4f782, so the fourth test stays red until open-webui/open-webui#35049 is fixed
(fix branch Classic298:fix/tether-mcp-your-devices).

The device is the fake of `harness/tether_device.py`; the admin's terminal is the fake terminal
server.

Discriminates: passes on the dev c15f28649 build. In a backend copy, with every relayed call
sent to `/` the device has no `run_command` to offer and the first test fails; with no
`machines:changed` sent when a device connects, changes or goes, the second test fails (the
open page never shows the device). With the user's Tether connections never looked up (dev
485103cde) the third test fails (no tool server under "Your devices"). With a Tether MCP
server never connected in a chat the fourth test fails (the model is not offered `echo`);
on the dev f80a4f782 build it fails at its last step (the MCP servers under "Tool Servers").
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.mcp_server import STDIO_COMMAND
from harness.openapi_server import openapi_spec
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.tether_device import DeviceRequest, TetherDevice, own_connection
from harness.upstream import conversation
from utils.cached_chat import ask, calling, offered_tools
from utils.chat_ui import chat_input, close_menu, expect_reply, send
from utils.model_selector import select_model
from utils.tool_servers import open_tools_menu, pick_tool
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DEVICES = "Your devices"
WORKSPACE_TOOL = '''
class Tools:
    def jot(self, note: str) -> str:
        """Keep a note."""
        return note
'''
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
    tooltip_button(page.get_by_role("main"), "Machine").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Machines", exact=True)).to_be_visible()
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


def _streamed_completion(text: str) -> list[bytes]:
    """A chat completion as an OpenAI-compatible provider streams it."""

    def chunk(delta: dict, finish_reason: str | None) -> bytes:
        event = {
            "id": "chatcmpl-device",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "device-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(event)}\n\n".encode()

    return [
        chunk({"role": "assistant", "content": ""}, None),
        chunk({"content": text}, None),
        chunk({}, "stop"),
        b"data: [DONE]\n\n",
    ]


@contextlib.contextmanager
def _workspace_tool(admin, reader, name: str) -> Iterator[None]:
    """A workspace tool only `reader` may use, so the Tools list shows more than one group."""
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": name,
                "content": WORKSPACE_TOOL,
                "meta": {"description": name},
                "access_grants": [read_grant(reader.id)],
            },
        )
        assert created.status_code == 200, f"creating the tool {name} failed: {created.text}"
        try:
            yield
        finally:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def test_a_persons_own_model_tool_server_and_terminal_from_tether_show_in_the_chat(
    page_for, admin, make_user, connect_device
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name, own_model = f"Laptop {tag}", f"home-model-{tag}"
    weather, lab = f"{name} · Weather", f"{name} · Lab"
    device = connect_device(
        person,
        name=name,
        services=["home_llm", "home_tools", "lab_box"],
        connections=[
            own_connection("openai", "home_llm", "Home LLM"),
            own_connection("openapi", "home_tools", "Weather"),
            own_connection("terminal", "lab_box", "Lab"),
        ],
    )

    def completion(_request: DeviceRequest):
        return 200, {"Content-Type": "text/event-stream"}, _streamed_completion(f"Hi from {tag}.")

    device.route("GET", "/models", json_answer({"data": [{"id": own_model, "object": "model"}]}))
    device.route("POST", "/chat/completions", completion)
    device.route("GET", "/openapi.json", json_answer(openapi_spec(f"Weather {tag}", ["forecast"])))

    with _workspace_tool(admin, person, f"Notes {tag}"):
        page = page_for(person)
        tools = open_tools_menu(page)
        expect(tools.get_by_role("menuitemcheckbox", name=weather)).to_be_visible()
        expect(tools.get_by_role("menuitemcheckbox", name=f"Notes {tag}")).to_be_visible()
        assert _under_your_devices(tools) == [weather], (
            f"under {DEVICES!r} the Tools list shows {_under_your_devices(tools)}"
        )
        close_menu(page)

    menu = _open_terminal_menu(page)
    expect(menu.get_by_role("menuitemradio", name=lab)).to_be_visible()
    assert _under_your_devices(menu) == [lab], (
        f"under {DEVICES!r} the terminal menu shows {_under_your_devices(menu)}"
    )
    _close_terminal_menu(page)

    select_model(page, own_model)
    prompt = f"hello laptop {tag}"
    send(page, prompt)
    expect_reply(page, f"Hi from {tag}.")
    asked = [entry.json() for entry in device.requests_to("/chat/completions")]
    assert [body["model"] for body in asked] == [own_model], (
        f"the device's provider was asked for {[body['model'] for body in asked]}"
    )
    assert prompt in json.dumps(asked[0]["messages"]), asked[0]["messages"]


def test_mcp_servers_from_tether_show_under_your_devices_and_the_picked_one_serves_the_model(
    page_for, admin, make_user, connect_device, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Laptop {tag}"
    notes, wiki = f"{name} · Notes", f"{name} · Wiki"
    device = connect_device(
        person,
        name=name,
        services=["home_mcp", "wiki_mcp"],
        connections=[
            own_connection("mcp", "home_mcp", "Notes", transport="stdio"),
            own_connection("mcp", "wiki_mcp", "Wiki", transport="streamable-http"),
        ],
    )
    device.run_mcp("home_mcp", STDIO_COMMAND)
    echo = re.sub(r"[^a-zA-Z0-9_-]", "_", f"tether:{device.connection_id}:home_mcp_echo")
    prompt = f"take a note {tag}"

    page = page_for(person)
    pick_tool(page, notes)
    ask(page, upstream, prompt, calling(echo, {"text": f"note {tag}"}), reply.text(f"Noted {tag}."))

    answered = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    assert answered and echo in offered_tools(answered[0]), (
        f"the model was offered {sorted(offered_tools(answered[0])) if answered else 'nothing'}"
    )
    results = [entry["content"] for entry in conversation(answered[-1]) if entry["role"] == "tool"]
    assert results and f"note {tag}" in results[-1], f"the model was told {results} of its call"
    assert device.processes, "the device never launched the MCP server's command"

    with _workspace_tool(admin, person, f"Jots {tag}"):
        tools = open_tools_menu(page)
        expect(tools.get_by_role("menuitemcheckbox", name=wiki)).to_be_visible()
        expect(tools.get_by_role("menuitemcheckbox", name=f"Jots {tag}")).to_be_visible()
        assert sorted(_under_your_devices(tools)) == [notes, wiki], (
            f"under {DEVICES!r} the Tools list shows {_under_your_devices(tools)}"
        )
