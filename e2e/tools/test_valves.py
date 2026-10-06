"""Journey: valves set in the browser reach the tool or function the next chat runs.

An admin opens a tool's Valves from the workspace tool list, finds every field on its default,
switches each one to a custom value (text, number, switch, choice list, option list and a
password) and saves; the tool the scripted model calls next returns exactly those values, and
the dialog shows them again after a reload. On the admin function list a pipe's valves are
saved the same way, and a valve switched back to Default runs with its default again. Users
set their own user valves from a chat, through the knob beside a tool in the Integrations menu
or the Valves section of the chat controls; the tool sees each user's own value, and the
default until a user saves one. A list user valve in the chat controls is saved the way the
valves dialog saves it (open-webui/open-webui#31300, issue #31299): left unset it keeps its
default when another valve is saved, a typed list loses its empty entries and later edits
still save.

Discriminates: passes on dev 176d31d1d; in a frontend copy, the valves dialog saving an empty
form turned the tool, pipe, reset and Integrations menu tests red and left the chat controls test
green, while the Default button leaving a custom value in place together with the chat controls
never saving turned only the reset and chat controls tests red. With `be35c8f65` reverted (the
015dbc861 mutation build) both list valve tests go red.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.plugins import installed_function
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button
from utils.valves import customise, valve

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def source(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


FRONT_DESK_TOOL = source(
    """
    import json
    from typing import Literal

    from pydantic import BaseModel, Field

    class Tools:
        class Valves(BaseModel):
            greeting: str = Field("hello", description="What the desk says first")
            max_rooms: int = Field(3, description="How many rooms to list")
            include_closed: bool = Field(False, description="List closed rooms too")
            mood: Literal["calm", "cheerful", "stern"] = Field(
                "calm", description="How the desk sounds"
            )
            region: str = Field(
                "eu-west",
                description="Where the rooms are",
                json_schema_extra={"input": {"type": "select", "options": ["eu-west", "us-east"]}},
            )
            api_key: str = Field(
                "",
                description="Key for the booking service",
                json_schema_extra={"input": {"type": "password"}},
            )

        def __init__(self):
            self.valves = self.Valves()

        def desk_settings(self) -> str:
            \"\"\"Report how the front desk is set up.\"\"\"
            return json.dumps(self.valves.model_dump(), sort_keys=True)
    """
)

NAMING_TOOL = source(
    """
    from pydantic import BaseModel, Field

    class Tools:
        class UserValves(BaseModel):
            nickname: str = Field("guest", description="What the tool calls you")

        def whoami(self, __user__: dict) -> str:
            \"\"\"Say what the tool calls the user.\"\"\"
            return f"calling you {__user__['valves'].nickname}"
    """
)

BOOKING_TOOL = source(
    """
    import json

    from pydantic import BaseModel, Field

    class Tools:
        class UserValves(BaseModel):
            nickname: str = Field("guest", description="What the tool calls you")
            rooms: list[str] = Field(["lobby"], description="Rooms you may book")

        def booking(self, __user__: dict) -> str:
            \"\"\"Say what the user may book.\"\"\"
            valves = __user__["valves"]
            return json.dumps({"nickname": valves.nickname, "rooms": valves.rooms})
    """
)

GREETING_PIPE = source(
    """
    from pydantic import BaseModel, Field

    class Pipe:
        class Valves(BaseModel):
            greeting: str = Field("hello", description="What the pipe says first")
            rooms: int = Field(1, description="How many rooms are free")
            open: bool = Field(False, description="Whether the desk is open")

        def __init__(self):
            self.valves = self.Valves()

        def pipe(self, body):
            state = "open" if self.valves.open else "closed"
            return f"{self.valves.greeting}, {self.valves.rooms} rooms, {state}"
    """
)


def tool_results(upstream, question: str) -> list[str]:
    answered = [
        request for request in upstream.chat_requests() if reply.answering(question)(request)
    ]
    return [entry["content"] for entry in answered[-1]["messages"] if entry["role"] == "tool"]


def turn_on_tool(page: Page, tool_name: str) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    page.get_by_role("button", name=tool_name).click()
    page.keyboard.press("Escape")


def ask_tool(page: Page, upstream, tool_name: str, function_name: str, question: str) -> None:
    upstream.queue(
        reply.tool_call(function_name, {}, match=reply.answering(question)),
        reply.text(f"answered {question}", match=reply.answering(question)),
    )
    page.goto("/")
    turn_on_tool(page, tool_name)
    send(page, question)
    expect_reply(page, f"answered {question}")


@pytest.fixture
def desk_admin(make_user):
    """A fresh admin; the tools it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")


def _create_tool(owner, name: str, content: str, access_grants: list[dict] | None = None) -> str:
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    with owner.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": name,
                "content": content,
                "meta": {"description": "installed by a regression test"},
                "access_grants": access_grants or [],
            },
        )
    assert created.status_code == 200, created.text
    return tool_id


def open_tool_valves(page: Page, name: str) -> Locator:
    page.goto("/workspace/tools")
    page.get_by_role("textbox", name="Search Tools").fill(name)
    card = page.get_by_role("main").get_by_role("button", name=re.compile(name))
    card.get_by_role("button", name="Valves").click()
    return page.get_by_role("dialog").filter(has_text="Valves")


def test_tool_valves_of_every_field_type_are_saved_and_reach_the_tool(
    page_for, desk_admin, upstream
):
    name = f"Front desk {uuid.uuid4().hex[:6]}"
    _create_tool(desk_admin, name, FRONT_DESK_TOOL)
    page = page_for(desk_admin)
    dialog = open_tool_valves(page, name)
    expect(dialog.get_by_role("button", name="Default")).to_have_count(6)
    expect(dialog.get_by_role("textbox")).to_have_count(0)

    greeting = valve(dialog, "Greeting", "What the desk says first")
    customise(greeting)
    expect(greeting.get_by_role("textbox")).to_have_value("hello")
    greeting.get_by_role("textbox").fill("welcome back")
    max_rooms = valve(dialog, "Max Rooms", "How many rooms to list")
    customise(max_rooms)
    max_rooms.get_by_role("textbox").fill("12")
    include_closed = valve(dialog, "Include Closed", "List closed rooms too")
    customise(include_closed)
    include_closed.get_by_role("switch").click()
    expect(include_closed.get_by_text("Enabled")).to_be_visible()
    mood = valve(dialog, "Mood", "How the desk sounds")
    customise(mood)
    mood.get_by_role("combobox").select_option("stern")
    region = valve(dialog, "Region", "Where the rooms are")
    customise(region)
    region.get_by_role("combobox").select_option("us-east")
    api_key = valve(dialog, "Api Key", "Key for the booking service")
    customise(api_key)
    api_key.get_by_label("Key for the booking service").fill("sk-desk-42")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Valves updated successfully")).to_be_visible()

    question = "how is the desk set up?"
    ask_tool(page, upstream, name, "desk_settings", question)
    saved = {
        "api_key": "sk-desk-42",
        "greeting": "welcome back",
        "include_closed": True,
        "max_rooms": 12,
        "mood": "stern",
        "region": "us-east",
    }
    assert [json.loads(result) for result in tool_results(upstream, question)] == [saved]

    page.reload()
    dialog = open_tool_valves(page, name)
    expect(dialog.get_by_role("button", name="Custom")).to_have_count(6)
    greeting = valve(dialog, "Greeting", "What the desk says first")
    expect(greeting.get_by_role("textbox")).to_have_value("welcome back")
    mood = valve(dialog, "Mood", "How the desk sounds")
    expect(mood.get_by_role("combobox")).to_have_value("stern")


def open_function_valves(page: Page, function_id: str) -> Locator:
    page.goto("/admin/functions")
    page.get_by_placeholder("Search Functions").fill(function_id)
    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^pipe {function_id}"))
    card.get_by_role("button", name="Valves").click()
    return page.get_by_role("dialog").filter(has_text="Valves")


def ask_pipe(page: Page, pipe_id: str, question: str, answer: str) -> None:
    page.goto(f"/?models={pipe_id}")
    send(page, question)
    expect_reply(page, answer)


def test_pipe_valves_saved_from_the_function_list_shape_its_next_reply(page_for, admin):
    with installed_function(admin, GREETING_PIPE) as pipe_id:
        page = page_for(admin)
        ask_pipe(page, pipe_id, "is the desk open?", "hello, 1 rooms, closed")

        dialog = open_function_valves(page, pipe_id)
        greeting = valve(dialog, "Greeting", "What the pipe says first")
        customise(greeting)
        greeting.get_by_role("textbox").fill("good evening")
        rooms = valve(dialog, "Rooms", "How many rooms are free")
        customise(rooms)
        rooms.get_by_role("textbox").fill("4")
        is_open = valve(dialog, "Open", "Whether the desk is open")
        customise(is_open)
        is_open.get_by_role("switch").click()
        dialog.get_by_role("button", name="Save").click()
        expect(page.get_by_text("Valves updated successfully")).to_be_visible()

        ask_pipe(page, pipe_id, "is the desk open now?", "good evening, 4 rooms, open")


def test_a_valve_switched_back_to_default_runs_with_its_default(page_for, admin):
    with installed_function(admin, GREETING_PIPE) as pipe_id:
        with admin.client() as client:
            saved = client.post(
                f"/api/v1/functions/id/{pipe_id}/valves/update",
                json={"greeting": "good evening", "rooms": 4},
            )
        assert saved.status_code == 200, saved.text
        page = page_for(admin)
        dialog = open_function_valves(page, pipe_id)
        greeting = valve(dialog, "Greeting", "What the pipe says first")
        greeting.get_by_role("button", name="Custom").click()
        expect(greeting.get_by_role("button", name="Default")).to_be_visible()
        expect(greeting.get_by_role("textbox")).to_have_count(0)
        dialog.get_by_role("button", name="Save").click()
        expect(page.get_by_text("Valves updated successfully")).to_be_visible()

        ask_pipe(page, pipe_id, "who greets me?", "hello, 4 rooms, closed")


@pytest.fixture
def naming_tool(admin, make_user):
    """A tool with a user valve, shared for reading with two fresh users."""
    first, second = make_user(), make_user()
    name = f"Name tag {uuid.uuid4().hex[:6]}"
    grants = [grant("user", first.id, "read"), grant("user", second.id, "read")]
    tool_id = _create_tool(admin, name, NAMING_TOOL, grants)
    yield name, first, second
    with admin.client() as client:
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def set_nickname_from_integrations(page: Page, tool_name: str, nickname: str) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    tool_row = page.get_by_role("button", name=tool_name)
    tool_row.hover()
    tooltip_button(tool_row, "Valves").click()
    dialog = page.get_by_role("dialog").filter(has_text="Valves")
    nickname_valve = valve(dialog, "Nickname", "What the tool calls you")
    customise(nickname_valve)
    nickname_valve.get_by_role("textbox").fill(nickname)
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Valves updated successfully")).to_be_visible()


def test_each_user_sets_their_own_user_valves_from_the_integrations_menu(
    page_for, naming_tool, upstream
):
    name, first, second = naming_tool
    first_page, second_page = page_for(first), page_for(second)
    set_nickname_from_integrations(first_page, name, "Captain")
    set_nickname_from_integrations(second_page, name, "Doc")

    ask_tool(first_page, upstream, name, "whoami", "what do you call me?")
    ask_tool(second_page, upstream, name, "whoami", "which name is mine?")

    assert tool_results(upstream, "what do you call me?") == ["calling you Captain"]
    assert tool_results(upstream, "which name is mine?") == ["calling you Doc"]


def open_chat_controls_valves(page: Page, tool_name: str) -> Locator:
    page.get_by_role("button", name="Controls").click()
    page.get_by_role("button", name="Valves").click()
    tool_picker = page.get_by_role("combobox").filter(
        has=page.get_by_role("option", name=tool_name)
    )
    tool_picker.select_option(label=tool_name)
    return page.locator("form").filter(has=tool_picker)


def test_user_valves_set_in_the_chat_controls_replace_the_default(page_for, naming_tool, upstream):
    name, first, _ = naming_tool
    page = page_for(first)
    ask_tool(page, upstream, name, "whoami", "who am I to you?")

    panel = open_chat_controls_valves(page, name)
    nickname_valve = valve(panel, "Nickname", "What the tool calls you")
    customise(nickname_valve)
    nickname_valve.get_by_role("textbox").fill("Skipper")
    nickname_valve.get_by_role("textbox").press("Tab")
    expect(page.get_by_text("Valves updated", exact=True)).to_be_visible()

    ask_tool(page, upstream, name, "whoami", "who am I to you now?")
    assert tool_results(upstream, "who am I to you?") == ["calling you guest"]
    assert tool_results(upstream, "who am I to you now?") == ["calling you Skipper"]


@pytest.fixture
def booking_tool(admin, make_user):
    """A tool with a text and a list user valve, shared for reading with a fresh user."""
    booker = make_user()
    name = f"Booking {uuid.uuid4().hex[:6]}"
    tool_id = _create_tool(admin, name, BOOKING_TOOL, [grant("user", booker.id, "read")])
    yield name, tool_id, booker
    with admin.client() as client:
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def booking_result(upstream, question: str) -> dict:
    return json.loads(tool_results(upstream, question)[0])


def type_into(row: Locator, text: str) -> None:
    # the panel saves a moment after a field changes
    row.get_by_role("textbox").fill(text)
    row.get_by_role("textbox").press("Tab")


def saved_user_valves(account, tool_id: str, expected: dict) -> dict:
    """The account's stored user valves, polled until they read `expected` or 10 seconds pass."""
    deadline = time.monotonic() + 10
    while True:
        with account.client() as client:
            saved = client.get(f"/api/v1/tools/id/{tool_id}/valves/user").json()
        if saved == expected or time.monotonic() > deadline:
            return saved
        time.sleep(0.2)


def test_an_unset_list_valve_keeps_its_default_when_the_chat_controls_save_another(
    page_for, booking_tool, upstream
):
    name, tool_id, booker = booking_tool
    page = page_for(booker)
    ask_tool(page, upstream, name, "booking", "what may I book?")

    panel = open_chat_controls_valves(page, name)
    rooms_valve = valve(panel, "Rooms", "Rooms you may book")
    expect(
        rooms_valve.get_by_role("button", name="Default"), "the unset list shows as Custom (#31300)"
    ).to_be_visible()
    nickname_valve = valve(panel, "Nickname", "What the tool calls you")
    customise(nickname_valve)
    type_into(nickname_valve, "Skipper")
    expect(page.get_by_text("Valves updated", exact=True)).to_be_visible()

    ask_tool(page, upstream, name, "booking", "what may I book now?")
    assert booking_result(upstream, "what may I book now?") == {
        "nickname": "Skipper",
        "rooms": ["lobby"],
    }, "saving another valve replaced the list's default (#31300)"


def test_a_list_valve_set_in_the_chat_controls_drops_empty_entries_and_keeps_saving(
    page_for, booking_tool, upstream
):
    name, tool_id, booker = booking_tool
    with booker.client() as client:
        seeded = client.post(
            f"/api/v1/tools/id/{tool_id}/valves/user/update", json={"rooms": ["lobby", "hall"]}
        )
    assert seeded.status_code == 200, seeded.text
    page = page_for(booker)
    ask_tool(page, upstream, name, "booking", "which rooms are mine?")

    panel = open_chat_controls_valves(page, name)
    rooms_valve = valve(panel, "Rooms", "Rooms you may book")
    expect(rooms_valve.get_by_role("textbox")).to_have_value("lobby,hall")
    with page.expect_response(lambda response: "/valves/user/update" in response.url):
        type_into(rooms_valve, "north, south, ")
    nickname_valve = valve(panel, "Nickname", "What the tool calls you")
    customise(nickname_valve)
    type_into(nickname_valve, "Skipper")

    expected = {"nickname": "Skipper", "rooms": ["north", "south"]}
    saved = saved_user_valves(booker, tool_id, expected)
    assert saved == expected, f"the list or a later edit was saved wrongly (#31300): {saved}"
    ask_tool(page, upstream, name, "booking", "which rooms are mine now?")
    assert booking_result(upstream, "which rooms are mine now?") == expected
