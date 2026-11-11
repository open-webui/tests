"""Journey: a model's system prompt is filled in for the person chatting, with their own values.

A fresh admin writes a system prompt with variables in the workspace's model editor, as the
models docs page describes them. The browser fills the person's name, email, language, time zone,
weekday and date as it sees them, so a person in Auckland gets Auckland's day; the server fills
the groups the person belongs to, inherited ones included. A user variable that a person saves
under Settings > Account (`{{user.variables.<key>}}`) reaches the model in that person's chats
only, and is empty for someone who never saved it. With User Location on, the browser's
position fills `{{USER_LOCATION}}`, and the message does not wait for the page to save that
position on the server (3e6f8b7f4). Chat variables are covered in
e2e/chat/test_chat_variables.py, placing the system prompt in
e2e/workspace/test_workspace_presets.py.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose chat sends no
variables with the message the browser test fails (the time zone and language stay literal); in a
backend copy that renders user variables from an empty set the user variable test fails, and in
one that leaves the groups out the group test fails. The location test passes on dev
f659c0d12 and fails in a frontend copy with 3e6f8b7f4 undone (the message waits for a save
that never answers and is never sent).

The browser test also runs in every way a side chat starts and every state it can be in
(`utils/chat_surface.py`), the side chat put on the preset in its own model selector. A side chat
sent the time zone and language as literal placeholders and the server's date
(open-webui/open-webui#34032, fix PR open-webui/open-webui#34064), so its side chat variants are
red on dev c15f28649. A side chat beside a chat adds its side conversation instruction after the
prompt, so the first line of the system message is read.

On dev c15f28649 and e791cbdde the tests that write the prompt in the model editor are red before
any chat opens: after Save & Update the editor asks to discard the changes it just saved and stays
open (issue draft pending). With that dialog confirmed by hand the browser test passes in the
main chat on both builds, its side chat variants fail on c15f28649 (the language and time zone
stay literal) and pass on e791cbdde.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Page, expect

from harness.access import make_group
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import chat_input, start_from_link
from utils.model_editor import open_chat_on, open_editor, save, sent_request
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

AUCKLAND = "Pacific/Auckland"
VIENNA = {"latitude": 48.20849, "longitude": 16.37208}


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def preset(builder) -> dict:
    """A preset on the scripted model that the builder made over the API, every account reads."""
    model = {
        "id": f"greeter-{uuid.uuid4().hex[:8]}",
        "name": f"Harbour greeter {uuid.uuid4().hex[:6]}",
    }
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    return model


def write_system_prompt(page: Page, model: dict, system_prompt: str) -> None:
    editor = open_editor(page, model)
    editor.get_by_role("textbox", name="System Prompt", exact=True).fill(system_prompt)
    save(editor)


def system_message(page: Page, upstream, model: dict, question: str) -> str:
    open_chat_on(page, model)
    messages = sent_request(page, upstream, question)["messages"]
    assert messages[0]["role"] == "system", messages
    return messages[0]["content"]


def system_message_in(chat: ChatSurface, upstream, model: dict, question: str) -> str:
    """The system message `chat` sends on `model`: the main chat opens the model's link."""
    start_from_link(chat, f"/?model={model['id']}")
    expect(chat.button(f"Selected model: {model['name']}")).to_be_visible()
    expect(chat_input(chat)).to_be_visible()
    messages = sent_request(chat, upstream, question)["messages"]
    assert messages[0]["role"] == "system", messages
    return messages[0]["content"]


def today_in(zone: str) -> str:
    return datetime.now(ZoneInfo(zone)).strftime("%A %Y-%m-%d")


# one model: the values come from the browser, whichever model the prompt belongs to
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_the_browser_fills_who_the_person_is_and_their_day(
    page_for, chat_for, builder, preset, upstream
):
    """In a side chat: open-webui/open-webui#34032. Saving the model: issue draft pending."""
    write_system_prompt(
        page_for(builder),
        preset,
        "You greet {{USER_NAME}} <{{USER_EMAIL}}>, who reads {{USER_LANGUAGE}}, "
        "in {{CURRENT_TIMEZONE}} on {{CURRENT_WEEKDAY}} {{CURRENT_DATE}}.",
    )
    day_before = today_in(AUCKLAND)

    chat = chat_for(builder, timezone_id=AUCKLAND)
    filled = system_message_in(chat, upstream, preset, "good morning").split("\n")[0]

    greeting = f"You greet {builder.name} <{builder.email}>, who reads en-US, in {AUCKLAND} on "
    assert filled.startswith(greeting), filled
    day = filled.removeprefix(greeting).removesuffix(".")
    # the date may turn while the chat is sent
    assert day in {day_before, today_in(AUCKLAND)}, f"{day!r} is not today in {AUCKLAND}"


def group_name(admin, group_id: str) -> str:
    with admin.client() as client:
        return client.get(f"/api/v1/groups/id/{group_id}").json()["name"]


def test_the_groups_of_the_person_chatting_fill_the_system_prompt(
    page_for, admin, builder, preset, make_user, upstream
):
    """Saving the model: issue draft pending."""
    sailor, visitor = make_user(), make_user()
    staff_id = make_group(admin, [])
    crew_id = make_group(admin, [sailor], parent_id=staff_id)
    write_system_prompt(page_for(builder), preset, "Groups of the caller: {{USER_GROUPS}}.")

    filled = system_message(page_for(sailor), upstream, preset, "may I board?")
    groups = filled.removeprefix("Groups of the caller: ").removesuffix(".").split(", ")
    assert sorted(groups) == sorted([group_name(admin, crew_id), group_name(admin, staff_id)])
    assert system_message(page_for(visitor), upstream, preset, "may I board too?") == (
        "Groups of the caller: ."
    )


def save_user_variable(page: Page, key: str, value: str) -> None:
    """Add a user variable under Settings > Account and save the tab."""
    expect(chat_input(page)).to_be_visible()
    open_settings(page)
    page.get_by_role("tab", name="Account").click()
    settings = page.get_by_role("dialog").first
    settings.get_by_role("button", name="Add", exact=True).click()
    page.get_by_label("Variable key").fill(key)
    page.get_by_label("Variable value").fill(value)
    page.get_by_role("button", name="Done", exact=True).click()
    expect(settings.get_by_text(value)).to_be_visible()
    with page.expect_response(lambda response: "/user/variables/update" in response.url):
        settings.get_by_role("button", name="Save", exact=True).click()
    page.keyboard.press("Escape")


def test_a_user_variable_saved_in_the_account_settings_reaches_only_that_persons_chats(
    page_for, builder, preset, make_user, upstream
):
    """Saving the model: issue draft pending."""
    write_system_prompt(page_for(builder), preset, "Moor at {{user.variables.home_port}}.")
    skipper, deckhand = make_user(), make_user()
    skipper_page = page_for(skipper)
    save_user_variable(skipper_page, "home_port", "Kirkwall")

    assert system_message(skipper_page, upstream, preset, "where do I moor?") == (
        "Moor at Kirkwall."
    )
    assert system_message(page_for(deckhand), upstream, preset, "where do I moor too?") == (
        "Moor at ."
    )


# the browser offers the location only in a secure context; over plain HTTP it reads Unknown
@pytest.mark.secure_context
def test_the_persons_location_fills_the_prompt_without_waiting_for_it_to_be_saved(
    page_for, builder, make_user, upstream
):
    model = {"id": f"pilot-{uuid.uuid4().hex[:8]}", "name": f"Harbour pilot {uuid.uuid4().hex[:6]}"}
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {"system": "Guide the ship from {{USER_LOCATION}}."},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    sailor = make_user()
    with sailor.client() as client:
        shared = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"userLocation": True}}
        )
    assert shared.status_code == 200, shared.text
    page = page_for(sailor, geolocation=VIENNA, permissions=["geolocation"])
    # the server never answers the page's save of the location
    saving: list = []
    page.route("**/api/v1/users/user/info/update", lambda route: saving.append(route))

    filled = system_message(page, upstream, model, "which way to the harbour?")

    assert filled == "Guide the ship from 48.208, 16.372 (lat, long).", filled
    assert saving, "the page never saved the location, so nothing showed it does not wait for it"
