"""Journey: a model's system prompt is filled in for the person chatting, with their own values.

A fresh admin writes a system prompt with variables in the workspace's model editor, as the
models docs page describes them. The browser fills the person's name, email, language, time zone,
weekday and date as it sees them, so a person in Auckland gets Auckland's day; the server fills
the groups the person belongs to, inherited ones included. A user variable that a person saves
under Settings > Account (`{{user.variables.<key>}}`) reaches the model in that person's chats
only, and is empty for someone who never saved it. Chat variables are covered in
e2e/chat/test_chat_variables.py, placing the system prompt in
e2e/workspace/test_workspace_presets.py.

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose chat sends no
variables with the message the browser test fails (the time zone and language stay literal); in a
backend copy that renders user variables from an empty set the user variable test fails, and in
one that leaves the groups out the group test fails.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Page, expect

from harness.access import make_group
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.model_editor import open_chat_on, open_editor, save, sent_request

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

AUCKLAND = "Pacific/Auckland"


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
    editor.get_by_role("textbox", name=re.compile("^Write your model system prompt")).fill(
        system_prompt
    )
    save(editor)


def system_message(page: Page, upstream, model: dict, question: str) -> str:
    open_chat_on(page, model)
    messages = sent_request(page, upstream, question)["messages"]
    assert messages[0]["role"] == "system", messages
    return messages[0]["content"]


def today_in(zone: str) -> str:
    return datetime.now(ZoneInfo(zone)).strftime("%A %Y-%m-%d")


def test_the_browser_fills_who_the_person_is_and_their_day(page_for, builder, preset, upstream):
    write_system_prompt(
        page_for(builder),
        preset,
        "You greet {{USER_NAME}} <{{USER_EMAIL}}>, who reads {{USER_LANGUAGE}}, "
        "in {{CURRENT_TIMEZONE}} on {{CURRENT_WEEKDAY}} {{CURRENT_DATE}}.",
    )
    day_before = today_in(AUCKLAND)

    page = page_for(builder, timezone_id=AUCKLAND)
    filled = system_message(page, upstream, preset, "good morning")

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
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("button", name="Settings").click()
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
