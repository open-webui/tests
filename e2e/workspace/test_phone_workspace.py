"""Journey: the workspace on a phone, from its lists to a model made in the model editor.

On a 390 by 844 touch screen the workspace opens from the sidebar drawer. Its section links scroll
sideways next to the Create button, so each of Models, Knowledge, Prompts, Skills and Tools can be
brought onto the screen and opens its own list with its search field on the screen. In the model
editor the name, the base model picker, the system prompt and Save & Create can all be reached and
tapped; the saved model is listed and answers a chat with its system prompt. A prompt made from the
Prompts list on the phone is listed with its command.

Discriminates: passes on the ebc6add67 build. In a frontend build of ebc6add67 whose workspace
section links do not scroll the lists test goes red; in a frontend build whose phone layout is 480
pixels wide (wider than the screen, so controls on the right fall off it) all three go red. In a
backend copy whose model create drops `params` the editor test goes red at the system prompt.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply
from utils.phone import (
    PHONE,
    SCREEN,
    expect_on_screen,
    expect_reachable,
    send_by_tapping,
    tap_on_screen,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SECTIONS = {
    "Models": "/workspace/models",
    "Knowledge": "/workspace/knowledge",
    "Prompts": "/workspace/prompts",
    "Skills": "/workspace/skills",
    "Tools": "/workspace/tools",
}


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models and prompts it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")


@pytest.fixture
def phone(page_for, builder) -> Page:
    page = page_for(builder, **PHONE)
    expect_on_screen(chat_input(page))
    return page


def open_workspace(page: Page) -> None:
    tap_on_screen(page.get_by_role("button", name="Open Sidebar").last)
    sidebar = page.get_by_role("navigation", name="Chat history")
    tap_on_screen(sidebar.get_by_role("link", name="Workspace"))
    expect(page).to_have_url(re.compile(r"/workspace/models"))


def expect_no_sideways_scroll(page: Page) -> None:
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= SCREEN["width"], f"the page scrolls sideways ({scroll_width}px wide)"


def choose_model(page: Page, name: str) -> None:
    search = page.get_by_role("textbox", name="Search In Models")
    expect_on_screen(search)
    search.fill(name)
    available = page.get_by_role("listbox", name="Available models")
    tap_on_screen(available.get_by_role("option", name=f"Select {name} model"))


def test_every_workspace_list_can_be_scrolled_onto_the_screen_and_opened(phone):
    open_workspace(phone)

    for name, path in SECTIONS.items():
        link = phone.get_by_role("link", name=re.compile(rf"^{name}"))
        expect_reachable(link)
        link.tap()
        expect(phone).to_have_url(re.compile(rf"{path}$"))
        expect_on_screen(phone.get_by_role("main").get_by_role("textbox").first)
        expect_on_screen(phone.get_by_role("main").get_by_role("button", name="Create").first)
        expect_no_sideways_scroll(phone)


def test_a_model_made_in_the_editor_on_a_phone_answers_with_its_system_prompt(phone, upstream):
    suffix = uuid.uuid4().hex[:6]
    name, system_prompt = f"Ranger {suffix}", f"Answer like a park ranger, trail {suffix}."
    open_workspace(phone)
    tap_on_screen(phone.get_by_role("main").get_by_role("button", name="Create", exact=True))
    expect(phone).to_have_url(re.compile(r"/workspace/models/create"))
    editor = phone.get_by_role("main")

    model_name = editor.get_by_role("textbox", name="Model Name")
    expect_reachable(model_name)
    model_name.fill(name)
    base_model = editor.get_by_role("button", name="Select a base model (e.g. llama3, gpt-4o)")
    expect_reachable(base_model)
    base_model.tap()
    choose_model(phone, MOCK_MODEL_ID)
    system = editor.get_by_role("textbox", name=re.compile("^Write your model system prompt"))
    expect_reachable(system)
    system.fill(system_prompt)
    expect_no_sideways_scroll(phone)
    save = editor.get_by_role("button", name="Save & Create")
    expect_reachable(save)
    save.tap()

    expect(phone).to_have_url(re.compile(r"/workspace/models$"))
    expect_on_screen(phone.get_by_role("main").get_by_text(name, exact=True).first)
    phone.goto("/")
    tap_on_screen(phone.get_by_role("button", name=re.compile("^Selected model")))
    choose_model(phone, name)
    upstream.queue(reply.text("Stay on the trail.", match=reply.answering("any advice")))
    send_by_tapping(phone, "any advice")
    expect_reply(phone, "Stay on the trail.")
    messages = next(filter(reply.answering("any advice"), upstream.chat_requests()))["messages"]
    assert messages[0] == {"role": "system", "content": system_prompt}, messages[0]


def test_a_prompt_made_on_a_phone_is_listed_with_its_command(phone):
    command = f"tidepool{uuid.uuid4().hex[:6]}"
    phone.goto("/workspace/prompts")
    tap_on_screen(phone.get_by_role("main").get_by_role("button", name="Create", exact=True))
    creating = phone.get_by_role("dialog").filter(has_text="Create Prompt")
    fields = {
        creating.get_by_role("textbox", name="Name", exact=True): f"Tide pools {command}",
        creating.get_by_role("textbox", name="Command"): command,
        creating.get_by_role(
            "textbox", name=re.compile("^Write a summary in 50 words")
        ): "Describe a tide pool.",
    }
    for field, text in fields.items():
        expect_reachable(field)
        field.fill(text)
    save = creating.get_by_role("button", name="Save & Create")
    expect_reachable(save)
    save.tap()

    expect_on_screen(phone.get_by_role("main").get_by_text(f"/{command}"))
