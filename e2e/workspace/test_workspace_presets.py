"""Journey: a model preset and a prompt made in the workspace are used from the chat.

A fresh admin creates a preset on the scripted model with a system prompt, picks it in the chat's
model selector and sends a message; the provider is sent the preset's system prompt ahead of the
message. The admin also saves a prompt with a slash command; typing `/` and the command in the
chat offers it, picking it fills the input with its text, and that text is what gets sent.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the preset's system prompt left
out of the provider payload the preset test fails (no system message), and with
`/api/v1/prompts/create` answering without storing the prompt the prompt test fails (it never
shows in the list).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def builder(make_user):
    """A fresh admin; the presets and prompts it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")


def _choose_from_models(page: Page, name: str) -> None:
    # the list renders only the rows in view, so search to bring the model into it
    page.get_by_role("textbox", name="Search In Models").fill(name)
    available = page.get_by_role("listbox", name="Available models")
    available.get_by_role("option", name=f"Select {name} model").click()


def _pick_model(page: Page, name: str) -> None:
    page.goto("/")
    page.get_by_role("button", name=re.compile("^Selected model")).click()
    _choose_from_models(page, name)
    expect(page.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()


def test_a_preset_sends_its_system_prompt_with_the_chat(page_for, builder, upstream):
    suffix = uuid.uuid4().hex[:6]
    name, system_prompt = f"Pirate {suffix}", f"Answer like a pirate, crew {suffix}."
    page = page_for(builder)
    page.goto("/workspace/models/create")
    editor = page.get_by_role("main")
    editor.get_by_role("textbox", name="Model Name").fill(name)
    editor.get_by_role("button", name="Select a base model (e.g. llama3, gpt-4o)").click()
    _choose_from_models(page, "mock-model")
    editor.get_by_role("textbox", name=re.compile("^Write your model system prompt")).fill(
        system_prompt
    )
    editor.get_by_role("button", name="Save & Create").click()
    expect(page).to_have_url(re.compile(r"/workspace/models$"))

    _pick_model(page, name)
    upstream.queue(reply.text("Arr, ahoy!", match=reply.answering("greet me")))
    send(page, "greet me")
    expect_reply(page, "Arr, ahoy!")

    # later task requests (title, tags) may follow the chat request
    messages = next(filter(reply.answering("greet me"), upstream.chat_requests()))["messages"]
    assert messages[0]["role"] == "system"
    assert system_prompt in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "greet me"}


def test_a_saved_prompt_is_offered_by_its_slash_command_and_sent(page_for, builder, upstream):
    suffix = uuid.uuid4().hex[:6]
    command, text = f"seahaiku{suffix}", f"Write a haiku about the sea at dawn, take {suffix}."
    page = page_for(builder)
    page.goto("/workspace/prompts")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    creating = page.get_by_role("dialog").filter(has_text="Create Prompt")
    creating.get_by_role("textbox", name="Name", exact=True).fill(f"Sea haiku {suffix}")
    creating.get_by_role("textbox", name="Command").fill(command)
    creating.get_by_role("textbox", name=re.compile("^Write a summary in 50 words")).fill(text)
    creating.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_role("main").get_by_text(f"/{command}")).to_be_visible()

    page.goto("/")
    chat_input(page).click()
    page.keyboard.type(f"/{command[:8]}")
    page.get_by_role("tooltip").get_by_role("button", name=re.compile(command)).click()
    expect(chat_input(page)).to_have_text(text)

    upstream.queue(reply.text("Waves fold into light.", match=reply.answering(text)))
    page.keyboard.press("Enter")
    expect_reply(page, "Waves fold into light.")
    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == text
