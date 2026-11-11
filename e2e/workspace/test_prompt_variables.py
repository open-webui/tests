"""Journey: a prompt with input variables asks for them in a form before it is sent.

A prompt's content can carry typed variables (`{{topic | text:required}}`, a `select` with its
options, a `number`, a `date`, a `checkbox`). Picking the prompt from the chat's `/` menu opens
the Input Variables form with one field of the matching kind per variable; saving the form puts
the answers in place of the placeholders, and that filled text is what the model is sent. A
field marked `required` holds the form open until it has a value, and a field left alone keeps
its `default`. A prompt written in the workspace Create Prompt dialog that mixes a variable with the
built-in placeholders (user name, language, timezone, weekday) has the placeholders filled on
insertion, and the form still asks for the variable only.

Discriminates: passes on the 176d31d1d build. In a copy of that build where the form's select and
date fields fall back to the plain text area, where a bare flag such as `required` is dropped
when a variable is parsed and where the form starts every field empty in place of its default,
each of those three tests goes red; in a frontend build whose insertion leaves the built-in
placeholders unfilled the Create dialog test goes red; the chat journeys pass on those copies.
Retargeted for 37138282f, where the rewritten prompt editor names its fields Prompt Name and
Prompt Content: the tests that fill the Create Prompt dialog pass on that build.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`); the Create dialog
test makes its prompt in a page of its own and picks it in the chat. The prompt is filled in the
message box, so a side chat fills it as the main chat does, also on dev c15f28649, unlike the
system prompt variables of open-webui/open-webui#34032.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERY_KIND = (
    'Plan {{topic | text:placeholder="Topic"}} for'
    ' {{audience | select:options=["Kids","Adults"]:placeholder="Audience"}},'
    ' {{count | number:placeholder="How many"}} stops,'
    ' on {{day | date:placeholder="Day"}}, notes {{notes | checkbox:label="Add notes"}}.'
)


@pytest.fixture
def prompt_with(make_user):
    """A fresh admin; `prompt_with(content)` saves a prompt of theirs and returns its command."""
    account = make_user(role="admin")

    def save(content: str) -> str:
        command = f"plan{uuid.uuid4().hex[:8]}"
        form = {"command": command, "name": f"Plan {command}", "content": content}
        with account.client() as client:
            created = client.post("/api/v1/prompts/create", json=form)
        assert created.status_code == 200, created.text
        return command

    return account, save


def _pick_prompt(chat: ChatSurface, command: str) -> Locator:
    """Pick `/command` in the chat input; returns the variables form it opens."""
    page = chat.page
    expect(chat_input(chat)).to_be_visible()
    chat_input(chat).click()
    page.keyboard.type(f"/{command}")
    page.get_by_role("tooltip").get_by_role("button", name=command).click()
    form = page.get_by_role("dialog").filter(has_text="Input Variables")
    expect(form).to_be_visible()
    return form


def _send_and_read(chat: ChatSurface, upstream, expected: str) -> str:
    upstream.queue(reply.text("Planned.", match=reply.answering(expected)))
    chat.page.keyboard.press("Enter")
    expect_reply(chat, "Planned.")
    sent = next(filter(reply.answering(expected), upstream.chat_requests()))
    return reply.conversation(sent)[-1]["content"]


def test_each_kind_of_variable_is_asked_for_and_the_filled_text_is_sent(
    chat_for, prompt_with, upstream
):
    account, save = prompt_with
    command = save(EVERY_KIND)
    chat = chat_for(account)

    form = _pick_prompt(chat, command)
    form.get_by_placeholder("Topic").fill("a harbour walk")
    form.get_by_role("combobox").select_option("Adults")
    form.get_by_placeholder("How many").fill("4")
    form.get_by_placeholder("Day").fill("2026-10-03")
    form.get_by_role("checkbox", name="Add notes").check()
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    filled = "Plan a harbour walk for Adults, 4 stops, on 2026-10-03, notes true."
    expect(chat_input(chat)).to_have_text(filled)
    assert _send_and_read(chat, upstream, filled) == filled


def test_a_required_variable_holds_the_form_until_it_is_filled(chat_for, prompt_with):
    account, save = prompt_with
    command = save('Review {{title | text:placeholder="Title":required}} today.')
    chat = chat_for(account)

    form = _pick_prompt(chat, command)
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_visible()
    assert form.get_by_placeholder("Title").evaluate("field => field.validity.valueMissing")

    form.get_by_placeholder("Title").fill("the launch notes")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()
    expect(chat_input(chat)).to_have_text("Review the launch notes today.")


def test_a_variable_left_alone_keeps_its_default(chat_for, prompt_with, upstream):
    account, save = prompt_with
    command = save(
        'Pack {{count | number:placeholder="How many":default=3}} bags for'
        ' {{trip | select:options=["Hiking","Sailing"]:default="Sailing"}}.'
    )
    chat = chat_for(account)

    form = _pick_prompt(chat, command)
    expect(form.get_by_placeholder("How many")).to_have_value("3")
    expect(form.get_by_role("combobox")).to_have_value("Sailing")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    filled = "Pack 3 bags for Sailing."
    expect(chat_input(chat)).to_have_text(filled)
    assert _send_and_read(chat, upstream, filled) == filled


def test_a_prompt_made_in_the_create_dialog_fills_its_built_in_placeholders(
    page_for, chat_for, make_user, upstream
):
    account = make_user(role="admin")
    zone = "Pacific/Auckland"
    suffix = uuid.uuid4().hex[:6]
    name = f"Itinerary {suffix}"
    page = page_for(account, path="/workspace/prompts/create", timezone_id=zone)

    creating = page.get_by_role("dialog").filter(has_text="Create Prompt")
    creating.get_by_role("textbox", name="Prompt Name").fill(name)
    creating.get_by_role("textbox", name="Command").fill(f"itinerary-{suffix}")
    creating.get_by_role("textbox", name="Prompt Content").fill(
        "{{USER_NAME}} ({{USER_LANGUAGE}}, {{CURRENT_TIMEZONE}}, {{CURRENT_WEEKDAY}}) plans"
        ' {{place | text:placeholder="Place"}}.'
    )
    creating.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_role("main").get_by_text(name)).to_be_visible()

    chat = chat_for(account, timezone_id=zone)
    form = _pick_prompt(chat, f"itinerary-{suffix}")
    form.get_by_placeholder("Place").fill("Wellington")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    language = chat.page.evaluate("localStorage.getItem('locale') || 'en-US'")
    weekday = datetime.now(ZoneInfo(zone)).strftime("%A")
    filled = f"{account.name} ({language}, {zone}, {weekday}) plans Wellington."
    expect(chat_input(chat)).to_have_text(filled)
    assert _send_and_read(chat, upstream, filled) == filled
