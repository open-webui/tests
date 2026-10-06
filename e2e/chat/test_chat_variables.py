"""Journey: a model's chat variables are asked for once, fill its system prompt and stay put.

A system prompt with `{{chat.variables.<key> | ...}}` placeholders turns into a Chat Variables form,
as the chat parameters docs page describes. The first message of a chat whose required variable
has no value opens the form in place of sending; once it is saved, the message goes out with the
values written into the system prompt. The values are kept with the chat, so later messages and
the chat after a reload use them without asking, and the Chat Variables control beside the
message box changes them for the messages after. A model whose variables all have defaults
answers the first message straight away, with the defaults filled in.

Discriminates: passes on dev 30f3f6a8f; in a backend copy that writes every chat variable as empty
into the system prompt every test fails, and in a frontend build that sends the first message
without asking the three tests that fill the form fail.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROJECT = '{{chat.variables.project_name | text:placeholder="Which project?":required}}'
TONE = '{{chat.variables.tone | select:options=["Formal","Casual"]:default="Formal"}}'
TRIP_PROMPT = f"You plan trips for {PROJECT} in a {TONE} tone."
DEFAULTS_PROMPT = (
    'You write for {{chat.variables.audience | text:placeholder="Audience":default="sailors"}}.'
)


@pytest.fixture
def variables_model(make_user):
    """A fresh admin and a function making their models with a given system prompt."""
    account = make_user(role="admin")
    made: list[str] = []

    def make(system: str) -> str:
        model_id = f"variables-{uuid.uuid4().hex[:6]}"
        form = {
            "id": model_id,
            "name": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "meta": {},
            "params": {"system": system},
        }
        with account.client() as client:
            created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        made.append(model_id)
        return model_id

    yield account, make
    with account.client() as client:
        for model_id in made:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def _system_prompt(upstream, question: str) -> str:
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def _ask(page: Page, upstream, question: str, answer: str = "Planned.") -> str:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, answer)
    return _system_prompt(upstream, question)


def _form(page: Page) -> Locator:
    form = page.get_by_role("dialog").filter(has_text="Chat Variables")
    expect(form).to_be_visible()
    return form


def _first_message_through_the_form(page: Page, upstream, project: str) -> str:
    question = f"plan the first day {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Day one planned.", match=reply.answering(question)))
    send(page, question)
    form = _form(page)
    assert not [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    form.get_by_placeholder("Which project?").fill(project)
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()
    # the message waits in the box until it is sent again
    expect(chat_input(page)).to_have_text(question)
    chat_input(page).click()
    page.keyboard.press("Enter")
    expect_reply(page, "Day one planned.")
    return _system_prompt(upstream, question)


def test_a_required_variable_is_asked_for_and_fills_the_system_prompt(
    page_for, variables_model, upstream
):
    account, make = variables_model
    page = page_for(account)
    page.goto(f"/?model={make(TRIP_PROMPT)}")

    system = _first_message_through_the_form(page, upstream, "Harbour Fair")
    assert "You plan trips for Harbour Fair in a Formal tone." in system


def test_the_values_stay_with_the_chat_for_later_messages_and_a_reload(
    page_for, variables_model, upstream
):
    account, make = variables_model
    page = page_for(account)
    page.goto(f"/?model={make(TRIP_PROMPT)}")
    _first_message_through_the_form(page, upstream, "Island Hop")

    later = _ask(page, upstream, f"and the second day? {uuid.uuid4().hex[:6]}")
    assert "You plan trips for Island Hop in a Formal tone." in later
    page.reload()
    expect_reply(page, "Planned.")
    after_reload = _ask(page, upstream, f"and the third day? {uuid.uuid4().hex[:6]}")
    assert "You plan trips for Island Hop in a Formal tone." in after_reload


def test_the_control_beside_the_message_box_changes_a_value_for_later_messages(
    page_for, variables_model, upstream
):
    account, make = variables_model
    page = page_for(account)
    page.goto(f"/?model={make(TRIP_PROMPT)}")
    _first_message_through_the_form(page, upstream, "Coast Walk")

    page.get_by_role("button", name="Chat Variables").click()
    form = _form(page)
    form.get_by_role("combobox").select_option("Casual")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(form).to_be_hidden()

    changed = _ask(page, upstream, f"anything else? {uuid.uuid4().hex[:6]}")
    assert "You plan trips for Coast Walk in a Casual tone." in changed


def test_variables_with_defaults_send_the_first_message_without_asking(
    page_for, variables_model, upstream
):
    account, make = variables_model
    page = page_for(account)
    page.goto(f"/?model={make(DEFAULTS_PROMPT)}")

    system = _ask(page, upstream, f"write a welcome {uuid.uuid4().hex[:6]}", "Welcome aboard.")
    assert "You write for sailors." in system
    expect(page.get_by_role("dialog").filter(has_text="Chat Variables")).to_have_count(0)
