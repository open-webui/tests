"""Journey: actions as buttons under a reply, and what pressing one does.

A global action shows its button under a reply on any model; pressing it rewrites the reply, as
the action docs promise for an action that returns `{"messages": [...]}`. An action that asks the
person something opens a dialog with its title and message: Cancel tells the action nobody
answered, and a typed answer reaches it. An action with sub-actions shows one button per
sub-action, each running its own branch, and the buttons are ordered by each action's `priority`
valve, so raising one in its Valves dialog moves its buttons to the end. An action switched off
in Admin Panel > Functions leaves the reply.

The rewrite test is red on dev: the action's new content is stored on the reply, but the page
renders a reply from its `output` items, which the rewrite leaves alone, so the old text stays on
screen live and after a reload.

Discriminates: on dev ebc6add67 all pass but the rewrite test (the bug above), which passes on a
frontend build that writes the action's content into the reply's output as well. One backend copy
whose action route hands the action no event caller, whose model list orders actions by id alone
and whose active function lookup ignores the switch turned the dialog, priority and switch tests
red, each on its own edit.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, last_reply, send
from utils.valves import customise, valve

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FAIR_COPY_ACTION = """class Action:
    async def action(self, body: dict):
        pressed = body["messages"][-1]
        return {"messages": [{"id": body["id"], "content": "Fair copy: " + pressed["content"]}]}
"""

BOARDING_ACTION = """class Action:
    async def action(self, body: dict, __event_call__=None, __event_emitter__=None):
        answer = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Who is boarding?",
                    "message": "Name the passenger for the pass.",
                    "placeholder": "Passenger name",
                },
            }
        )
        note = f"Boarding pass printed for {answer}" if answer else "Boarding cancelled"
        await __event_emitter__({"type": "notification", "data": {"type": "info", "content": note}})
"""

STAMPING_ACTION = """from pydantic import BaseModel, Field


class Action:
    actions = [
        {"id": "arrival", "name": "Stamp arrival"},
        {"id": "departure", "name": "Stamp departure"},
    ]

    class Valves(BaseModel):
        priority: int = Field(1, description="Lower shows first")

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict, __id__=None, __event_emitter__=None):
        note = {"type": "success", "content": f"Stamped {__id__}"}
        await __event_emitter__({"type": "notification", "data": note})
"""

LOGGING_ACTION = """from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        priority: int = Field(2, description="Lower shows first")

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict, __event_emitter__=None):
        note = {"type": "success", "content": "Logged at the desk"}
        await __event_emitter__({"type": "notification", "data": note})
"""


def answered_page(page: Page, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, answer)


def last_message(page: Page) -> Locator:
    """The last reply with the buttons under it."""
    return last_reply(page).locator("xpath=ancestor::div[starts-with(@id, 'message-')][1]")


def action_button(page: Page, name: str) -> Locator:
    return last_message(page).get_by_role("button", name=name, exact=True)


def function_card(page: Page, kind: str, function_id: str) -> Locator:
    page.goto("/admin/functions")
    page.get_by_placeholder("Search Functions").fill(function_id)
    card = page.get_by_role("main").get_by_role("button", name=re.compile(f"^{kind} {function_id}"))
    expect(card).to_be_visible()
    return card


def test_a_global_action_rewrites_the_reply_it_was_pressed_on(page_for, admin, make_user, upstream):
    with installed_function(admin, FAIR_COPY_ACTION, is_global=True) as action_id:
        page = page_for(make_user())
        answered_page(page, upstream, "when does the ferry leave?", "The ferry leaves at noon.")

        with page.expect_response(lambda response: "/api/chat/actions/" in response.url) as ran:
            action_button(page, action_id).click()
        [rewrite] = ran.value.json()["messages"]
        assert rewrite["content"] == "Fair copy: The ferry leaves at noon."

        shown = "the page keeps showing the reply's output, not the action's rewrite"
        expect(last_reply(page), shown).to_contain_text("Fair copy: The ferry leaves at noon.")
        page.reload()
        expect(last_reply(page), shown).to_contain_text("Fair copy: The ferry leaves at noon.")


def test_an_action_asks_the_person_and_gets_their_answer_or_their_cancel(
    page_for, admin, make_user, upstream
):
    with installed_function(admin, BOARDING_ACTION, is_global=True) as action_id:
        page = page_for(make_user())
        answered_page(page, upstream, "may I board?", "Boarding opens now.")

        action_button(page, action_id).click()
        dialog = page.get_by_role("dialog", name="Who is boarding?")
        expect(dialog).to_contain_text("Name the passenger for the pass.")
        dialog.get_by_role("button", name="Cancel").click()
        expect(page.get_by_text("Boarding cancelled")).to_be_visible()

        action_button(page, action_id).click()
        dialog.get_by_placeholder("Passenger name").fill("Ada Lovelace")
        dialog.get_by_role("button", name="Confirm").click()
        expect(page.get_by_text("Boarding pass printed for Ada Lovelace")).to_be_visible()


def action_order(page: Page, names: list[str]) -> list[str]:
    """The reply's buttons among `names`, left to right."""
    labels = (
        last_message(page)
        .get_by_role("button")
        .evaluate_all("(buttons) => buttons.map((button) => button.getAttribute('aria-label'))")
    )
    return [label for label in labels if label in names]


def test_sub_actions_each_run_and_follow_their_priority_valve(page_for, admin, make_user, upstream):
    with (
        installed_function(admin, STAMPING_ACTION, is_global=True) as stamping_id,
        installed_function(admin, LOGGING_ACTION, is_global=True) as logging_id,
    ):
        names = ["Stamp arrival", "Stamp departure", logging_id]
        page = page_for(make_user())
        answered_page(page, upstream, "may I land here?", "Welcome ashore.")
        expect(action_button(page, logging_id)).to_be_visible()
        assert action_order(page, names) == names

        action_button(page, "Stamp departure").click()
        expect(page.get_by_text("Stamped departure")).to_be_visible()
        action_button(page, "Stamp arrival").click()
        expect(page.get_by_text("Stamped arrival")).to_be_visible()

        admin_page = page_for(admin)
        function_card(admin_page, "action", stamping_id).get_by_role(
            "button", name="Valves"
        ).click()
        dialog = admin_page.get_by_role("dialog").filter(has_text="Valves")
        priority = valve(dialog, "Priority", "Lower shows first")
        customise(priority)
        priority.get_by_role("textbox").fill("3")
        dialog.get_by_role("button", name="Save").click()
        expect(admin_page.get_by_text("Valves updated successfully")).to_be_visible()

        page.reload()
        expect(action_button(page, logging_id)).to_be_visible()
        assert action_order(page, names) == [logging_id, "Stamp arrival", "Stamp departure"]


def test_an_action_switched_off_leaves_the_reply(page_for, admin, make_user, upstream):
    with installed_function(admin, LOGGING_ACTION, is_global=True) as action_id:
        page = page_for(make_user())
        answered_page(page, upstream, "where do I check in?", "At the harbour desk.")
        expect(action_button(page, action_id)).to_be_visible()

        admin_page = page_for(admin)
        switch = function_card(admin_page, "action", action_id).get_by_role("switch")
        switch.click()
        expect(switch).not_to_be_checked()

        page.reload()
        expect_reply(page, "At the harbour desk.")
        expect(last_message(page).get_by_role("button", name="Copy")).to_have_count(1)
        expect(action_button(page, action_id)).to_have_count(0)
