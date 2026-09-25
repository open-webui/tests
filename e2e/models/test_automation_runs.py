"""Journey: an automation made on the Automations page runs on demand and its run opens a chat.

A fresh admin (plain accounts need the automations permission) creates a daily automation with a
prompt and the scripted model, lands on its page, and presses Run now. The run is listed under
Runs, and its chat holds the prompt and the model's answer.

Discriminates: passes on dev ac00d40e3; in a backend copy, with `POST /api/v1/automations/{id}/run`
answering without starting the run no run is ever listed.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

RUN_TIMEOUT_MS = 30_000


@pytest.fixture
def scheduler(make_user):
    """A fresh admin, whose automations are deleted afterwards so none runs on its own later."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


def test_run_now_lists_a_run_whose_chat_holds_the_answer(page_for, scheduler, upstream):
    title = f"Morning digest {uuid.uuid4().hex[:6]}"
    prompt = f"Summarize the overnight tickets, batch {uuid.uuid4().hex[:6]}."
    page = page_for(scheduler)
    page.goto("/automations")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    creating = page.get_by_role("dialog")
    creating.get_by_role("button", name="Select model").first.click()
    picker = page.get_by_role("menu")
    picker.get_by_role("textbox", name="Search a model").fill("mock-model")
    picker.get_by_role("button", name="mock-model").click()
    creating.get_by_role("textbox", name="Enter prompt here.").fill(prompt)
    # the dialog clears the title once its folders and channels load, so it goes in last
    creating.get_by_role("textbox", name="Automation title").fill(title)
    creating.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile(r"/automations/[0-9a-f-]+$"))
    details = page.get_by_role("main")
    expect(details).to_contain_text(title)
    expect(details).to_contain_text(prompt)
    expect(details).to_contain_text("No runs yet")

    upstream.queue(reply.text("Three tickets came in overnight.", match=reply.answering(prompt)))
    details.get_by_role("button", name="Run now").click()
    view_chat = details.get_by_role("button", name="View chat")
    expect(view_chat).to_be_visible(timeout=RUN_TIMEOUT_MS)

    view_chat.click()
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    expect(conversation(page).get_by_text(prompt)).to_be_visible()
    expect_reply(page, "Three tickets came in overnight.")
