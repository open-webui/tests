"""Journey: an automation made on the Automations page runs on demand and its run opens a chat.

A fresh admin (plain accounts need the automations permission) creates a daily automation with a
prompt and the scripted model, lands on its page, and presses Run now. The run is listed under
Runs, and its chat holds the prompt and the model's answer. Every further run joins the history
with its own chat, and a run whose model no longer exists is listed with the error it failed on
and no chat.

Bug, open-webui/open-webui#31580, fixed by PR #31583: a Run now run never set the automation's last
run time, so after it the page still said "Last run Never" above the run it listed (the list page
said "Never" as well), since only the scheduler's own claim wrote that time.
`test_a_manual_run_shows_as_the_last_run` pins it.

`test_a_manual_run_shows_as_the_last_run`, `test_run_now_lists_a_run_whose_chat_holds_the_answer`
and `test_the_run_history_lists_every_run_with_its_chat` are red on dev 62f70a844: since de73bb830 a
chat request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev a5bc78300, and the last run test fails on dev 176d31d1d, before PR
#31583. On dev ac00d40e3, in a backend copy, with `POST /api/v1/automations/{id}/run` answering
without starting the run no run is ever listed. On dev 176d31d1d a frontend copy keeping only the
newest run turns the history test red, and one hiding a run's error turns the failed run test red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

RUN_TIMEOUT_MS = 30_000


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


def _run_now(page: Page) -> None:
    triggered = page.get_by_text("Automation triggered")
    # the previous run's toast covers the button and stays while the pointer rests on it
    page.mouse.move(0, 0)
    expect(triggered).to_be_hidden(timeout=RUN_TIMEOUT_MS)
    page.get_by_role("button", name="Run now").click()
    expect(triggered).to_be_visible()


def test_the_run_history_lists_every_run_with_its_chat(
    page_for, scheduler, make_automation, upstream
):
    automation = make_automation(scheduler)
    prompt = automation["data"]["prompt"]
    upstream.queue(
        reply.text("First report.", match=reply.answering(prompt)),
        reply.text("Second report.", match=reply.answering(prompt)),
    )
    page = page_for(scheduler)
    page.goto(f"/automations/{automation['id']}")
    details = page.get_by_role("main")
    expect(details).to_contain_text("No runs yet")

    view_chat = details.get_by_role("button", name="View chat")
    _run_now(page)
    expect(view_chat).to_have_count(1, timeout=RUN_TIMEOUT_MS)
    _run_now(page)
    expect(view_chat).to_have_count(2, timeout=RUN_TIMEOUT_MS)

    page.reload()
    expect(view_chat).to_have_count(2)


def test_a_run_whose_model_is_gone_is_listed_as_failed(page_for, scheduler, make_automation):
    automation = make_automation(scheduler, model_id=f"retired-{uuid.uuid4().hex[:6]}")
    page = page_for(scheduler)
    page.goto(f"/automations/{automation['id']}")
    details = page.get_by_role("main")
    expect(details).to_contain_text("No runs yet")

    _run_now(page)
    expect(details.get_by_text("Model not found")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    expect(details.get_by_role("button", name="View chat")).to_have_count(0)


def test_a_manual_run_shows_as_the_last_run(page_for, scheduler, make_automation, upstream):
    automation = make_automation(scheduler)
    prompt = automation["data"]["prompt"]
    upstream.queue(reply.text("The report.", match=reply.answering(prompt)))
    page = page_for(scheduler)
    page.goto(f"/automations/{automation['id']}")
    details = page.get_by_role("main")
    expect(details).to_contain_text("Last run Never")

    _run_now(page)
    expect(details.get_by_role("button", name="View chat")).to_be_visible(timeout=RUN_TIMEOUT_MS)
    page.reload()
    expect(details.get_by_role("button", name="View chat")).to_be_visible()

    expect(
        details,
        "the page says it never ran while its history lists a run (open-webui/open-webui#31580)",
    ).not_to_contain_text("Last run Never")
