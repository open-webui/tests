"""Journey: an automation made on the Automations page runs on demand and its run opens a chat.

A fresh admin (plain accounts need the automations permission) creates a daily automation with a
prompt and the scripted model, lands on its page, and presses Run now. The run is listed in the
page's Activity, and its chat holds the prompt and the model's answer. Every further run joins
the Activity with its own chat, and a run whose model no longer exists is listed with the error it
failed on, next to the chat it started (since dev de446952c a failed run keeps its chat). Since dev
9723b3a6a the page is the automation's editor and lists runs as Ran or Failed with their time.

Bug, open-webui/open-webui#31580, fixed by PR #31583: a Run now run never set the automation's last
run time, so after it the page still said "Last run Never" above the run it listed and the list
page said "Never" as well, since only the scheduler's own claim wrote that time. Dev 9723b3a6a took
the last run off the page, so `test_a_manual_run_shows_as_the_last_run` pins it on the list.

`test_a_manual_run_shows_as_the_last_run`, `test_run_now_lists_a_run_whose_chat_holds_the_answer`
and `test_the_run_history_lists_every_run_with_its_chat` are red on dev 62f70a844: since de73bb830 a
chat request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev 2fdf4ce6b; a backend copy whose Run now no longer records the last
run time (PR #31583 undone) turns the last run test red. On dev ac00d40e3, in a backend copy, with
`POST /api/v1/automations/{id}/run` answering without starting the run no run is ever listed. On
dev 2fdf4ce6b a frontend copy keeping only the newest run turns the history test red, and one
hiding a run's error turns the failed run test red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.automations_page import (
    automation_row,
    expect_no_runs,
    instructions,
    name_box,
    run_entries,
    run_now,
)
from utils.chat_ui import conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def open_run(page: Page, entry) -> None:
    """Open a run's chat from its Activity entry."""
    entry.click()
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))


def test_run_now_lists_a_run_whose_chat_holds_the_answer(page_for, scheduler, upstream):
    title = f"Morning digest {uuid.uuid4().hex[:6]}"
    prompt = f"Summarize the overnight tickets, batch {uuid.uuid4().hex[:6]}."
    page = page_for(scheduler, path="/automations")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    creating = page.get_by_role("dialog")
    creating.get_by_role("button", name="Select model").first.click()
    picker = page.get_by_role("menu")
    picker.get_by_role("textbox", name="Search a model").fill("mock-model")
    picker.get_by_role("menuitem", name="mock-model").click()
    creating.get_by_role("textbox", name="Instructions").fill(prompt)
    # the dialog clears the title once its folders and channels load, so it goes in last
    creating.get_by_role("textbox", name="Automation title").fill(title)
    creating.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile(r"/automations/[0-9a-f-]+$"))
    expect(name_box(page)).to_have_value(title)
    expect(instructions(page)).to_have_value(prompt)
    expect_no_runs(page)

    upstream.queue(reply.text("Three tickets came in overnight.", match=reply.answering(prompt)))
    run_now(page, scheduler, page.url.rsplit("/", 1)[-1])
    expect(run_entries(page)).to_contain_text("Ran")

    open_run(page, run_entries(page))
    expect(conversation(page).get_by_text(prompt)).to_be_visible()
    expect_reply(page, "Three tickets came in overnight.")


def test_the_run_history_lists_every_run_with_its_chat(
    page_for, scheduler, make_automation, upstream
):
    automation = make_automation(scheduler)
    prompt = automation["data"]["prompt"]
    upstream.queue(
        reply.text("First report.", match=reply.answering(prompt)),
        reply.text("Second report.", match=reply.answering(prompt)),
    )
    page = page_for(scheduler, path=f"/automations/{automation['id']}")
    expect_no_runs(page)

    run_now(page, scheduler, automation["id"], runs=1)
    run_now(page, scheduler, automation["id"], runs=2)

    # newest first
    open_run(page, run_entries(page).last)
    expect_reply(page, "First report.")


def test_a_run_whose_model_is_gone_is_listed_as_failed(page_for, scheduler, make_automation):
    automation = make_automation(scheduler, model_id=f"retired-{uuid.uuid4().hex[:6]}")
    prompt = automation["data"]["prompt"]
    page = page_for(scheduler, path=f"/automations/{automation['id']}")
    expect_no_runs(page)

    run_now(page, scheduler, automation["id"])
    failed = run_entries(page)
    expect(failed).to_contain_text("Failed")
    expect(failed).to_contain_text("Model not found")

    open_run(page, failed)
    expect(conversation(page).get_by_text(prompt)).to_be_visible()


def _list_row(page: Page, name: str):
    page.goto("/automations")
    page.get_by_role("textbox", name="Search Automations").fill(name)
    row = automation_row(page, name)
    expect(row).to_contain_text("Daily at 9:00 AM")
    return row


def test_a_manual_run_shows_as_the_last_run(page_for, scheduler, make_automation, upstream):
    automation = make_automation(scheduler)
    prompt = automation["data"]["prompt"]
    upstream.queue(reply.text("The report.", match=reply.answering(prompt)))
    page = page_for(scheduler)
    expect(_list_row(page, automation["name"])).to_contain_text("Never")

    page.goto(f"/automations/{automation['id']}")
    run_now(page, scheduler, automation["id"])

    expect(
        _list_row(page, automation["name"]),
        "the list says it never ran while its history lists a run (open-webui/open-webui#31580)",
    ).not_to_contain_text("Never")
