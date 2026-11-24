"""Journey: a person follows the chat's sub-agents in the status popover and in the chat.

The lead model hands tasks to sub-agents. The Status popover of the lead chat lists them under
Threads, each by its title and age: a working one with a Stop button, finished ones folded under
"N done". The list follows a sub-agent from working to done while it stays open, a title opens
the sub-agent's own chat in a tab, and the lead chat's tab brings the person back. Every finished
sub-agent leaves a result row in the lead chat that reads how it ended (finished, failed or
stopped), opens to its result and opens its thread. A message a sub-agent sends the lead chat
shows as a row naming the sub-agent it came from, the lead answers it, and the row opens the
sub-agent's thread. A reload keeps the rows and the list.

Discriminates: passes on dev 9b626e4e5 (threads and messages added in c37725b7a, e85500e77 and
352c63be1). In a backend copy whose Threads list reports a working sub-agent as done, the list
test and the how-they-ended test turn red while the message test stays green.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_surface import main_chat, side_chat_titled
from utils.chat_ui import REPLY_TIMEOUT_MS, Chat, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True})
    assert saved.status_code == 200, saved.text


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def delegations(prompt: str, *tasks: tuple[str, str]) -> reply.Reply:
    """One reply of the lead model handing each (task, title) to a sub-agent."""
    calls = [
        {
            "id": f"call_{number}",
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps({"task": task, "title": title}),
            },
        }
        for number, (task, title) in enumerate(tasks)
    ]
    return reply.Reply(tool_calls=calls, match=reply.answering(prompt))


def report_of(task: str):
    """A `match` for the request that hands the finished sub-agent's report to the lead."""

    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def open_status(page: Page) -> None:
    page.get_by_role("button", name="Status", exact=True).first.click()
    expect(page.get_by_text("Chat ID", exact=True)).to_be_visible()


def thread_row(page: Page, title: str) -> Locator:
    """The Threads entry of the sub-agent titled `title` in the open Status popover."""
    menu = page.get_by_role("menu")
    return menu.get_by_role("listitem").filter(
        has=page.get_by_role("menuitem", name=title, exact=True)
    )


def done_toggle(page: Page, count: int) -> Locator:
    return page.get_by_role("menu").get_by_role("menuitem", name=f"{count} done", exact=True)


def result_row(lead: Chat, ending: str, title: str) -> Locator:
    """The row a sub-agent's report leaves in the lead chat, e.g. "Sub-agent finished · <title>"."""
    name = re.compile(rf"^{re.escape(f'{ending} · {title}')}")
    return conversation(lead).get_by_role("button", name=name)


def test_the_threads_list_follows_a_subagent_to_done_and_opens_its_thread(
    subagents_on, page_for, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("check the ledger of Demo User")
    title, answer = unique("Ledger check"), unique("the ledger balances")
    working = threading.Event()
    upstream.queue(
        delegations(prompt, (task, title)),
        reply.text(answer, hold=working, match=reply.answering(task)),
        reply.text("It is being checked.", match=reply.answering(prompt)),
        reply.text("The ledger is fine.", match=report_of(task)),
    )
    page = page_for(make_user())
    lead = main_chat(page)
    send(page, prompt)
    expect_reply(lead, "It is being checked.")

    open_status(page)
    expect(page.get_by_text("Threads", exact=True)).to_be_visible()
    row = thread_row(page, title)
    expect(row, "the working sub-agent is not listed under Threads").to_be_visible()
    expect(row.get_by_role("menuitem", name="Stop", exact=True)).to_be_visible()
    expect(row, "the sub-agent's age is not shown").to_contain_text(re.compile(r"\b\d+[smhd]\b"))
    expect(page.get_by_role("menuitem", name=re.compile(r"^\d+ done$"))).to_have_count(0)

    working.set()

    expect(done_toggle(page, 1), "the open list never showed the sub-agent as done").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect(row).to_have_count(0)
    done_toggle(page, 1).click()
    expect(done_toggle(page, 1)).to_have_attribute("aria-expanded", "true")
    expect(row).to_be_visible()
    expect(row.get_by_role("menuitem", name="Stop", exact=True)).to_have_count(0)

    row.get_by_role("menuitem", name=title, exact=True).click()
    thread = side_chat_titled(page, title, place="sub-agent thread")
    expect(thread.conversation).to_contain_text(task)
    expect(thread.conversation).to_contain_text(answer)

    lead.show()
    expect_reply(lead, "The ledger is fine.")
    finished = result_row(lead, "Sub-agent finished", title)
    expect(finished, "the lead chat shows no result row for the sub-agent").to_be_visible()
    expect(finished).to_contain_text(re.compile(r"\d+(\.\d+)?s"))
    finished.click()
    expect(conversation(lead)).to_contain_text(answer)
    expect(conversation(lead).get_by_role("button", name="Open thread")).to_be_visible()

    page.reload()

    expect_reply(lead, "The ledger is fine.")
    finished = result_row(lead, "Sub-agent finished", title)
    expect(finished, "the result row is gone after a reload").to_be_visible()
    finished.click()
    expect(conversation(lead)).to_contain_text(answer)
    conversation(lead).get_by_role("button", name="Open thread").click()
    thread = side_chat_titled(page, title, place="sub-agent thread")
    expect(thread.conversation).to_contain_text(answer)
    lead.show()
    open_status(page)
    expect(
        done_toggle(page, 1), "the Threads list lost the sub-agent after a reload"
    ).to_be_visible()


def test_finished_failed_and_stopped_subagents_each_show_how_they_ended(
    subagents_on, page_for, make_user, upstream
):
    prompt = unique("split this up")
    done_task, done_title = unique("count the apples"), unique("Apple count")
    failing_task, failing_title = unique("ask the broken helper"), unique("Broken helper")
    stopped_task, stopped_title = unique("watch the queue"), unique("Queue watch")
    never = threading.Event()
    upstream.queue(
        delegations(
            prompt,
            (done_task, done_title),
            (failing_task, failing_title),
            (stopped_task, stopped_title),
        ),
        reply.text("forty apples", match=reply.answering(done_task)),
        reply.error(500, "the helper model is down", match=reply.answering(failing_task)),
        reply.text("the queue is empty", hold=never, match=reply.answering(stopped_task)),
        reply.text("Three are on it.", match=reply.answering(prompt)),
    )
    page = page_for(make_user())
    lead = main_chat(page)
    send(page, prompt)
    # a report can already follow it
    expect(conversation(lead)).to_contain_text("Three are on it.", timeout=REPLY_TIMEOUT_MS)

    try:
        open_status(page)
        expect(
            done_toggle(page, 2), "the two ended sub-agents are not counted as done"
        ).to_be_visible(timeout=REPLY_TIMEOUT_MS)
        working = thread_row(page, stopped_title)
        expect(working).to_be_visible()
        working.get_by_role("menuitem", name="Stop", exact=True).click()
        expect(done_toggle(page, 3), "the stopped sub-agent is not counted as done").to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )
        expect(working).to_have_count(0)
        page.keyboard.press("Escape")
    finally:
        never.set()

    finished = result_row(lead, "Sub-agent finished", done_title)
    failed = result_row(lead, "Sub-agent failed", failing_title)
    stopped = result_row(lead, "Sub-agent stopped", stopped_title)
    for row, ending in ((finished, "finished"), (failed, "failed"), (stopped, "stopped")):
        expect(row, f"no result row reads that a sub-agent {ending}").to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )
    failed.click()
    expect(conversation(lead)).to_contain_text("the helper model is down")
    expect(conversation(lead)).not_to_contain_text("the queue is empty")

    page.reload()

    for row in (
        result_row(lead, "Sub-agent finished", done_title),
        result_row(lead, "Sub-agent failed", failing_title),
        result_row(lead, "Sub-agent stopped", stopped_title),
    ):
        expect(row, "a result row is gone after a reload").to_be_visible()
    open_status(page)
    expect(done_toggle(page, 3)).to_be_visible()


def test_a_message_from_a_subagent_shows_in_the_lead_chat_and_opens_its_thread(
    subagents_on, page_for, make_user, upstream
):
    greeting, prompt = unique("good morning"), unique("hand this over")
    task, title = unique("check the harbour log"), unique("Harbour log")
    note = unique("two ships are late, should I wait for them?")
    upstream.queue(reply.text("Morning.", match=reply.answering(greeting)))
    page = page_for(make_user())
    lead = main_chat(page)
    send(page, greeting)
    expect_reply(lead, "Morning.")
    page.wait_for_url(re.compile(r"/c/[0-9a-f-]+$"))
    lead_chat_id = page.url.rsplit("/", 1)[-1]

    upstream.queue(
        delegations(prompt, (task, title)),
        reply.tool_call(
            "send_message", {"chat_id": lead_chat_id, "message": note}, match=reply.answering(task)
        ),
        reply.text("Logged both ships.", match=reply.answering(task)),
        reply.text("A helper is on it.", match=reply.answering(prompt)),
        reply.text("Yes, wait for them.", match=reply.answering(note)),
        reply.text("The log is done.", match=report_of(task)),
    )
    send(page, prompt)
    expect(conversation(lead)).to_contain_text("A helper is on it.", timeout=REPLY_TIMEOUT_MS)

    message_row = conversation(lead).get_by_role(
        "button", name=re.compile(rf"^{re.escape(f'{title} · {note}')}")
    )
    expect(message_row, "the sub-agent's message never showed in the lead chat").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect(conversation(lead)).to_contain_text("Yes, wait for them.")
    expect(result_row(lead, "Sub-agent finished", title)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect_reply(lead, "The log is done.")

    page.reload()

    message_row = conversation(lead).get_by_role(
        "button", name=re.compile(rf"^{re.escape(f'{title} · {note}')}")
    )
    expect(message_row, "the message row is gone after a reload").to_be_visible()
    message_row.click()
    expect(message_row).to_have_attribute("aria-expanded", "true")
    opened = conversation(lead).get_by_role("button", name="Open thread")
    expect(opened).to_have_count(1)
    opened.click()
    thread = side_chat_titled(page, title, place="sub-agent thread")
    expect(thread.conversation).to_contain_text(task)
    expect(thread.conversation).to_contain_text("Logged both ships.")
    lead.show()
    expect(conversation(lead)).to_contain_text("Yes, wait for them.")
