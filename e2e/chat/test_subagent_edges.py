"""Journey: a sub-agent stopped from Threads, and a message from another chat whose answer fails.

The model delegates a task and its reply ends while the sub-agent works. The chat's Status menu
lists the sub-agent under Threads with a Stop button; Stop ends it, the chat shows a "Sub-agent
stopped" row for the task, the model carries on from it without the person typing, and Threads
then counts the sub-agent as done with no Stop left.

An agent in one chat sends a message to another chat the person has open. The open chat shows
the message as a row that names the chat it came from, without a reload. When the chat's answer
to it fails, the error shows under it and the next message the person types is answered.

The integration twin is integration/tools/test_agent_messages_edges.py, which pins who can be
messaged, the order of several messages and the delete of a chat with a running sub-agent.

Discriminates: passes on dev 9b626e4e5 with its CI frontend build, two runs in a row. In backend
copies of it the Stop test turns red with the chat stop endpoint no longer stopping the task (the
thread keeps its Stop), and the message test with the reload the server sends once the message
is stored removed (the row never shows in the open chat).
"""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")


@pytest.fixture
def sub_agents_on(admin, preserve) -> None:
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True})
    assert saved.status_code == 200, saved.text


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def report_of(task: str):
    """A `match` for the request that carries the sub-agent's report back to the model."""

    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def open_status(page: Page) -> Locator:
    """The chat's Status menu, opened."""
    menu = page.get_by_role("menu").filter(has_text="Chat ID")
    page.get_by_role("button", name="Status").click()
    expect(menu).to_be_visible()
    return menu


def close_status(page: Page, menu: Locator) -> None:
    page.keyboard.press("Escape")
    expect(menu).to_be_hidden()


def test_stopping_a_sub_agent_from_threads_shows_it_stopped_and_the_chat_carries_on(
    sub_agents_on, page_for, make_user, upstream
):
    prompt, title = unique("hand this over"), unique("Crate count")
    task = unique("count the crates in the yard")
    release = threading.Event()
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": task, "title": title}, match=reply.answering(prompt)
        ),
        reply.text("never reported", hold=release, match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("The count was stopped.", match=report_of(task)),
    )
    try:
        page = page_for(make_user())
        send(page, prompt)
        expect_reply(page, "Handed over.")

        menu = open_status(page)
        thread = menu.get_by_role("listitem").filter(has_text=title)
        expect(menu.get_by_text("Threads", exact=True)).to_be_visible()
        expect(thread).to_be_visible()
        thread.get_by_role("menuitem", name="Stop").click()
        expect(thread.get_by_role("menuitem", name="Stop")).to_have_count(0)
        close_status(page, menu)

        stopped = conversation(page).get_by_role(
            "button", name=re.compile(re.escape(f"Sub-agent stopped · {title}"))
        )
        expect(stopped, "the stopped sub-agent never showed in the chat").to_be_visible(
            timeout=REPLY_TIMEOUT_MS
        )
        expect_reply(page, "The count was stopped.")

        menu = open_status(page)
        expect(menu.get_by_role("menuitem", name=re.compile(r"^1\s*done"))).to_be_visible()
        expect(menu.get_by_role("menuitem", name="Stop")).to_have_count(0)
    finally:
        release.set()


def test_a_message_from_another_chat_names_its_sender_and_a_failed_answer_leaves_the_chat_usable(
    sub_agents_on, page_for, make_user, upstream
):
    person = make_user()
    opening, sender_title = unique("standing by"), unique("Harbour log")
    prompt, text = unique("pass this on"), unique("the ferry leaves at nine")
    with person.client() as client:
        upstream.queue(reply.text("Standing by.", match=reply.answering(opening)))
        target, _ = ask(client, opening)
        upstream.queue(reply.text("Ready.", match=reply.answering(sender_title)))
        sender, _ = ask(client, sender_title)
        renamed = client.post(
            f"/api/v1/chats/{sender.chat_id}", json={"chat": {"title": sender_title}}
        )
        assert renamed.status_code == 200, renamed.text

    page = page_for(person, path=f"/c/{target.chat_id}")
    expect_reply(page, "Standing by.")

    upstream.queue(
        reply.tool_call(
            "send_message",
            {"chat_id": target.chat_id, "message": text},
            match=reply.answering(prompt),
        ),
        reply.text("Passed on.", match=reply.answering(prompt)),
        reply.error(500, "the model is down", match=reply.answering(text)),
    )
    with person.client() as client:
        history = client.get(f"/api/v1/chats/{sender.chat_id}").json()["chat"]["history"]
        turn = send_message(client, prompt, chat_id=sender.chat_id, parent_id=history["currentId"])
        wait_for_reply(client, turn)

    row = conversation(page).get_by_role("button", name=re.compile(re.escape(sender_title)))
    expect(row, "the message from the other chat never showed in the open chat").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect(row).to_contain_text(text)
    expect_reply(page, "the model is down")

    later = unique("are you back?")
    upstream.queue(reply.text("Back again.", match=reply.answering(later)))
    send(page, later)
    expect_reply(page, "Back again.")
    expect(row).to_be_visible()
