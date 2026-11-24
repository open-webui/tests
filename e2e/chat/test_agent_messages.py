"""Journey: what a person sees when agents message each other's chats.

A message an agent sends with the send_message tool shows in the receiving chat as a row naming
the chat it came from and what it said, followed by that chat's answer, in the open page without a
reload. The row opens to the full message and to the chat that sent it. In the sending chat the
call reads "Sent message", and an answer sent back shows there as a row of its own with the reply
to it. A sub-agent asking its lead shows the same way in the lead's chat, before its result row.

Discriminates: passes on dev 9b626e4e5. In backend copies both tests turn red with the message
stored without the sender's id and title, and with an idle chat not woken by a message.
"""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat import ask
from utils.chat_ui import REPLY_TIMEOUT_MS, Chat, conversation, send, stop_button, tool_step

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
    return f"{label} {uuid.uuid4().hex[:8]}"


def message_to(chat_id: str, text: str, prompt: str, **options) -> reply.Reply:
    return reply.tool_call(
        "send_message",
        {"chat_id": chat_id, "message": text},
        match=reply.answering(prompt),
        **options,
    )


def report_of(task: str):
    def matches(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        content = str(users[-1].get("content")) if users else ""
        return "[ASYNC SUBAGENT COMPLETE" in content and task in content

    return matches


def message_row(chat: Chat, sender_title: str, text: str) -> Locator:
    """The row of a received message, which reads the sending chat's title and the message."""
    name = re.compile(re.escape(f"{sender_title} · {text}"))
    return conversation(chat).get_by_role("button", name=name)


def expect_in_chat(chat: Chat, text: str) -> None:
    expect(conversation(chat)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def saved_chat(client, prompt: str, answer: str, title: str, upstream) -> str:
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    turn, _ = ask(client, prompt)
    renamed = client.post(f"/api/v1/chats/{turn.chat_id}", json={"chat": {"title": title}})
    assert renamed.status_code == 200, renamed.text
    return turn.chat_id


def test_a_message_and_its_answer_show_in_both_chats_with_where_they_came_from(
    subagents_on, page_for, make_user, upstream
):
    person = make_user()
    asker_title, helper_title = unique("Demo planning"), unique("Demo calendar")
    question, answer = unique("When is the review?"), unique("The review is on Friday.")
    prompt = unique("ask the calendar chat")
    asker_idle = threading.Event()
    with person.client() as client:
        helper_id = saved_chat(
            client, unique("keep the calendar"), "Calendar ready.", helper_title, upstream
        )
        asker_id = saved_chat(
            client, unique("plan the week"), "Planning ready.", asker_title, upstream
        )
    upstream.queue(
        message_to(helper_id, question, prompt),
        reply.text("Asked the calendar.", match=reply.answering(prompt)),
        # held so the asking chat has finished its reply when the answer comes back
        message_to(asker_id, answer, question, hold=asker_idle),
        reply.text("Answered.", match=reply.answering(question)),
        reply.text("Friday it is.", match=reply.answering(answer)),
    )
    page = page_for(person, path=f"/c/{asker_id}")
    expect_in_chat(page, "Planning ready.")

    send(page, prompt)

    expect_in_chat(page, "Asked the calendar.")
    expect(stop_button(page)).to_have_count(0, timeout=REPLY_TIMEOUT_MS)
    expect(tool_step(conversation(page), "Sent message", "send_message")).to_be_visible()
    asker_idle.set()
    answer_row = message_row(page, helper_title, answer)
    expect(answer_row, "the answer never showed in the asking chat").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect_in_chat(page, "Friday it is.")
    expect(conversation(page).get_by_text(answer, exact=True)).to_have_count(0)
    answer_row.click()
    expect(conversation(page).get_by_text(answer, exact=True)).to_be_visible()

    conversation(page).get_by_role("button", name="Open thread").click()
    helper_tab = page.get_by_role("tabpanel", name=re.compile(re.escape(helper_title)))
    expect(helper_tab).to_contain_text("Answered.", timeout=REPLY_TIMEOUT_MS)
    expect(
        helper_tab.get_by_role("button", name=re.compile(re.escape(f"{asker_title} · {question}")))
    ).to_be_visible()


def test_a_subagent_question_shows_in_the_lead_chat_before_its_result(
    subagents_on, page_for, make_user, upstream
):
    person = make_user()
    prompt, task = unique("hand this over"), unique("draft the agenda")
    sub_title = unique("Demo agenda")
    question = unique("Should the agenda include the budget?")
    lead_idle = threading.Event()
    with person.client() as client:
        lead_id = saved_chat(
            client, unique("start the lead"), "Ready.", unique("Demo lead"), upstream
        )
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": task, "title": sub_title}, match=reply.answering(prompt)
        ),
        message_to(lead_id, question, task, hold=lead_idle),
        reply.text("The agenda is drafted.", match=reply.answering(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text("Yes, include the budget.", match=reply.answering(question)),
        reply.text("The agenda is in.", match=report_of(task)),
    )
    page = page_for(person, path=f"/c/{lead_id}")
    expect_in_chat(page, "Ready.")

    send(page, prompt)

    expect_in_chat(page, "Handed over.")
    expect(stop_button(page)).to_have_count(0, timeout=REPLY_TIMEOUT_MS)
    lead_idle.set()
    expect(
        message_row(page, sub_title, question), "the sub-agent's question never showed"
    ).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect_in_chat(page, "Yes, include the budget.")
    result_row = conversation(page).get_by_role(
        "button", name=re.compile(re.escape(f"Sub-agent finished · {sub_title}"))
    )
    expect(result_row).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect_in_chat(page, "The agenda is in.")
    result_row.click()
    expect(conversation(page)).to_contain_text("The agenda is drafted.")
