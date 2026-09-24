"""The everyday chat journeys, driven through the browser against the scripted model.

Each test signs in as a fresh account, so no test sees another test's chats, and scripts the
exact reply the model gives, so every assertion is about what the page does with it.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import (
    conversation,
    expect_reply,
    last_reply,
    replies,
    send,
    stop_button,
)

pytestmark = [pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def chat_page(page_for, make_user):
    page = page_for(make_user())
    page.goto("/")
    return page


def test_a_sent_message_gets_its_reply_and_both_survive_a_reload(chat_page, upstream):
    upstream.queue(reply.text("Paris is the capital of France."))
    send(chat_page, "What is the capital of France?")
    expect_reply(chat_page, "Paris is the capital of France.")
    expect(chat_page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))

    chat_page.reload()
    expect(conversation(chat_page).get_by_text("What is the capital of France?")).to_be_visible()
    expect_reply(chat_page, "Paris is the capital of France.")


def test_a_follow_up_is_sent_with_the_earlier_turn(chat_page, upstream):
    upstream.queue(reply.text("first answer"), reply.text("second answer"))
    send(chat_page, "first question")
    expect_reply(chat_page, "first answer")
    send(chat_page, "second question")
    expect_reply(chat_page, "second answer")

    sent = [entry["content"] for entry in upstream.chat_requests()[-1]["messages"]]
    assert sent[-3:] == ["first question", "first answer", "second question"]


def test_stop_keeps_the_partial_reply_and_the_next_message_still_sends(chat_page, upstream):
    pieces = [f"part-{index} " for index in range(30)]
    upstream.queue(reply.text(pieces, chunk_delay=0.3), reply.text("fresh reply"))
    send(chat_page, "tell me a long story")
    expect(last_reply(chat_page)).to_contain_text("part-1", timeout=30_000)
    stop_button(chat_page).click()
    expect(stop_button(chat_page)).to_be_hidden()

    stopped_text = last_reply(chat_page).inner_text()
    chat_page.wait_for_timeout(1500)  # five more chunk intervals
    assert last_reply(chat_page).inner_text() == stopped_text
    assert "part-29" not in stopped_text

    send(chat_page, "and now something else")
    expect_reply(chat_page, "fresh reply")


def test_regenerate_replaces_the_reply_with_a_new_one(chat_page, upstream):
    upstream.queue(reply.text("first try"), reply.text("second try"))
    send(chat_page, "give me a name")
    expect_reply(chat_page, "first try")

    last_reply(chat_page).hover()
    conversation(chat_page).get_by_role("button", name="Regenerate").last.click()
    chat_page.get_by_text("Try Again").click()
    expect_reply(chat_page, "second try")
    expect(replies(chat_page)).to_have_count(1)


def test_editing_a_sent_message_resends_it(chat_page, upstream):
    upstream.queue(reply.text("answer to the typo"), reply.text("answer to the fix"))
    send(chat_page, "wht is 2+2")
    expect_reply(chat_page, "answer to the typo")

    question = conversation(chat_page).locator(".chat-user").last
    question.hover()
    question.get_by_role("button", name="Edit").click()
    editor = question.locator("textarea")
    editor.fill("what is 2+2")
    chat_page.locator("#confirm-edit-message-button").click()

    expect_reply(chat_page, "answer to the fix")
    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == "what is 2+2"


def test_reasoning_is_folded_away_above_the_answer(chat_page, upstream):
    upstream.queue(reply.text("the answer", reasoning="a private chain of thought"))
    send(chat_page, "think first")
    expect_reply(chat_page, "the answer")
    expect(last_reply(chat_page).get_by_text("Thought for")).to_be_visible()
    thoughts = conversation(chat_page).get_by_text("a private chain of thought")
    expect(thoughts).to_be_hidden()

    last_reply(chat_page).get_by_text("Thought for").click()
    expect(thoughts).to_be_visible()


def test_a_tool_call_runs_and_the_answer_follows_it(chat_page, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is now."))
    send(chat_page, "what time is it?")
    expect_reply(chat_page, "It is now.")

    replayed = upstream.chat_requests()[-1]["messages"]
    assert any(
        entry["role"] == "tool" and "current_timestamp" in entry["content"] for entry in replayed
    )
