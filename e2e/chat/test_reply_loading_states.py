"""Journey: what a person sees while a slow provider thinks, and when they stop a reply.

A provider that takes seconds before its first word leaves the reply empty: the page shows the
reply with a blinking cursor and the Stop button until the words come, then the cursor and Stop
go and the reply offers Regenerate. Stopping a reply before its first word ends it there, so the
answer the provider sends late never shows up, even after the next message has its own reply.
Stopping halfway keeps exactly what arrived, and a reload reads the same.

Discriminates: passes on dev ebc6add67; the waiting test fails on a build that only shows the
cursor once there are words, and the stop tests fail on a backend copy whose stop endpoint
leaves the reply running (the late answer shows and the stopped reply keeps growing).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.inflight import LAST_PIECE, SLOW_PIECES, wait_until_no_reply_runs
from utils.chat_ui import (
    conversation,
    expect_reply,
    last_reply,
    regenerate_buttons,
    replies,
    send,
    stop_button,
    typing_cursor,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST_WORD_DELAY = 4


@pytest.fixture
def chat_page(page_for, make_user):
    return page_for(make_user())


def test_the_reply_shows_as_loading_until_a_slow_providers_first_word(chat_page, upstream):
    prompt = "what is the tallest mountain?"
    answer = "Mount Everest is the tallest."
    upstream.queue(reply.text(answer, delay=FIRST_WORD_DELAY, match=reply.answering(prompt)))

    send(chat_page, prompt)

    waiting = last_reply(chat_page)
    expect(typing_cursor(waiting)).to_be_visible()
    expect(stop_button(chat_page)).to_be_visible()
    expect(waiting).to_have_text("")
    expect(regenerate_buttons(chat_page)).to_have_count(0)

    expect_reply(chat_page, answer)
    expect(typing_cursor(waiting)).to_have_count(0)
    expect(stop_button(chat_page)).to_be_hidden()
    expect(regenerate_buttons(chat_page).last).to_be_visible()


def test_a_reply_stopped_before_its_first_word_stays_empty(chat_page, upstream):
    slow_prompt, next_prompt = "write me a poem", "never mind, what time is it?"
    upstream.queue(
        reply.text("A late poem.", delay=FIRST_WORD_DELAY, match=reply.answering(slow_prompt)),
        reply.text("It is noon.", match=reply.answering(next_prompt)),
    )

    send(chat_page, slow_prompt)
    expect(typing_cursor(last_reply(chat_page))).to_be_visible()
    stop_button(chat_page).click()
    expect(stop_button(chat_page)).to_be_hidden()
    send(chat_page, next_prompt)
    expect_reply(chat_page, "It is noon.")

    chat_page.wait_for_timeout(FIRST_WORD_DELAY * 1000)  # past the moment the late answer starts
    expect(conversation(chat_page).get_by_text("A late poem.")).to_have_count(0)
    chat_page.reload()
    expect(replies(chat_page)).to_have_count(2)
    expect(conversation(chat_page).get_by_text("A late poem.")).to_have_count(0)


def test_a_reply_stopped_halfway_reads_the_same_after_a_reload(page_for, make_user, upstream):
    account = make_user()
    chat_page = page_for(account)
    prompt = "tell me a long story"
    upstream.queue(reply.text(SLOW_PIECES, chunk_delay=0.3, match=reply.answering(prompt)))

    send(chat_page, prompt)
    expect(last_reply(chat_page)).to_contain_text("part-2")
    stop_button(chat_page).click()
    expect(stop_button(chat_page)).to_be_hidden()
    chat_page.wait_for_timeout(1500)  # five more piece intervals
    stopped_text = last_reply(chat_page).inner_text()
    assert LAST_PIECE not in stopped_text

    next_piece = SLOW_PIECES[stopped_text.count("part-")].strip()
    with account.client() as client:
        wait_until_no_reply_runs(client, chat_page.url.rsplit("/", 1)[1])
    chat_page.reload()
    expect(last_reply(chat_page)).to_contain_text(stopped_text.strip())
    expect(last_reply(chat_page)).not_to_contain_text(next_piece)
