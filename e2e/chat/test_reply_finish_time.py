"""Journey: an assistant reply shows the time it finished, not the time its prompt was sent.

New in 0.13.1 (#33895). A reply used to carry the same time as the message above it however long
it took. It now shows when it finished: live as soon as it is done, and after the chat is opened
again from the finish time Open WebUI stores with every reply (`response_ended_at`). A reply that
failed shows its finish time after a reload too. A reply stored before v0.12.5, which has no
finish time, still shows the time it was sent.

Discriminates: passes on dev e8c0ad85e. On a build from before #33895 (6113e15af, a branch off
dev of Nov 5) with the e8c0ad85e backend the live, stored and failed tests fail (the reply shows
its send time); the reply without a finish time passes on both.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), a stored chat
dragged into a tab beside a new chat. The chat is opened again by the address the page shows. On
dev c15f28649 a side chat's messages showed no time and a failed reply showed no error (no filed
issue for either), so the side variants are red there; they pass on dev e791cbdde, with PR
open-webui/open-webui#34064.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from threading import Event
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# a gap the page's own clock must pass while the reply is held, so the two times differ
GAP_SECONDS = 2
HOUR = 3600


def shown_time(message: Locator) -> float:
    """The moment a message's time element stands for, in epoch seconds."""
    stamp = message.locator("time").first
    expect(stamp).to_have_attribute("datetime", re.compile(r".+"))
    return datetime.fromisoformat(
        stamp.get_attribute("datetime").replace("Z", "+00:00")
    ).timestamp()


def user_message(page: Page | ChatSurface) -> Locator:
    return conversation(page).locator("[id^='message-'].user-message").last


def assistant_message(page: Page | ChatSurface) -> Locator:
    return conversation(page).locator("[id^='message-']:has(.chat-assistant)").last


def reopen(page_for, account, page: Page) -> Page:
    return page_for(account, path=urlparse(page.url).path)


def test_a_reply_that_took_a_while_shows_when_it_finished(page_for, chat_for, make_user, upstream):
    account = make_user()
    chat = chat_for(account)
    question = f"how long is the crossing? {uuid.uuid4().hex[:6]}"
    released = Event()
    upstream.queue(reply.text("About an hour.", hold=released, match=reply.answering(question)))
    try:
        send(chat, question)
        sent_at = shown_time(user_message(chat))
        chat.page.wait_for_function(f"Date.now() / 1000 >= {sent_at + GAP_SECONDS}")
    finally:
        released.set()
    expect_reply(chat, "About an hour.")

    finished_at = shown_time(assistant_message(chat))
    assert finished_at >= sent_at + GAP_SECONDS, (
        f"the reply shows {finished_at}, not after the {sent_at + GAP_SECONDS} it finished by"
    )
    reopened = reopen(page_for, account, chat.page)
    expect_reply(reopened, "About an hour.")
    assert shown_time(assistant_message(reopened)) >= sent_at + GAP_SECONDS
    assert shown_time(user_message(reopened)) == sent_at


def test_a_stored_reply_shows_its_recorded_finish_time(chat_for, make_user):
    account = make_user()
    sent_at = int(time.time()) - 2 * HOUR
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "When does the ferry leave?", "timestamp": sent_at},
                {
                    "role": "assistant",
                    "content": "At nine.",
                    "timestamp": sent_at,
                    "meta": {"response_ended_at": sent_at + HOUR},
                },
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "At nine.")

    assert shown_time(assistant_message(chat)) == sent_at + HOUR
    assert shown_time(user_message(chat)) == sent_at


def test_a_reply_without_a_finish_time_shows_when_it_was_sent(chat_for, make_user):
    account = make_user()
    sent_at = int(time.time()) - 2 * HOUR
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "When does the ferry leave?", "timestamp": sent_at},
                {"role": "assistant", "content": "At ten.", "timestamp": sent_at},
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "At ten.")

    assert shown_time(assistant_message(chat)) == sent_at


def test_a_failed_reply_shows_its_finish_time_after_a_reload(
    page_for, chat_for, make_user, upstream
):
    account = make_user()
    chat = chat_for(account)
    question = f"is the harbour open? {uuid.uuid4().hex[:6]}"
    released = Event()
    upstream.queue(
        reply.error(
            500, "the harbour office is closed", hold=released, match=reply.answering(question)
        )
    )
    try:
        send(chat, question)
        sent_at = shown_time(user_message(chat))
        chat.page.wait_for_function(f"Date.now() / 1000 >= {sent_at + GAP_SECONDS}")
    finally:
        released.set()
    expect_reply(chat, "the harbour office is closed")

    reopened = reopen(page_for, account, chat.page)
    expect_reply(reopened, "the harbour office is closed")
    assert shown_time(assistant_message(reopened)) >= sent_at + GAP_SECONDS
