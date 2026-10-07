"""Journey: the chat follows a long reply as it streams, and gives the reader the scroll back.

While a long reply streams the chat keeps its newest line in view. Scrolling up while it streams
stops the following, so the reader keeps their place, and a "Scroll to bottom" button appears
that brings the end of the chat back into view and goes away again. A long chat opened from its
link opens at its end. With Response Auto-Scroll off in the account's settings the view stays
where it was while the reply grows below it.

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: the scroll that follows a streaming reply removed and the button never shown) the
following, the scroll-up and the opened-chat tests go red; the setting-off test stays green there
as a control.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, send, stop_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LINE_COUNT = 80
LONG_REPLY = [f"Line {number} of the long answer.\n\n" for number in range(1, LINE_COUNT + 1)]


def line(page: Page, number: int) -> Locator:
    return conversation(page).get_by_text(f"Line {number} of the long answer.", exact=True)


def scroll_to_bottom_button(page: Page) -> Locator:
    return page.get_by_role("button", name="Scroll to bottom")


def start_long_reply(page: Page, upstream, prompt: str) -> None:
    upstream.queue(reply.text(LONG_REPLY, chunk_delay=0.15, match=reply.answering(prompt)))
    send(page, prompt)
    expect(line(page, 1)).to_be_attached(timeout=REPLY_TIMEOUT_MS)


def wait_for_the_end(page: Page) -> None:
    expect(line(page, LINE_COUNT)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(page)).to_be_hidden(timeout=REPLY_TIMEOUT_MS)


def scroll_up(page: Page) -> None:
    viewport = page.viewport_size
    page.mouse.move(viewport["width"] / 2, viewport["height"] / 3)
    for _ in range(20):
        page.mouse.wheel(0, -400)


def test_the_chat_keeps_the_newest_line_in_view_while_a_long_reply_streams(
    page_for, make_user, upstream
):
    page = page_for(make_user())

    start_long_reply(page, upstream, "tell me a very long story")

    expect(line(page, 50)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(page), "the reply already ended").to_be_visible()
    expect(line(page, 50)).to_be_in_viewport()
    expect(line(page, 1)).not_to_be_in_viewport()
    wait_for_the_end(page)
    expect(line(page, LINE_COUNT)).to_be_in_viewport()
    expect(scroll_to_bottom_button(page)).to_be_hidden()


def test_scrolling_up_while_it_streams_keeps_the_place_and_the_button_brings_the_end_back(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    prompt = "tell me another very long story"
    start_long_reply(page, upstream, prompt)
    expect(line(page, 30)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    question = conversation(page).get_by_text(prompt)
    expect(question).not_to_be_in_viewport()

    scroll_up(page)

    expect(question).to_be_in_viewport()
    expect(scroll_to_bottom_button(page)).to_be_visible()
    wait_for_the_end(page)
    expect(question).to_be_in_viewport()
    expect(line(page, LINE_COUNT)).not_to_be_in_viewport()

    scroll_to_bottom_button(page).click()

    expect(line(page, LINE_COUNT)).to_be_in_viewport()
    expect(scroll_to_bottom_button(page)).to_be_hidden()


def test_a_long_chat_opens_at_its_end_and_the_button_returns_there(page_for, make_user):
    account = make_user()
    turns = []
    for number in range(1, 5):
        paragraphs = [f"Answer number {number}, part {part}." for part in range(1, 13)]
        turns.append({"role": "user", "content": f"Question number {number}"})
        turns.append({"role": "assistant", "content": "\n\n".join(paragraphs)})
    with account.client() as client:
        chat_id, _ = seed_chat(client, turns)
    page = page_for(account)

    page.goto(f"/c/{chat_id}")

    last_answer = conversation(page).get_by_text("Answer number 4, part 12.", exact=True)
    first_question = conversation(page).get_by_text("Question number 1", exact=True)
    expect(last_answer).to_be_in_viewport()
    expect(first_question).not_to_be_in_viewport()
    expect(scroll_to_bottom_button(page)).to_be_hidden()
    scroll_up(page)
    expect(first_question).to_be_in_viewport()
    scroll_to_bottom_button(page).click()
    expect(last_answer).to_be_in_viewport()
    expect(scroll_to_bottom_button(page)).to_be_hidden()


def test_with_response_auto_scroll_off_the_view_stays_while_the_reply_grows(
    page_for, make_user, upstream
):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"scrollOnResponseGeneration": False}},
        )
    saved.raise_for_status()
    page = page_for(account)
    prompt = "tell me a long story without following"

    start_long_reply(page, upstream, prompt)

    wait_for_the_end(page)
    expect(conversation(page).get_by_text(prompt)).to_be_in_viewport()
    expect(line(page, 1)).to_be_in_viewport()
    expect(line(page, LINE_COUNT)).not_to_be_in_viewport()
