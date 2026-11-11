"""Journey: the chat follows a long reply as it streams, and gives the reader the scroll back.

While a long reply streams the chat keeps its newest line in view. Scrolling up while it streams
stops the following, so the reader keeps their place, and a "Scroll to bottom" button appears
that brings the end of the chat back into view and goes away again. A long chat opened from its
link opens at its end. With Response Auto-Scroll off in the account's settings the view stays
where it was while the reply grows below it.

On a slow device the chat stops following on its own partway through a reply
(open-webui/open-webui#32728): it takes its own scroll to the bottom for the reader scrolling up
whenever new text lands before the browser reports that scroll. Until that fix merges the
following test fails on a throttled CPU in some runs, more often on a busy machine.

Discriminates: on a throttled CPU under load the following test fails on the dev 079fb19e7 build
and passes five runs in a row on that build with the #32728 fix. It passes on the dev ebc6add67
build. In its mutation build (the `rendering-front` copy: the scroll that follows a streaming reply
removed and the button never shown) the following, the scroll-up and the opened-chat tests go red;
the setting-off test stays green there as a control.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), a stored chat
dragged into a tab beside a new chat. The wheel turns over the chat's own panel. On dev c15f28649
a side chat kept following its reply with Response Auto-Scroll off (no filed issue), so that side
variant is red there; it passes on dev e791cbdde, with PR open-webui/open-webui#34064.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, send, stop_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LINE_COUNT = 80
LONG_REPLY = [f"Line {number} of the long answer.\n\n" for number in range(1, LINE_COUNT + 1)]


def line(chat: ChatSurface, number: int) -> Locator:
    return conversation(chat).get_by_text(f"Line {number} of the long answer.", exact=True)


def scroll_to_bottom_button(chat: ChatSurface) -> Locator:
    return chat.panel.get_by_role("button", name="Scroll to bottom")


def start_long_reply(chat: ChatSurface, upstream, prompt: str) -> None:
    upstream.queue(reply.text(LONG_REPLY, chunk_delay=0.15, match=reply.answering(prompt)))
    send(chat, prompt)
    expect(line(chat, 1)).to_be_attached(timeout=REPLY_TIMEOUT_MS)


def wait_for_the_end(chat: ChatSurface) -> None:
    expect(line(chat, LINE_COUNT)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(chat)).to_be_hidden(timeout=REPLY_TIMEOUT_MS)


def scroll_up(chat: ChatSurface) -> None:
    box = chat.panel.bounding_box()
    assert box, "the chat is not on screen"
    chat.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 3)
    for _ in range(20):
        chat.page.mouse.wheel(0, -400)


def test_the_chat_keeps_the_newest_line_in_view_while_a_long_reply_streams(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())

    start_long_reply(chat, upstream, "tell me a very long story")

    expect(line(chat, 50)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    expect(stop_button(chat), "the reply already ended").to_be_visible()
    expect(line(chat, 50)).to_be_in_viewport()
    expect(line(chat, 1)).not_to_be_in_viewport()
    wait_for_the_end(chat)
    expect(line(chat, LINE_COUNT)).to_be_in_viewport()
    expect(scroll_to_bottom_button(chat)).to_be_hidden()


def test_scrolling_up_while_it_streams_keeps_the_place_and_the_button_brings_the_end_back(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())
    prompt = "tell me another very long story"
    start_long_reply(chat, upstream, prompt)
    expect(line(chat, 30)).to_be_attached(timeout=REPLY_TIMEOUT_MS)
    question = conversation(chat).get_by_text(prompt)
    expect(question).not_to_be_in_viewport()

    scroll_up(chat)

    expect(question).to_be_in_viewport()
    expect(scroll_to_bottom_button(chat)).to_be_visible()
    wait_for_the_end(chat)
    expect(question).to_be_in_viewport()
    expect(line(chat, LINE_COUNT)).not_to_be_in_viewport()

    scroll_to_bottom_button(chat).click()

    expect(line(chat, LINE_COUNT)).to_be_in_viewport()
    expect(scroll_to_bottom_button(chat)).to_be_hidden()


def test_a_long_chat_opens_at_its_end_and_the_button_returns_there(chat_for, make_user):
    """open-webui/open-webui#33877: on a wide window the chat opened at its first message."""
    account = make_user()
    turns = []
    for number in range(1, 5):
        paragraphs = [f"Answer number {number}, part {part}." for part in range(1, 13)]
        turns.append({"role": "user", "content": f"Question number {number}"})
        turns.append({"role": "assistant", "content": "\n\n".join(paragraphs)})
    with account.client() as client:
        chat_id, _ = seed_chat(client, turns)
    chat = chat_for(account, chat_id=chat_id)

    last_answer = conversation(chat).get_by_text("Answer number 4, part 12.", exact=True)
    first_question = conversation(chat).get_by_text("Question number 1", exact=True)
    expect(last_answer).to_be_in_viewport()
    expect(first_question).not_to_be_in_viewport()
    expect(scroll_to_bottom_button(chat)).to_be_hidden()
    scroll_up(chat)
    expect(first_question).to_be_in_viewport()
    scroll_to_bottom_button(chat).click()
    expect(last_answer).to_be_in_viewport()
    expect(scroll_to_bottom_button(chat)).to_be_hidden()


def test_with_response_auto_scroll_off_the_view_stays_while_the_reply_grows(
    chat_for, make_user, upstream
):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"scrollOnResponseGeneration": False}},
        )
    saved.raise_for_status()
    chat = chat_for(account)
    prompt = "tell me a long story without following"

    start_long_reply(chat, upstream, prompt)

    wait_for_the_end(chat)
    expect(conversation(chat).get_by_text(prompt)).to_be_in_viewport()
    expect(line(chat, 1)).to_be_in_viewport()
    expect(line(chat, LINE_COUNT)).not_to_be_in_viewport()
