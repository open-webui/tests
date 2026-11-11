"""Regression: single line breaks in chat messages ran together after a reload or on a shared link.

Fix 808b90c03 (PR open-webui/open-webui#31869). Lines separated by one newline showed on separate
lines while chatting, but opening the chat by its URL, reloading it or reading it through a shared
link ran them into one line. The single-newline setting of the markdown renderer was only switched
on once the chat input mounted, so a message drawn before that rendered without it. The renderer
now has it on from the start.

Discriminates: passes on the dev b859124f9 build, fails on that build with 808b90c03 reverted
(the lines of a message opened by URL or shared link share one row).

The opened and reloaded chat tests also run with the chat dragged into a tab beside a new chat
(`utils/chat_surface.py`); the shared link is a page of its own.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION_LINES = ["Which birds", "winter by the lake", "this year?"]
ANSWER_LINES = ["Herons stay all winter", "Grebes arrive in November", "Geese pass through"]

# the top of the row each of the texts is drawn on
ROWS = """
(scope, texts) => texts.map((text) => {
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node && !node.textContent.includes(text)) node = walker.nextNode();
    const range = document.createRange();
    range.selectNodeContents(node);
    return Math.round(range.getBoundingClientRect().top);
})
"""


@pytest.fixture
def seeded(make_user):
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "\n".join(QUESTION_LINES)},
                {"role": "assistant", "content": "\n".join(ANSWER_LINES)},
            ],
        )
    return owner, chat_id


def expect_one_line_each(message: Locator, lines: list[str]) -> None:
    expect(message).to_contain_text(lines[-1])
    rows = message.evaluate(ROWS, lines)
    assert len(set(rows)) == len(lines), f"the lines share rows: {rows}"
    assert rows == sorted(rows), f"the lines are out of order: {rows}"


def expect_both_messages_broken_into_lines(page: Page | ChatSurface) -> None:
    expect_one_line_each(conversation(page).locator(".chat-user"), QUESTION_LINES)
    expect_one_line_each(conversation(page).locator(".chat-assistant"), ANSWER_LINES)


def test_a_chat_opened_by_its_url_shows_each_line_on_its_own_row(chat_for, seeded):
    owner, chat_id = seeded
    chat = chat_for(owner, chat_id=chat_id)

    expect_both_messages_broken_into_lines(chat)


def test_a_reloaded_chat_keeps_each_line_on_its_own_row(chat_for, seeded):
    owner, chat_id = seeded
    chat = chat_for(owner, chat_id=chat_id)
    expect(conversation(chat).locator(".chat-assistant")).to_be_visible()

    chat = chat.reload()

    expect_both_messages_broken_into_lines(chat)


def test_a_shared_chat_link_shows_each_line_on_its_own_row(page_for, seeded):
    owner, chat_id = seeded
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
    shared.raise_for_status()
    page = page_for(owner, path=f"/s/{shared.json()['share_id']}")

    expect_both_messages_broken_into_lines(page)
