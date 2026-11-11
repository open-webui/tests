"""Regression: typing with an input method broke in the empty chat input under a follow-up.

Fix PR open-webui/open-webui#31393 (issue open-webui/open-webui#31372) in the chat input. After a
reply the first suggested follow-up sits as grey ghost text in the empty input. The ghost text
was written into the input's own line, so the first keystroke of an input method rebuilt that
line while the word was still being composed: in Chromium typing "ni" and picking the character
left "n" behind. The ghost text is now drawn over the empty input without being written into
it. Tab still accepts the suggestion.

A seeded chat has a reply that carries a follow-up. The composition is driven through the
browser's input protocol, the way a real input method does it.

Discriminates: passes on the dev a5bc78300 build; with the ghost text written into the input
again (the mutation build) the composed character arrives as "n" plus the character.

Both tests run on the saved chat in the main chat and dragged into a tab beside another chat
(`utils/chat_surface.py`), the only places a saved chat opens; they pass in the tab on dev
c15f28649 and on dev e791cbdde.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.chat_history import seed_chat
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT, ChatSurface
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLLOW_UP = "How do tides differ on other planets?"


def open_chat_with_follow_up(chat_for, make_user) -> ChatSurface:
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "tides?"},
                {"role": "assistant", "content": "Twice a day.", "followUps": [FOLLOW_UP]},
            ],
        )
    chat = chat_for(owner, chat_id=chat_id)
    expect(chat_input(chat).locator("p[data-suggestion]")).to_have_attribute(
        "data-suggestion", FOLLOW_UP
    )
    return chat


@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_composing_a_word_over_the_ghost_text_leaves_only_the_composed_word(chat_for, make_user):
    chat = open_chat_with_follow_up(chat_for, make_user)
    page = chat.page
    session = page.context.new_cdp_session(page)
    chat_input(chat).click()

    session.send("Input.imeSetComposition", {"text": "n", "selectionStart": 1, "selectionEnd": 1})
    # the ghost text goes as soon as the input holds anything, composing or not
    expect(chat_input(chat).locator("[data-suggestion]")).to_have_count(0)
    session.send("Input.imeSetComposition", {"text": "ni", "selectionStart": 2, "selectionEnd": 2})
    session.send("Input.insertText", {"text": "你"})

    expect(chat_input(chat), "the composed word was broken by the ghost text").to_have_text("你")


@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_tab_accepts_the_follow_up_suggestion(chat_for, make_user):
    chat = open_chat_with_follow_up(chat_for, make_user)
    chat_input(chat).click()

    chat.page.keyboard.press("Tab")

    expect(chat_input(chat)).to_have_text(FOLLOW_UP)
