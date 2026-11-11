"""Regression: the unsent text of a saved chat was gone after a reload, in the main chat and in a
side chat (open-webui/open-webui#34023).

The page keeps the text typed into a chat's message box in the tab's session, per chat, and puts
it back when the chat opens. A saved chat put it back before its message box was shown, and the
box came up empty, then stored that empty text over the kept one. A new chat applied its text
once the box was shown and kept it.

Discriminates: the saved chat test fails on dev 262a8c6f7 (both message boxes empty after the
reload) and passes on the build of the fix 09f4625a4 (branch fix-unsent-text-lost-on-reload); the
new chat test passes on both.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import (
    PARENT_CHAT,
    ChatSurface,
    ask_aside,
    main_chat,
    reload_beside_the_chat,
)
from utils.chat_ui import REPLY_TIMEOUT_MS

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _type_and_wait_until_stored(chat: ChatSurface, text: str) -> None:
    chat.input.click()
    chat.page.keyboard.type(text)
    # the text is stored in the tab's session a moment after the last key
    chat.page.wait_for_function(
        "text => Object.values(sessionStorage).some((value) => value.includes(text))", arg=text
    )


def _unsent(place: str) -> str:
    return f"Not sent yet in the {place} {uuid.uuid4().hex[:6]}"


def test_the_unsent_text_of_a_saved_chat_and_its_side_chat_is_kept_over_a_reload(
    page_for, make_user, upstream
):
    """Red on dev 262a8c6f7 (#34023)."""
    # an admin has side chats without a permission
    author = make_user(role="admin")
    with author.client() as client:
        chat_id, _ = seed_chat(client, PARENT_CHAT)
    page = page_for(author, path=f"/c/{chat_id}")
    chat = main_chat(page)
    chat.expect_reply(PARENT_CHAT[1]["content"])
    question = f"How long is the tunnel? ({uuid.uuid4().hex[:6]})"
    upstream.queue(reply.text("5.3 kilometres.", match=reply.answering(question)))
    side = ask_aside(page, question)
    side.expect_reply("5.3 kilometres.")
    expect(side.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
    chat_text, side_text = _unsent("chat"), _unsent("side chat")
    _type_and_wait_until_stored(side, side_text)
    _type_and_wait_until_stored(chat, chat_text)

    side = reload_beside_the_chat(side, question)

    expect(chat.input, "the reload emptied the chat's message box").to_have_text(
        chat_text, timeout=REPLY_TIMEOUT_MS
    )
    expect(side.input, "the reload emptied the side chat's message box").to_have_text(
        side_text, timeout=REPLY_TIMEOUT_MS
    )


def test_the_unsent_text_of_a_new_chat_is_kept_over_a_reload(page_for, make_user):
    page: Page = page_for(make_user())
    chat = main_chat(page)
    text = _unsent("new chat")
    _type_and_wait_until_stored(chat, text)

    page.reload()

    expect(chat.input, "the reload emptied the new chat's message box").to_have_text(
        text, timeout=REPLY_TIMEOUT_MS
    )
