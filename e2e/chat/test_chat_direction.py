"""Journey: the chat direction setting changes the layout of the messages and the composer.

Interface settings offer Auto, LTR and RTL as one button that cycles through them. In RTL the
user's message moves from the right edge to the left one, the reply's text moves toward the
avatar's new side and the composer reads right to left; cycling on to Auto puts everything back.
The choice is saved with the account, so a reload keeps it and another account is unaffected.

Discriminates: passes on dev 176d31d1d; in a frontend copy, dropping the direction from the user
message row turns the user message test red, dropping it from the reply row turns the reply test
red, dropping it from the composer turns the composer test red, and the settings modal opening on
Auto whatever was saved turns the reload test red.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def chat_page(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    upstream.queue(reply.text("plain english answer"))
    send(chat, "hello there")
    expect_reply(chat, "plain english answer")
    return chat


def _open_direction_button(page: Page) -> Locator:
    open_settings(page)
    page.get_by_role("tab", name="Interface").click()
    return page.get_by_role("button", name="Chat direction")


def _choose(chat: ChatSurface, mode: str) -> None:
    page = chat.page
    button = _open_direction_button(page)
    for _ in range(3):
        if re.search(mode, button.inner_text()):
            break
        button.click()
    expect(button).to_contain_text(mode)
    page.get_by_role("dialog").get_by_role("button", name="Back", exact=True).click()
    expect(page.get_by_role("dialog")).to_be_hidden()


def _composer(chat: ChatSurface) -> Locator:
    # the editor inside follows its text, so the box around it carries the setting; the main
    # chat's box is #message-input-container, a side tab's has an id of its own
    return chat.panel.locator("[data-composer-id]")


def _left_edge(locator: Locator) -> float:
    return locator.evaluate("element => element.getBoundingClientRect().left")


def _message_text(chat: ChatSurface, role: str) -> Locator:
    return conversation(chat).locator(f".chat-{role}").locator("p").first


def test_rtl_moves_the_users_message_from_the_right_edge_to_the_left(chat_page):
    message = _message_text(chat_page, "user")
    right_side = _left_edge(message)

    _choose(chat_page, "RTL")

    assert _left_edge(message) < right_side / 2, "the message stayed on the right"


def test_rtl_moves_the_replys_text_toward_the_avatars_new_side(chat_page):
    text = last_reply(chat_page).locator("p").first
    avatar_on_left = _left_edge(text)

    _choose(chat_page, "RTL")

    assert _left_edge(text) < avatar_on_left, "the reply did not shift with the avatar"


def test_rtl_makes_the_composer_read_right_to_left_and_auto_restores_it(chat_page):
    expect(_composer(chat_page)).to_have_css("direction", "ltr")

    _choose(chat_page, "RTL")
    expect(_composer(chat_page)).to_have_css("direction", "rtl")

    _choose(chat_page, "Auto")
    expect(_composer(chat_page)).to_have_css("direction", "ltr")


def test_the_choice_survives_a_reload_and_is_the_accounts_own(chat_page, make_user, chat_for):
    _choose(chat_page, "RTL")
    expect(_composer(chat_page)).to_have_css("direction", "rtl")

    chat_page = chat_page.reload()
    expect(_composer(chat_page)).to_have_css("direction", "rtl")
    expect(_open_direction_button(chat_page.page)).to_contain_text("RTL")

    other = chat_for(make_user())
    expect(chat_input(other)).to_be_visible()
    expect(_composer(other)).to_have_css("direction", "ltr")
    expect(_open_direction_button(other.page)).to_contain_text("Auto")
