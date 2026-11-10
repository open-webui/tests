"""Regression: on a phone, opening a chat put the cursor in the message box and raised the keyboard.

Issue open-webui/open-webui#34059. Opening a chat from the sidebar drawer or by its link, and
switching to another chat, focused the message box. A focused text field is what raises a phone's
on-screen keyboard, which then covered the conversation the person came to read. On a phone-sized
screen an opened chat now leaves the message box alone until it is tapped. A new chat still starts
with the cursor in the message box, and it stays there after a message is sent, also in a new chat
started with the top bar's New Chat button inside an opened chat.

Discriminates: the opening test is red on the dev c15f28649 build (the message box holds the focus
after the chat opens) and passes on the build of the fix; the new chat tests pass on both.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import slow_machine
from harness import upstream as reply
from harness.chat import ask
from utils.chat_ui import chat_input, expect_reply
from utils.phone import PHONE, expect_on_screen, send_by_tapping, tap_on_screen

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

KEYBOARD_UP = "the message box took the focus, which raises the keyboard (#34059)"

# the tag and name of the focused element when a keyboard would come up for it, else null
FOCUSED_TEXT_FIELD = """() => {
    const element = document.activeElement;
    if (!element) return null;
    const typesText = element.isContentEditable || ['INPUT', 'TEXTAREA'].includes(element.tagName);
    const name = element.id || element.getAttribute('placeholder') || '';
    return typesText ? `${element.tagName} ${name}` : null;
}"""


def saved_chat(account, upstream, answer: str) -> tuple[str, str]:
    """A chat of the account with one answered question, as its id and its title."""
    title = f"Kestrel {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(answer, match=reply.answering(title)))
    with account.client() as client:
        turn, _ = ask(client, f"{title}: how do they hunt?")
        renamed = client.post(f"/api/v1/chats/{turn.chat_id}", json={"chat": {"title": title}})
        assert renamed.status_code == 200, f"renaming the chat failed: {renamed.text}"
    return turn.chat_id, title


def on_a_phone(page_for, account, path: str = "/") -> Page:
    page = page_for(account, path=path, **PHONE)
    expect_on_screen(chat_input(page))
    return page


def open_from_the_drawer(page: Page, title: str) -> None:
    tap_on_screen(page.get_by_role("button", name="Open Sidebar").last)
    chats = page.get_by_role("navigation", name="Chat history")
    tap_on_screen(chats.get_by_role("link", name=title))


def focused_text_field_once_settled(page: Page) -> str | None:
    # the focus an opening chat sets lands shortly after the chat shows
    page.wait_for_timeout(slow_machine.scaled(1) * 1000)
    return page.evaluate(FOCUSED_TEXT_FIELD)


def ask_by_tapping(page: Page, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send_by_tapping(page, question)
    expect_reply(page, answer)


def test_opening_a_chat_on_a_phone_leaves_the_message_box_unfocused(page_for, make_user, upstream):
    account = make_user()
    first_id, first_title = saved_chat(account, upstream, "They hover in the wind.")
    second_id, second_title = saved_chat(account, upstream, "They dive from a perch.")
    page = on_a_phone(page_for, account)
    # the new chat page focuses the message box once after load; let that land first
    expect(chat_input(page)).to_be_focused()

    open_from_the_drawer(page, first_title)
    expect(page).to_have_url(re.compile(first_id))
    expect_reply(page, "They hover in the wind.")
    focused = focused_text_field_once_settled(page)
    assert focused is None, f"opened from the sidebar, focus on {focused}: {KEYBOARD_UP}"

    open_from_the_drawer(page, second_title)
    expect(page).to_have_url(re.compile(second_id))
    expect_reply(page, "They dive from a perch.")
    focused = focused_text_field_once_settled(page)
    assert focused is None, f"switched to another chat, focus on {focused}: {KEYBOARD_UP}"

    by_link = on_a_phone(page_for, account, path=f"/c/{first_id}")
    expect_reply(by_link, "They hover in the wind.")
    focused = focused_text_field_once_settled(by_link)
    assert focused is None, f"opened by its link, focus on {focused}: {KEYBOARD_UP}"

    tap_on_screen(chat_input(by_link))
    expect(chat_input(by_link)).to_be_focused()


def test_a_new_chat_on_a_phone_keeps_the_cursor_in_the_message_box(page_for, make_user, upstream):
    page = on_a_phone(page_for, make_user())
    expect(chat_input(page)).to_be_focused()

    ask_by_tapping(page, upstream, f"first {uuid.uuid4().hex[:6]}", "The first answer.")
    expect(chat_input(page)).to_be_focused()
    ask_by_tapping(page, upstream, f"second {uuid.uuid4().hex[:6]}", "The second answer.")
    expect(chat_input(page)).to_be_focused()

    tap_on_screen(page.get_by_role("button", name="Open Sidebar").last)
    chats = page.get_by_role("navigation", name="Chat history")
    # the logo link is named New Chat too; this is the labelled entry under it
    tap_on_screen(chats.get_by_role("link", name="New Chat").filter(has_text="New Chat"))
    expect(page.get_by_text("The second answer.")).to_have_count(0)
    expect(chat_input(page)).to_be_focused()


def test_a_new_chat_from_the_top_bar_keeps_the_cursor_after_its_first_message(
    page_for, make_user, upstream
):
    account = make_user()
    chat_id, _ = saved_chat(account, upstream, "They hover in the wind.")
    page = on_a_phone(page_for, account, path=f"/c/{chat_id}")
    expect_reply(page, "They hover in the wind.")

    tap_on_screen(page.get_by_role("button", name="New Chat", exact=True))
    expect(page.get_by_text("They hover in the wind.")).to_have_count(0)
    expect(chat_input(page)).to_be_focused()

    ask_by_tapping(page, upstream, f"first {uuid.uuid4().hex[:6]}", "The first answer.")
    expect(chat_input(page)).to_be_focused()
    ask_by_tapping(page, upstream, f"second {uuid.uuid4().hex[:6]}", "The second answer.")
    expect(chat_input(page)).to_be_focused()
