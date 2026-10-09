"""Driving the chat page the way a person does: type, press Enter, read what appears."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

REPLY_TIMEOUT_MS = 30_000


def chat_input(page: Page) -> Locator:
    return page.locator("#chat-input")


def send(page: Page, text: str) -> None:
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    chat_input(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def link_dialog(page: Page) -> Locator:
    """The dialog a chat link asks in before it loads its `load-url` or `youtube` page or calls."""
    return page.get_by_role("dialog", name="Open link")


def conversation(page: Page) -> Locator:
    return page.get_by_label("Chat Conversation")


def replies(page: Page) -> Locator:
    return conversation(page).locator(".chat-assistant")


def last_reply(page: Page) -> Locator:
    return replies(page).last


def expect_reply(page: Page, text: str) -> None:
    expect(last_reply(page)).to_contain_text(text, timeout=REPLY_TIMEOUT_MS)


def stop_button(page: Page) -> Locator:
    return page.get_by_role("button", name="Stop")


def regenerate_buttons(page: Page) -> Locator:
    """The Regenerate button under each finished reply."""
    return conversation(page).get_by_label("Regenerate", exact=True)


def typing_cursor(reply: Locator) -> Locator:
    """The blinking cursor a reply shows while it is still being written."""
    return reply.locator(".animate-cursor-pulse")
