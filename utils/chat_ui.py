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
