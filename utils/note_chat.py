"""The chat beside an open note: its message box, found the way a person sees it.

Since dev da850248d a note opens as a tab next to the chat tab, whose own message box stays in
the page unseen, so the note chat's box is found by its name among the boxes that show.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import REPLY_TIMEOUT_MS


def open_note_chat(page: Page) -> Locator:
    """Open the chat beside the note with the Chat button in its header; its message box."""
    page.get_by_role("main").get_by_role("button", name="Chat", exact=True).click()
    expect(note_chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return note_chat_input(page)


def note_chat_input(page: Page) -> Locator:
    return page.get_by_role("main").get_by_role("textbox", name="Send a Message", exact=True)


def send_in_note_chat(page: Page, text: str) -> None:
    box = note_chat_input(page)
    expect(box).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    box.click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")
