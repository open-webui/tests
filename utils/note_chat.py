"""The chat of an open note: its message box, found the way a person sees it.

Since dev da850248d a note opens as a tab next to the chat tab, whose own message box stays in
the page unseen, so the note chat's box is found by its name among the boxes that show. Since dev
f80331e08 the box floats at the bottom of every open note, with no Chat button to open it, under
an Expand button that grows the conversation above it; sending a message expands it too.
Since dev 6db63cddb the note records audio from the box's + menu (More), where Capture opens
Record and Capture Audio; in a note the recording goes into the note.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import REPLY_TIMEOUT_MS


def open_note_chat(page: Page) -> Locator:
    """The open note's chat message box, once it shows."""
    expect(note_chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return note_chat_input(page)


def expand_note_chat(page: Page) -> Locator:
    """Grow the note chat with its Expand button, which shows its suggested prompts; its box."""
    box = open_note_chat(page)
    expand = page.get_by_role("main").get_by_role("button", name="Expand", exact=True)
    expect(expand).to_have_attribute("aria-expanded", "false")
    expand.click()
    expect(
        page.get_by_role("main").get_by_role("button", name="Collapse", exact=True)
    ).to_be_visible()
    return box


def note_chat_input(page: Page) -> Locator:
    return page.get_by_role("main").get_by_role("textbox", name="Send a Message", exact=True)


def open_capture_menu(page: Page) -> Locator:
    """Open Capture in the note chat's + menu; the menu listing Record and Capture Audio."""
    open_note_chat(page)
    message_box = page.get_by_role("textbox", name="Send a Message", exact=True)
    composer = page.get_by_role("main").locator("form").filter(has=message_box)
    composer.get_by_role("button", name="More", exact=True).click()
    menu = page.get_by_role("menu")
    menu.get_by_role("menuitem", name="Capture", exact=True).click()
    expect(menu.get_by_role("menuitem", name="Record", exact=True)).to_be_visible()
    return menu


def send_in_note_chat(page: Page, text: str) -> None:
    box = note_chat_input(page)
    expect(box).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    box.click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")
