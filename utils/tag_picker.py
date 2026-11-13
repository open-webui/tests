"""The tag picker an open chat's "Chat actions" menu opens under Tags.

The picker lists the chat's tags ticked and the account's other tags unticked; a click ticks or
unticks one, and a name typed into its search box is added with Enter. It draws its entries once
the chat's tags have loaded, after the menu opened.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input


def open_tag_picker(page: Page) -> Locator:
    """Open the open chat's Chat actions menu and its Tags entry; return the picker's menu."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Chat actions").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name=re.compile(r"^Tags\b")).click()
    picker = page.get_by_role("menu").filter(has=page.get_by_label("Find or create a tag"))
    expect(picker.get_by_label("Find or create a tag")).to_be_visible()
    return picker


def tag_entry(picker: Locator, tag: str) -> Locator:
    return picker.get_by_role("menuitemcheckbox", name=tag, exact=True)


def add_tag_by_typing(picker: Locator, tag: str) -> None:
    picker.get_by_label("Find or create a tag").fill(tag)
    picker.page.keyboard.press("Enter")
    # ticked only once the server has stored the tag
    expect(tag_entry(picker, tag)).to_have_attribute("aria-checked", "true")
