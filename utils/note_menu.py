"""The Note Menu of the open note's editor, the "..." button in its header, opened by its name."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect


def open_note_menu(page: Page) -> Locator:
    page.get_by_role("main").get_by_role("button", name="Note Menu", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Download")).to_be_visible()
    return menu
