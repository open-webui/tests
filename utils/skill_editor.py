"""The skill editor's file list and code editor, found the way a person finds them."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page


def code_editor(page: Page) -> Locator:
    # the code editor has no accessible name; its editable area is the only one in the page
    return page.get_by_role("main").locator(".cm-content")


def replace_text(page: Page, text: str) -> None:
    code_editor(page).click()
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.press("Backspace")
    page.keyboard.insert_text(text)


def file_row(page: Page, name: str) -> Locator:
    """The row of the file or folder `name`; its text is the name, then the size for a file."""
    rows = page.get_by_role("main").locator("li[data-file-row]")
    return rows.filter(has_text=re.compile(rf"^\s*{re.escape(name)}(?![\w.-])"))


def open_row(page: Page, name: str) -> None:
    file_row(page, name).get_by_role("button").first.click()


def row_menu(page: Page, name: str, item: str) -> None:
    row = file_row(page, name)
    row.hover()
    row.get_by_label("More", exact=True).click()
    page.get_by_role("button", name=item, exact=True).click()


def actions(page: Page, item: str) -> None:
    page.get_by_role("main").get_by_label("Actions", exact=True).click()
    page.get_by_role("menu").get_by_role("button", name=item, exact=True).click()
