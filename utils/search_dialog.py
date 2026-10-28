"""The search dialog, opened with Ctrl+K or the sidebar's Search entry, used the way a person does.

Retargeted for b4db56ae9, which rebuilt the dialog: the results are options in a list, the
selected one is repeated in a preview pane beside it, and an empty query lists pinned and recent
chats. Since 0a5c105ab the list is named by its caption, "Pinned & recent" or "Results".
`search` returns once the server has answered the typed query, so a count or an absence read
after it is about that query and not the list it replaces.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Locator, Page, expect

from utils.chat_ui import chat_input

SEARCH_PATH = "/api/v1/search"
RESULT_LIST_NAMES = re.compile(r"^(Pinned & recent|Results)$")


def search_dialog(page: Page) -> Locator:
    return page.get_by_role("dialog", name="Search workspace")


def search_box(dialog: Locator) -> Locator:
    return dialog.get_by_role("combobox", name="Search your workspace")


def open_search(page: Page) -> Locator:
    """Opens the dialog with the shortcut, which binds once the chat input renders."""
    expect(chat_input(page)).to_be_visible()
    page.keyboard.press("Control+K")
    dialog = search_dialog(page)
    expect(search_box(dialog)).to_be_visible()
    return dialog


def open_search_from_sidebar(page: Page) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name="Search", exact=True).first.click()
    dialog = search_dialog(page)
    expect(search_box(dialog)).to_be_visible()
    return dialog


def answers_query(text: str):
    def matches(response) -> bool:
        url = urlparse(response.url)
        return url.path == SEARCH_PATH and parse_qs(url.query).get("q", [""]) == [text]

    return matches


def search(dialog: Locator, text: str) -> None:
    """Types `text` into the search box and waits for the server's answer to it."""
    with dialog.page.expect_response(answers_query(text)):
        search_box(dialog).fill(text)


def results(dialog: Locator) -> Locator:
    return dialog.get_by_role("listbox", name=RESULT_LIST_NAMES).get_by_role("option")


def result(dialog: Locator, title: str) -> Locator:
    return results(dialog).filter(has_text=title)


def no_matches(dialog: Locator) -> Locator:
    return dialog.get_by_text("No matches found")


def preview(dialog: Locator) -> Locator:
    return dialog.get_by_role("region", name="Result preview")


def open_selected(dialog: Locator) -> None:
    """Opens the highlighted result with the footer's Open button, named with its Enter key."""
    dialog.get_by_role("button", name=re.compile(r"^Open\W*$")).click()


def answers_filter(name: str, value: str):
    def matches(response) -> bool:
        url = urlparse(response.url)
        return url.path == SEARCH_PATH and parse_qs(url.query).get(name) == [value]

    return matches


def choose_content_type(dialog: Locator, label: str, kind: str) -> None:
    """Picks `label` ("Chats", "Notes", ...) in the content type menu; waits for its answer."""
    dialog.get_by_role("button", name=re.compile(r"^Content type:")).click()
    choice = dialog.get_by_role("button", name=label, exact=True)
    expect(choice).to_be_visible()
    with dialog.page.expect_response(answers_filter("kind", kind)):
        choice.click()


def choose_view(dialog: Locator, label: str, sharing: str) -> None:
    """Picks `label` ("Created by you", ...) under Filters, View; waits for its answer."""
    page = dialog.page
    dialog.get_by_role("button", name="Filters").click()
    page.get_by_role("button", name=re.compile(r"^View")).click()
    choice = page.get_by_role("button", name=label, exact=True)
    expect(choice).to_be_visible()
    with page.expect_response(answers_filter("sharing", sharing)):
        choice.click()


def open_actions(dialog: Locator) -> Locator:
    """Opens the footer's Actions menu and returns the dialog, where its entries are buttons."""
    actions = dialog.get_by_role("button", name="Actions")
    actions.click()
    expect(actions).to_have_attribute("aria-expanded", "true")
    return dialog
