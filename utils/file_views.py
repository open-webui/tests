"""A file opened from a terminal's file browser: getting to it, its toolbar and what it shows.

`files_of` connects an Open Terminal for a person and opens Files in their new chat, inside a
folder of the terminal's home when one is named (`connect_for` and `files_in` are its two halves,
for a page that is set up in between). A clicked file opens in a tab of its own, so
`file_browser` is always the one on screen: `open_file` clicks a file in the list and waits for
its tab, `back_to` returns to the folder's list, `tab` is a tab of the workspace by its name.
The list's rows and menus are `utils/file_browser.py`'s.
`comment_shown_by` rests on a comment marker of a document and returns the comment that opens.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from harness.terminal_server import configure_terminals, read_grant
from utils.cached_chat import pick_terminal
from utils.file_browser import file_browser


def tab(page: Page, name: str) -> Locator:
    return page.get_by_role("tab", name=re.compile(rf"^{re.escape(name)}\b"))


def rest_elsewhere(page: Page) -> None:
    """Take the pointer off the list, where it would open a row's hover preview over the rows."""
    page.mouse.move(0, 0)


def enter(browser: Locator, folder: str) -> None:
    browser.get_by_text(folder, exact=True).click()
    expect(browser.get_by_role("button", name=folder, exact=True)).to_be_visible()
    rest_elsewhere(browser.page)


def connect_for(admin, person, terminal, **fields) -> str:
    """The admin connects `terminal` for `person` (after `preserve`); returns its name."""
    connection = terminal.connection(config={"access_grants": [read_grant(person.id)]}, **fields)
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection["name"]


def files_in(page: Page, terminal_name: str, folder: str | None = None) -> Locator:
    """Pick the terminal in the page's chat and open its file browser, in `folder` of its home."""
    pick_terminal(page, terminal_name)
    browser = file_browser(page)
    if folder:
        enter(browser, folder)
    return browser


def files_of(page_for, admin, person, terminal, folder: str | None = None, **fields) -> Locator:
    """Connect `terminal` for `person` (after `preserve`) and open Files in their new chat."""
    name = connect_for(admin, person, terminal, **fields)
    return files_in(page_for(person), name, folder)


def open_file(browser: Locator, name: str) -> None:
    """Click the file in the list and wait until its own tab is the one on screen."""
    browser.get_by_text(name, exact=True).click()
    expect(tab(browser.page, name)).to_have_attribute("aria-selected", "true")
    rest_elsewhere(browser.page)


def back_to(browser: Locator, folder: str) -> None:
    """From an opened file back to the list of the folder it is in."""
    browser.get_by_role("button", name=folder, exact=True).click()
    expect(browser.get_by_role("list")).to_be_visible()
    rest_elsewhere(browser.page)


def comment_shown_by(page: Page, marker: Locator) -> Locator:
    """Rest on a document's comment marker; returns the comment that opens beside it."""
    marker.hover()
    comment = page.get_by_role("dialog", name="Comment", exact=True)
    expect(comment).to_be_visible()
    return comment
