"""The Notes page, located the way a person sees it: the header's menus and the list of notes.

Since dev db884d2c3 the header holds three icon buttons, each opening a menu. New Note offers
New Note and Import txt/md; Filter offers All, Created by you and Shared with you under Show,
and Write and Read Only under Access; Display options offers List and Grid under Layout. The
choices of Filter and Display options are radio items whose current one reads checked; the
buttons no longer name it. `create_note` makes a note from New Note and returns its id,
`new_note_menu` opens that menu, `choose(page, menu, option)` picks a choice in Filter or Display
options, `expect_chosen` checks the current one and `open_header_menu` opens either menu.

A row holds the note's title link, its relative time and, on hover, the Note Menu and Delete
buttons. `note_row(page, title)` is one note's row, for clicking it open or reaching its menu;
`note_rows(page)` is every row in the list's order. The former row carried the label `Open note`,
now gone from the markup.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from playwright.sync_api import Locator, Page, expect

from utils.workspace_list import item_rows

NOTE_URL = re.compile(r"/notes/[0-9a-f-]+$")


def note_rows(page: Page) -> Locator:
    return item_rows(page)


def note_row(page: Page, title: str) -> Locator:
    return note_rows(page).filter(has_text=title)


def new_note_menu(
    page: Page, header: Locator | None = None, press: Callable[[Locator], None] = Locator.click
) -> Locator:
    """Open the header's New Note menu. A phone passes the drawer as `header` and taps."""
    header = header or page.get_by_role("main")
    press(header.get_by_role("button", name="New Note", exact=True))
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Import txt/md")).to_be_visible()
    return menu


def create_note(
    page: Page, header: Locator | None = None, press: Callable[[Locator], None] = Locator.click
) -> str:
    """Create a note from the header's New Note menu and wait for it to open; its id."""
    menu = new_note_menu(page, header, press)
    press(menu.get_by_role("menuitem", name="New Note", exact=True))
    expect(page).to_have_url(NOTE_URL)
    return page.url.rsplit("/", 1)[1]


def open_header_menu(page: Page, menu: str) -> Locator:
    page.get_by_role("main").get_by_role("button", name=menu, exact=True).click()
    opened = page.get_by_role("menu")
    expect(opened.get_by_role("menuitemradio").first).to_be_visible()
    return opened


def _close_menu(page: Page, menu: str) -> None:
    # by mouse: Escape leaves keyboard focus on the button, which keeps Shift from deleting
    page.get_by_role("main").get_by_role("button", name=menu, exact=True).click()
    expect(page.get_by_role("menu")).to_have_count(0)


def choose(page: Page, menu: str, option: str) -> None:
    """Pick `option` in the header's `menu` (Filter or Display options) and close the menu."""
    item = open_header_menu(page, menu).get_by_role("menuitemradio", name=option, exact=True)
    item.click()
    expect(item).to_be_checked()
    _close_menu(page, menu)


def expect_chosen(page: Page, menu: str, option: str) -> None:
    """The header's `menu` reads `option` as its current choice."""
    item = open_header_menu(page, menu).get_by_role("menuitemradio", name=option, exact=True)
    expect(item).to_be_checked()
    _close_menu(page, menu)
