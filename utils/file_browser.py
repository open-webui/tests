"""The file browser of a chat's terminal: its toolbar, its rows and their menus.

`connected_browser` connects an Open Terminal for a person and opens Files in their chat.
`listed` is what the list shows from top to bottom, `row` one entry, `select` adds an entry to
the selection and `rest_pointer` takes the pointer off the list so that no preview covers it.
`choose_action` and `sort_by` pick from the toolbar's Actions and Sort menus, `row_menu` opens an
entry's More menu and `folder_menu` the menu a right click on the empty part of the list opens.
`crumb` is a folder in the path above the list and `enter` opens a folder made on disk after the
list loaded.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from playwright.sync_api import Locator, Page, expect

from harness.backends import wait_until
from harness.terminal_server import configure_terminals, read_grant
from utils.cached_chat import pick_terminal

LIST_SECONDS = 10
LOAD_FAILED = "Failed to load directory. Check your Terminal connection in Settings → Integrations."


def file_browser(page: Page) -> Locator:
    return page.get_by_role("region", name="File browser")


def connected_browser(page_for, admin, person, terminal, **fields) -> Locator:
    """Connect `terminal` for `person` (after `preserve`) and open Files in their new chat."""
    connection = terminal.connection(config={"access_grants": [read_grant(person.id)]}, **fields)
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    return file_browser(page)


def row(browser: Locator, name: str) -> Locator:
    return browser.locator("[data-file-row]").filter(has=browser.page.get_by_text(name, exact=True))


def rest_pointer(browser: Locator) -> None:
    """Take the pointer off the list and wait until no file's preview is open any more."""
    # the preview of the file the pointer rested on lies over the rows and the space under it
    corner = browser.bounding_box()
    assert corner, "the file browser has no place on the page"
    browser.page.mouse.move(corner["x"] + 2, corner["y"] + 2)
    expect(browser.locator("[data-file-list]").get_by_role("button", expanded=True)).to_have_count(
        0
    )


def select(browser: Locator, name: str, key: str = "Control") -> None:
    """Click the entry with Ctrl held (or `key`), which adds it to the selection."""
    rest_pointer(browser)
    row(browser, name).get_by_text(name, exact=True).click(modifiers=[key])


def listed(browser: Locator) -> list[str]:
    """The names in the list from top to bottom, a folder with a slash after it."""
    # each row carries its entry's path
    paths = browser.locator("[data-file-row]").evaluate_all(
        "rows => rows.map(row => row.getAttribute('data-file-row'))"
    )
    return [PurePosixPath(path).name + ("/" if path.endswith("/") else "") for path in paths]


def expect_listed(browser: Locator, *names: str) -> None:
    """Wait until the list shows exactly `names` in this order, a folder with a slash after it."""
    expected = list(names)
    assert wait_until(lambda: listed(browser) == expected, LIST_SECONDS), (
        f"the list shows {listed(browser)}, not {expected}"
    )


def refresh(browser: Locator) -> None:
    browser.get_by_label("Refresh", exact=True).first.click()


def enter(browser: Locator, folder: str, shows: str) -> None:
    """Open a folder made on disk since the list loaded and wait for one of its entries."""
    refresh(browser)
    browser.get_by_text(folder, exact=True).click()
    expect(browser.get_by_text(shows, exact=True)).to_be_visible()


def open_actions(browser: Locator) -> Locator:
    browser.get_by_label("Actions", exact=True).click()
    menu = browser.page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def choose_action(browser: Locator, action: str) -> None:
    open_actions(browser).get_by_role("menuitem", name=action, exact=True).click()


def sort_by(browser: Locator, order: str) -> None:
    browser.get_by_label("Sort", exact=True).click()
    menu = browser.page.get_by_role("menu")
    menu.get_by_role("menuitem", name=order, exact=True).click()
    expect(menu).to_have_count(0)


def row_menu(browser: Locator, name: str) -> Locator:
    entry = row(browser, name)
    entry.hover()
    entry.get_by_label("More", exact=True).click()
    menu = browser.page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def entries_of(menu: Locator) -> list[str]:
    return [text.strip() for text in menu.get_by_role("menuitem").all_inner_texts()]


def folder_menu(browser: Locator) -> Locator:
    """Right click the empty part of the list under its last row; returns the menu that opens."""
    rest_pointer(browser)
    below = browser.locator("[data-file-row]").last.bounding_box()
    assert below, "the list has no row on the page"
    browser.page.mouse.click(
        below["x"] + below["width"] / 2, below["y"] + below["height"] + 16, button="right"
    )
    # the menu has no role or name of its own: the innermost block holding its first and last entry
    page = browser.page
    menu = (
        browser.locator("div")
        .filter(has=page.get_by_role("button", name="Refresh", exact=True))
        .filter(has=page.get_by_role("button", name="Upload Folder", exact=True))
        .last
    )
    expect(menu).to_be_visible()
    return menu


def crumb(browser: Locator, label: str) -> Locator:
    return browser.get_by_role("button", name=label, exact=True)
