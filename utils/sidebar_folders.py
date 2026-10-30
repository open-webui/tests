"""A folder's row in the sidebar, located the way a person sees it.

The row holds the expand toggle (its first button), the folder's name that opens the folder (a
double click renames it, hovering shows a preview of the folder) and, on hover, the "More" menu
button (its last button). `folder_link(scope, name)` is the name to click; `folder_row(scope,
name)` is the whole row, for hovering, dragging and its buttons. `scope` is the sidebar or the
page. The name carries an unread count after the folder's name when there is one, so `name`
matches that start.

Retargeted for b704d0613: the folder's name is the trigger of its hover preview, so it is a
button that opens a dialog, still an anchor to the folder page. Chat rows are buttons as well,
so the name is looked up inside folder rows only.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page

_ROWS = "[id^='folder-'][id$='-button']"


def _named(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)}( \d+\+?)?$")


def folder_link(scope: Locator | Page, name: str) -> Locator:
    return scope.locator(_ROWS).get_by_role("button", name=_named(name))


def folder_row(scope: Locator | Page, name: str) -> Locator:
    page = scope if isinstance(scope, Page) else scope.page
    return scope.locator(_ROWS).filter(has=page.get_by_role("button", name=_named(name)))


def click_folder_menu_button(row: Locator) -> None:
    """Click the row's "More" button, shown on hover, hovering the expand toggle to show it.

    Hovering the name instead opens its preview after a moment, and the preview covers the button.
    """
    row.get_by_role("button").first.hover()
    row.get_by_role("button").last.click()
