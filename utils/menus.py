"""Entries of a dropdown menu that the menu draws after it opened.

Entries drawn after a menu opened (search results, loaded lists, tag chips) are menu items since
05e115366 (open-webui/open-webui#33374) and plain buttons on builds before it, so they are found
under either role.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def late_menu_entry(scope: Locator | Page, name: str, exact: bool = False) -> Locator:
    return scope.get_by_role("menuitem", name=name, exact=exact).or_(
        scope.get_by_role("button", name=name, exact=exact)
    )
