"""Workspace and knowledge lists, located the way a person sees them: each item is a row named by
its title. A row holds the item's link, its details and, on hover, the item's menu and any switch.
The row was a button before; it is the presentation element around the link now. The page around
the list is a presentation element too and holds every link, so a row is the innermost one.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def item_rows(page: Page) -> Locator:
    """Every item row in the list: the innermost presentation elements that hold an item link."""
    # `has` and `has_not` are queried from inside each element, so they must not start from main
    presentation = "div[role='presentation']"
    holding_a_link = page.locator(presentation).filter(has=page.get_by_role("link"))
    rows = page.get_by_role("main").locator(presentation).filter(has=page.get_by_role("link"))
    return rows.filter(has_not=holding_a_link)


def item_row(page: Page, name: str) -> Locator:
    """The row of the item titled `name`."""
    return item_rows(page).filter(has_text=name)
