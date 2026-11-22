"""The comments popover of a workspace item: opening it and pressing the controls inside it."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect


def comments_panel(page: Page) -> Locator:
    heading = page.get_by_role("heading", name=re.compile(r"^Comments"))
    return page.get_by_role("menu").filter(has=heading)


def open_comments(page: Page, count: int = 0) -> Locator:
    """Press the item's Comments button, named with the number of comments when there are any."""
    name = f"Comments ({count})" if count else "Comments"
    page.get_by_role("button", name=name, exact=True).click()
    panel = comments_panel(page)
    expect(panel).to_be_visible()
    return panel


def panel_control(scope: Locator, name: str) -> Locator:
    """A button in the popover such as Reply, More or Save."""
    # the popover is a menu that marks every button in it as a menu item
    return scope.get_by_role("button", name=name, exact=True).or_(
        scope.get_by_role("menuitem", name=name, exact=True)
    )
