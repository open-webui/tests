"""The visibility menu at the top of an access dialog: workspace items, chat links, tool servers."""

from __future__ import annotations

from playwright.sync_api import Locator


def visibility(dialog: Locator) -> Locator:
    """The button that shows the current visibility (Private, Public or Open) and opens the menu."""
    # the menu's trigger wrapper takes the button's name too, so pick the labelled button itself
    return dialog.get_by_label("Visibility", exact=True)


def choose_visibility(dialog: Locator, label: str) -> None:
    visibility(dialog).click()
    dialog.page.get_by_role("menuitemradio", name=label, exact=True).click()


def visibility_choices(dialog: Locator) -> Locator:
    """The visibility menu's entries, after opening it."""
    visibility(dialog).click()
    return dialog.page.get_by_role("menuitemradio")
