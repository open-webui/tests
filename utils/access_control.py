"""Access dialogs of workspace items, chat links and tool servers: visibility menu and list."""

from __future__ import annotations

from playwright.sync_api import Locator, expect


def visibility(dialog: Locator) -> Locator:
    """The button that shows the current visibility (Private, Public or Open) and opens the menu."""
    # its name now reads "Visibility" then the current choice, so match the word it starts with
    return dialog.get_by_role("button", name="Visibility")


def expect_visibility(dialog: Locator, label: str) -> None:
    """The visibility button shows `label` as the current choice."""
    expect(visibility(dialog)).to_have_accessible_name(f"Visibility {label}")


def choose_visibility(dialog: Locator, label: str) -> None:
    visibility(dialog).click()
    dialog.page.get_by_role("menuitemradio", name=label, exact=True).click()


def visibility_choices(dialog: Locator) -> Locator:
    """The visibility menu's entries, after opening it."""
    visibility(dialog).click()
    return dialog.page.get_by_role("menuitemradio")


def access_row(dialog: Locator, name: str) -> Locator:
    """The row of the person or group `name` in the access list, with its Access level."""
    page = dialog.page
    # the innermost box holding both the name and a level picker is that principal's row
    return (
        dialog.locator("div")
        .filter(has=page.get_by_text(name))
        .filter(has=page.get_by_role("combobox", name="Access level"))
        .last
    )


def remove_access(dialog: Locator, name: str) -> None:
    # the row's remove button has no label; it is the row's last button
    access_row(dialog, name).get_by_role("button").last.click()
