"""The automations list, located the way a person sees it: each automation is a row named by its
title, showing its schedule and holding the Active switch and the Automation Menu. The row's former
`Open automation` label is gone from the markup; the title is a link inside the row now.

The Create and Edit dialog's Terminal dropdown: `wait_until_loaded`, `terminal_button`,
`open_terminal_menu`, `toggle_terminal`, `close_terminal_menu` and `working_directory`.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.workspace_list import item_rows


def automation_row(page: Page, name: str) -> Locator:
    return item_rows(page).filter(has_text=name)


def wait_until_loaded(dialog: Locator, save: str = "Create") -> None:
    """The dialog resets its fields once its terminals, folders and channels have loaded."""
    expect(dialog.get_by_role("button", name=save, exact=True)).to_be_enabled()


def terminal_button(dialog: Locator, shown: str = "Terminal") -> Locator:
    """The dialog's Terminal dropdown, which reads the picked terminal's name once one is picked."""
    return dialog.get_by_role("button", name=shown, exact=True).first


def open_terminal_menu(page: Page, dialog: Locator, shown: str = "Terminal") -> Locator:
    terminal_button(dialog, shown).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Terminal", exact=True)).to_be_visible()
    return menu


def toggle_terminal(page: Page, menu: Locator, name: str) -> None:
    """Click the terminal in the open dropdown, which closes it: picks the terminal, or takes
    it off when it was the picked one."""
    menu.get_by_role("menuitem", name=name).click()
    expect(page.get_by_role("menu")).to_have_count(0)


def close_terminal_menu(page: Page, dialog: Locator, shown: str = "Terminal") -> None:
    terminal_button(dialog, shown).click()
    # a dropdown opened while this one still fades out would be a second menu on the page
    expect(page.get_by_role("menu")).to_have_count(0)


def working_directory(menu: Locator) -> Locator:
    """The Working Directory field of the open Terminal dropdown; it has no label of its own."""
    return menu.get_by_placeholder("/home/user/project")
