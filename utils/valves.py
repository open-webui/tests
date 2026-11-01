"""Setting a tool's or function's valves the way a person does, in its Valves dialog or panel."""

from __future__ import annotations

from playwright.sync_api import Locator, expect

from utils.menus import late_menu_entry


def valve(dialog: Locator, title: str, description: str) -> Locator:
    """One valve's row: its title, Default or Custom button, input and description."""
    rows = dialog.locator("div").filter(has=dialog.page.get_by_text(title, exact=True))
    return rows.filter(has=dialog.page.get_by_text(description, exact=True)).last


def customise(row: Locator) -> None:
    """Switch a valve from Default to Custom; in the chat's Controls its button is a menu item."""
    late_menu_entry(row, "Default").click()
    expect(late_menu_entry(row, "Custom")).to_be_visible()
