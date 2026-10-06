"""Setting a tool's or function's valves the way a person does, in its Valves dialog or panel."""

from __future__ import annotations

from playwright.sync_api import Locator, expect


def valve(dialog: Locator, title: str, description: str) -> Locator:
    """One valve's row: its title, Default or Custom button, input and description."""
    rows = dialog.locator("div").filter(has=dialog.page.get_by_text(title, exact=True))
    return rows.filter(has=dialog.page.get_by_text(description, exact=True)).last


def customise(row: Locator) -> None:
    row.get_by_role("button", name="Default").click()
    expect(row.get_by_role("button", name="Custom")).to_be_visible()
