"""Opening the Manage Ollama dialog of one connection, the way an admin does."""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

from utils.admin_connections import connection_row
from utils.tooltips import tooltip_button


def open_manage_ollama(page: Page, url: str) -> Locator:
    """The Manage dialog of the Ollama connection saved with `url`, opened from Connections."""
    page.goto("/admin/settings/connections")
    row = connection_row(page.get_by_role("dialog"), url)
    tooltip_button(row, "Manage").click()
    dialog = page.get_by_role("dialog").filter(has_text="Pull a model from Ollama.com")
    expect(dialog).to_be_visible()
    return dialog
