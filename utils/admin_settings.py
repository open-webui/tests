"""Saving an Admin Settings page the way an admin does.

Since f2a30994c (#34534) a page with pending edits shows their count ("2 Changes") beside Save, and
Save opens an "Unsaved changes" dialog listing them; only its own Save submits. With nothing
changed, Save submits at once.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

SAVED = "Settings saved successfully!"
PENDING_CHANGES = re.compile(r"^\d+ Changes?$")


def review_dialog(page: Page) -> Locator:
    return page.get_by_role("dialog", name="Unsaved changes")


def submit(settings: Locator) -> None:
    """Click Save on `settings` and confirm the list of changes when the page shows one."""
    has_changes = settings.get_by_role("button", name=PENDING_CHANGES).count() > 0
    settings.get_by_role("button", name="Save", exact=True).click()
    if has_changes:
        dialog = review_dialog(settings.page)
        dialog.get_by_role("button", name="Save", exact=True).click()
        expect(dialog).to_be_hidden()


def save(settings: Locator) -> None:
    """Save `settings` and wait for the page to report it saved."""
    submit(settings)
    expect(settings.page.get_by_text(SAVED).first).to_be_visible()
    # a toast left by an earlier save also matches; the count only goes once this save is stored
    expect(settings.get_by_role("button", name=PENDING_CHANGES)).to_have_count(0)
