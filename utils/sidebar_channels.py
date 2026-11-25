"""The sidebar's Channels section, opened the way a person opens it.

`channels_section(page)` opens the sidebar, expands Channels when it is folded and returns the
sidebar once the section has finished sliding open, so a click on a channel lands on that row.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def channels_section(page: Page) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels", exact=True)
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    # the section slides open, and a click while it moves lands on the row that slid under it
    sidebar.locator("#sidebar-channels-content").evaluate(
        "content => Promise.allSettled("
        "content.parentElement.getAnimations().map((animation) => animation.finished))"
    )
    return sidebar
