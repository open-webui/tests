"""The Git panel under a chat's Files tab.

`expand_git_panel` opens a folder of the file browser that stands in the terminal's start folder
and expands the Git panel that shows once the browser is in a repository.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect


def expand_git_panel(page: Page, folder: str) -> Locator:
    """Open `folder` in the file browser and expand the Git panel under it; returns the panel."""
    files = page.get_by_role("region", name="File browser")
    files.get_by_text(folder, exact=True).click()
    git = page.get_by_role("region", name="Git")
    git.get_by_role("button", name="Expand Git panel").click()
    expect(git.get_by_role("button", name="Collapse Git panel")).to_be_visible()
    return git
