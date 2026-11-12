"""Pages that open as page tabs inside the chat view: Workspace, Notes, Automations, Calendar and
the like. Since dev da850248d each shows in a tab panel named by its page tab, whose label is the
page's own title.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page


def page_tab(page: Page, label: str) -> Locator:
    """The panel of the page tab labelled `label`."""
    return page.get_by_role("tabpanel", name=label, exact=True)
