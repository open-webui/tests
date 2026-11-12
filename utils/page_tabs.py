"""Pages that open as page tabs inside the chat view: Workspace, Notes, Automations, Calendar and
the like. Since dev da850248d each shows in a tab panel named by its page tab, whose label is the
page's own title. A page the account may not open keeps its address and shows only a permission
message in its panel.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

NO_PERMISSION = "You do not have permission to access this resource."


def page_tab(page: Page, label: str) -> Locator:
    """The panel of the page tab labelled `label`."""
    return page.get_by_role("tabpanel", name=label, exact=True)


def expect_no_permission(page: Page, path: str) -> None:
    """The page at `path` stays open but shows only the permission message."""
    expect(page).to_have_url(re.compile(rf"{re.escape(path)}$"))
    expect(page.get_by_role("tabpanel").get_by_text(NO_PERMISSION, exact=True)).to_be_visible()
