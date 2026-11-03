"""The role button in a row of Admin Panel > Users, which shows the role and opens the editor."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, expect


def role_button(row: Locator) -> Locator:
    return row.get_by_role("button", name="Change User Role")


def expect_role(row: Locator, role: str) -> None:
    """The row shows `role` as a word of its own, beside the button's screen-reader label."""
    expect(role_button(row)).to_have_text(re.compile(rf"(^|\s){re.escape(role)}(\s|$)"))
