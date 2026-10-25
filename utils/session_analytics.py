"""The admin analytics Session Explorer and Session Detail, read the way an admin does.

The explorer is the Sessions view of the dashboard on `/admin/analytics`: a toolbar (search,
model filter), a summary of signals and a sortable table, each row a chat. Clicking a row opens
the Session Detail in a modal: the session's stats, a token or timeline visualization and the
saved execution trace (responses, tool calls and errors), filterable to tools or errors.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, expect


def sessions_tab(dialog: Locator, selected: bool | None = None) -> Locator:
    """The Sessions / Overview switch, shown when admin chat access is on."""
    return dialog.get_by_role("button", name="Sessions", exact=True, pressed=selected)


def explorer(dialog: Locator) -> Locator:
    """The Session Explorer region, once the Sessions view is open."""
    return dialog.get_by_role("region", name="Sessions", exact=True)


def search_box(dialog: Locator) -> Locator:
    return explorer(dialog).get_by_role("searchbox", name="Search sessions")


def model_filter(dialog: Locator) -> Locator:
    return explorer(dialog).get_by_role("combobox", name="Model")


def sort_header(dialog: Locator, label: str) -> Locator:
    """The table header button that sorts by its column."""
    return explorer(dialog).get_by_role("button", name=label, exact=True)


def session_rows(dialog: Locator) -> Locator:
    """The body rows, each a chat; the header row carries column headers, not cells."""
    return explorer(dialog).get_by_role("row").filter(has=dialog.page.get_by_role("cell"))


def session_row(dialog: Locator, title: str) -> Locator:
    return session_rows(dialog).filter(has_text=title)


# The explorer's columns after the title cell, in table order.
COLUMNS = {"user": 1, "last_active": 2, "responses": 3, "tokens": 4, "errors": 5, "span": 6}


def session_cell(row: Locator, column: str) -> Locator:
    return row.get_by_role("cell").nth(COLUMNS[column])


def signal(dialog: Locator, label: str) -> Locator:
    """A value beside its label in the explorer's session summary."""
    summary = dialog.get_by_role("region", name="Session summary")
    term = dialog.page.get_by_role("term").filter(has_text=re.compile(f"^{re.escape(label)}$"))
    return summary.locator("dl > div").filter(has=term).get_by_role("definition")


def open_session(dialog: Locator, title: str) -> Locator:
    """Click the session's row and return its detail modal."""
    session_row(dialog, title).get_by_role("button").first.click()
    detail = dialog.page.get_by_role("dialog", name="Session details")
    expect(detail).to_be_visible()
    return detail


def detail_stat(detail: Locator, label: str) -> Locator:
    """A value beside its label in the detail's stats row (Prompts, Responses, Tokens)."""
    term = detail.page.get_by_role("term").filter(has_text=re.compile(f"^{re.escape(label)}$"))
    return detail.locator("dl > div").filter(has=term).get_by_role("definition")


def visualization(detail: Locator, name: str) -> Locator:
    """The Tokens / Timeline switch above the visualization."""
    return detail.get_by_role("group", name="Session visualization").get_by_role(
        "button", name=name, exact=True
    )


def trace_filter(detail: Locator, name: str) -> Locator:
    """The All / Tools / Errors filter above the trace; the names are exact, not the row labels."""
    return detail.get_by_role("button", name=name, exact=True)
