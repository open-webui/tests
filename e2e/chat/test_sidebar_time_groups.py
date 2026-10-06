"""Journey: the sidebar groups chats under Today, Yesterday, the last days, months and years.

Each chat sits under the heading for when it was last updated, newest first: Today, Yesterday,
Previous 7 days, Previous 30 days, then the month for earlier chats of this year and the year for
older ones. The browser's clock is fixed so the headings do not depend on when the test runs.

One test is red on dev: a chat from the last day of the previous month is not filed under
Yesterday on the first of the month, because Yesterday is only given when both days fall in the
same month, so it lands under Previous 7 days (open-webui/open-webui#31964).

Discriminates: passes on dev 30f3f6a8f apart from the month boundary test (the bug above), which
passes in a frontend build giving Yesterday for the previous calendar day; in a backend copy whose
chat import stamps every chat with the time of the import the grouping test fails.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _import(account, title: str, updated: datetime) -> None:
    stamp = int(updated.timestamp())
    message = {"id": "m1", "parentId": None, "childrenIds": [], "role": "user", "content": title}
    chat = {"title": title, "history": {"currentId": "m1", "messages": {"m1": message}}}
    form = {"chat": chat, "created_at": stamp, "updated_at": stamp}
    with account.client() as client:
        imported = client.post("/api/v1/chats/import", json={"chats": [form]})
    assert imported.status_code == 200, imported.text


def _sidebar_at(page_for, account, now: datetime):
    page: Page = page_for(account, timezone_id="UTC")
    page.clock.set_fixed_time(now)
    page.reload()
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    return sidebar


def _heading_above(sidebar, title: str) -> str:
    """The time heading the chat titled `title` is listed under."""
    row = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    # headings carry no id, the chat rows between them do
    heading = row.locator("xpath=preceding-sibling::div[not(@id='sidebar-chat-group')][1]")
    return heading.inner_text().strip()


def test_each_chat_is_listed_under_the_heading_for_when_it_was_updated(page_for, make_user):
    account = make_user()
    chats = {
        "Harbour lunch": ("2026-10-15T09:00:00", "Today"),
        "Ferry booking": ("2026-10-14T09:00:00", "Yesterday"),
        "Sail repair": ("2026-10-11T09:00:00", "Previous 7 days"),
        "Mooring fees": ("2026-09-25T09:00:00", "Previous 30 days"),
        "Spring regatta": ("2026-03-10T09:00:00", "March"),
        "Old logbook": ("2024-06-01T09:00:00", "2024"),
    }
    for title, (updated, _) in chats.items():
        _import(account, title, _at(updated))
    sidebar = _sidebar_at(page_for, account, _at("2026-10-15T12:00:00"))

    for title, (_, heading) in chats.items():
        expect(sidebar.get_by_role("button", name=title)).to_be_visible()
        assert _heading_above(sidebar, title) == heading, title


def test_the_last_day_of_last_month_is_yesterday_on_the_first(page_for, make_user):
    account = make_user()
    _import(account, "Month end tally", _at("2026-10-31T18:00:00"))
    sidebar = _sidebar_at(page_for, account, _at("2026-11-01T09:00:00"))

    expect(sidebar.get_by_role("button", name="Month end tally")).to_be_visible()
    assert _heading_above(sidebar, "Month end tally") == "Yesterday", (
        "a chat from the evening before is not filed under Yesterday on the first of the month: "
        "Yesterday is only given when both days are in the same month (#31964)"
    )
