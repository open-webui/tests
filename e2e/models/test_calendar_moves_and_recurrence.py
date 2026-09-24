"""Regression: an event moved to another day in the calendar editor shows on that day.

open-webui 0.11.2 fix `1c5128c4a` (#29085, issue #29067). The editor has one date field, so
moving an event kept its end on the old date and saved an end before the start; the month view
spread each event from its start day to its end day, so an end before the start put it on no
day at all, and the range query dropped it too. The editor now keeps the event's duration from
the new start, the month view never ends an event before its start day, and the range query
accepts an event that starts inside the range.

Twin of unit/models/test_calendar_moves_and_recurrence.py.

Discriminates: passes on dev bbfa876af; with the frontend half reverted (the editor saves the
stale end, the month view spreads from start to end) the moved event shows on no day of the
month view, and with only the editor reverted the stored end lands before the start.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.calendar_api import HOUR_NS, create_event, default_calendar_id, to_ns

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _this_month_at_ten(day: int) -> dt.datetime:
    today = dt.date.today()
    return dt.datetime(today.year, today.month, day, 10, 0).astimezone()


def _day_cell(page: Page, title: str):
    """The month-view day that holds the event's chip."""
    return page.get_by_role("button").filter(has=page.get_by_role("button", name=title))


def test_an_event_moved_a_week_later_in_the_editor_shows_on_its_new_day(page_for, make_user):
    owner = make_user()
    title = f"Planning {uuid.uuid4().hex[:6]}"
    original_start = _this_month_at_ten(1)
    with owner.client() as client:
        created = create_event(
            client,
            default_calendar_id(client),
            title=title,
            start_at=to_ns(original_start),
            end_at=to_ns(original_start) + HOUR_NS,
        )
    assert created.status_code == 200, created.text

    page = page_for(owner)
    page.goto("/calendar")
    expect(_day_cell(page, title).get_by_text("1", exact=True)).to_be_visible()
    page.get_by_role("button", name=title).last.click()

    moved_start = _this_month_at_ten(8)
    page.locator('input[type="date"]').fill(moved_start.date().isoformat())
    page.get_by_role("button", name="Save").click()

    expect(_day_cell(page, title).get_by_text("8", exact=True)).to_be_visible()
    with owner.client() as client:
        stored = client.get(f"/api/v1/calendars/events/{created.json()['id']}").json()
    assert (stored["start_at"], stored["end_at"]) == (
        to_ns(moved_start),
        to_ns(moved_start) + HOUR_NS,
    ), "the editor kept the old end date, so the event now ends before it starts (#29067)"
