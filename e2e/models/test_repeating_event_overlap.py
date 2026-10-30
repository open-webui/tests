"""Regression: a repeating event running into the first day of the month view was not shown there.

Fix `ad01bca2b` (#31606, issue #31605): the calendar page loads the events of the days it shows,
and the server expanded a repeating event only into occurrences starting inside them. An
occurrence that began the evening before the first day of the month view and ran past midnight
was left out, so that first day showed nothing, while the same event without a repeat was
shown. The server now keeps every occurrence that ends after the range starts.

The browser runs in UTC, as does the account, so the page's range and the server's expansion
agree on midnight.

Twin of integration/models/test_repeating_event_overlap.py.

Discriminates: passes on dev a5bc78300 with its build; with ad01bca2b reverted in the backend the
month view's first day lacks the occurrence that began the Saturday before it.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.calendar_api import HOUR_NS, create_event, default_calendar_id, set_timezone, to_ns
from utils.calendar_month import days_showing

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

IN_UTC = {"timezone_id": "UTC", "locale": "en-US"}


def _month_grid(today: dt.date) -> list[dt.date]:
    """The 42 days the month view shows, from the Sunday on or before the first."""
    first = today.replace(day=1)
    grid_start = first - dt.timedelta(days=(first.weekday() + 1) % 7)
    return [grid_start + dt.timedelta(days=offset) for offset in range(42)]


def _chip(page: Page, title: str) -> Locator:
    """The event's own button, not the day cell around it that shares its name."""
    return page.get_by_role("button", name=title).filter(has_not=page.get_by_role("button"))


def _days_showing(page: Page, title: str) -> list[int]:
    """The day numbers of the month-view squares holding the event, in grid order."""
    return days_showing(page, _chip(page, title))


def test_a_weekly_overnight_event_shows_on_the_first_day_of_the_month_view(page_for, make_user):
    owner = make_user()
    title = f"Night shift {uuid.uuid4().hex[:6]}"
    grid = _month_grid(dt.datetime.now(dt.timezone.utc).date())
    saturday_before_grid = grid[0] - dt.timedelta(days=1)
    series_start = dt.datetime.combine(
        saturday_before_grid - dt.timedelta(weeks=1), dt.time(23), dt.timezone.utc
    )
    with owner.client() as client:
        set_timezone(client, "UTC")
        created = create_event(
            client,
            default_calendar_id(client),
            title=title,
            start_at=to_ns(series_start),
            end_at=to_ns(series_start) + 2 * HOUR_NS,
            rrule="RRULE:FREQ=WEEKLY",
        )
    assert created.status_code == 200, created.text

    page = page_for(owner, **IN_UTC)
    with page.expect_response(lambda response: "/calendars/events?" in response.url):
        page.goto("/calendar")
    expect(_chip(page, title).first).to_be_visible()

    weekend_days = [day.day for day in grid if day.weekday() in (5, 6)]
    shown_on = _days_showing(page, title)
    assert shown_on == weekend_days, (
        f"the 23:00 to 01:00 weekly event shows on days {shown_on}; the occurrence that began the "
        f"evening before the first day {grid[0]} is missing from it (#31605)"
    )
