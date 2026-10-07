"""Journey: a person gets reminded of their events and sees them in their own time zone.

* An event made in the editor with a Reminder pops up a toast with its title and how many minutes
  are left once the reminder is due; clicking the toast opens the calendar. An event whose
  Reminder is None raises nothing, even when it is due in the same poll. The editor shows the
  chosen Reminder again when the event is reopened.
* An event made in the editor by someone in one zone shows on a calendar shared with someone in
  another zone at the reader's local time and on the reader's local day.
* Turning a repeating event back to No Repeat leaves only its first day; deleting a repeating event
  from one of its occurrences removes the whole series.
* An event put on a calendar made in the sidebar, picked in the editor's Calendar field, wears
  that calendar's colour; the week view puts an event in the column of the day it starts.

Discriminates: in a backend copy that dropped the reminder choice on save the reopened editor went
back to 10 minutes before and the None event raised a toast; with the reminder alert never sent
the toast test went red; with the update ignoring a removed repeat the event kept repeating, and
with delete leaving the event in place the series test went red. The zone, colour and week-view
tests watch the frontend only.
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.backends import wait_until
from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    create_event,
    default_calendar_id,
    events_between,
    to_ns,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOAST_TIMEOUT_MS = 40_000
# zones whose clock is nowhere near midnight for at least some hours of every day
ZONES = ["Asia/Tokyo", "Europe/Berlin", "America/New_York", "Pacific/Auckland", "America/Denver"]


def _title(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def _midday_zone() -> ZoneInfo:
    """A zone where it is now between 06:00 and 18:00, so the run is far from local midnight."""
    now = dt.datetime.now(dt.timezone.utc)
    for name in ZONES:
        if 6 <= now.astimezone(ZoneInfo(name)).hour < 18:
            return ZoneInfo(name)
    raise AssertionError("no candidate zone has daytime hours")


def _month_grid(today: dt.date) -> list[dt.date]:
    """The 42 days the month view shows, from the Sunday on or before the first."""
    first = today.replace(day=1)
    grid_start = first - dt.timedelta(days=(first.weekday() + 1) % 7)
    return [grid_start + dt.timedelta(days=offset) for offset in range(42)]


def _chip(page: Page, title: str) -> Locator:
    """The event's own button, not the day or hour cell around it that shares its name."""
    return page.get_by_role("button", name=title).filter(has_not=page.get_by_role("button"))


def _days_showing(page: Page, title: str) -> list[int]:
    """The day numbers of the month-view cells holding the event, in grid order."""
    cells = page.get_by_role("button").filter(has=_chip(page, title))
    return [int(text.split()[0]) for text in cells.all_inner_texts()]


def _open_editor(page: Page) -> Locator:
    page.goto("/calendar")
    page.get_by_role("button", name="Create", exact=True).click()
    return page.get_by_role("dialog")


def _select(editor: Locator, option: str) -> Locator:
    """The editor's drop-down that offers that option."""
    return editor.locator("select").filter(has=editor.page.get_by_role("option", name=option))


def _fill_when(editor: Locator, day: dt.date, start: str, end: str) -> None:
    editor.locator('input[type="date"]').fill(day.isoformat())
    editor.locator('input[type="time"]').first.fill(start)
    editor.locator('input[type="time"]').last.fill(end)


def _save(editor: Locator, button: str) -> None:
    with editor.page.expect_response(lambda response: "/calendars/events/" in response.url):
        editor.get_by_role("button", name=button, exact=True).click()
    expect(editor).to_be_hidden()


def _make_in_editor(
    page: Page, title: str, day: dt.date, start: str, end: str, **choices: str
) -> None:
    editor = _open_editor(page)
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, day, start, end)
    for label in choices.values():
        _select(editor, label).select_option(label=label)
    _save(editor, "Create")


def _event(owner: Actor, title: str, start: dt.datetime, calendar_id: str | None = None, **fields):
    with owner.client() as client:
        created = create_event(
            client,
            calendar_id or default_calendar_id(client),
            title=title,
            start_at=to_ns(start),
            end_at=to_ns(start) + HOUR_NS,
            **fields,
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _default_calendar(owner: Actor) -> str:
    with owner.client() as client:
        return default_calendar_id(client)


def _stored(owner: Actor, event_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/calendars/events/{event_id}").json()


def _by_title(owner: Actor, title: str, around: dt.datetime) -> dict:
    start = to_ns(around) - 40 * DAY_NS
    with owner.client() as client:
        listed = events_between(client, start, start + 80 * DAY_NS)
    matching = [event for event in listed if event["title"] == title]
    assert matching, f"no event titled {title!r} was saved"
    return matching[0]


# ---------------------------------------------------------------- reminders


def test_a_reminder_raises_a_toast_that_opens_the_calendar_and_none_stays_quiet(
    page_for, make_user
):
    owner = make_user()
    zone = _midday_zone()
    context = {"timezone_id": zone.key, "locale": "en-US"}
    reminded, quiet = _title("Board meeting"), _title("Dentist")
    # the toast arrives on the account's other open page, which is not on the calendar
    listener = page_for(owner, **context)
    listener.goto("/")
    expect(listener.locator("#chat-input")).to_be_visible()
    editor_page = page_for(owner, **context)

    start = dt.datetime.now(zone).replace(second=0, microsecond=0) + dt.timedelta(minutes=8)
    end = start + dt.timedelta(hours=1)
    for title, reminder in ((quiet, "None"), (reminded, "10 minutes before")):
        _make_in_editor(
            editor_page,
            title,
            start.date(),
            start.strftime("%H:%M"),
            end.strftime("%H:%M"),
            reminder=reminder,
        )

    toast = listener.get_by_text(re.compile(r"Starting in \d+ minutes?"))
    expect(listener.get_by_text(reminded)).to_be_visible(timeout=TOAST_TIMEOUT_MS)
    saved = _by_title(owner, reminded, start)
    assert wait_until(lambda: _stored(owner, saved["id"])["meta"].get("alerted_at"), 20)
    # the quiet event is due in the same poll, so its toast would be here by now
    expect(listener.get_by_text(quiet)).to_have_count(0)
    assert not _stored(owner, _by_title(owner, quiet, start)["id"])["meta"].get("alerted_at")
    expect(toast).to_have_text(re.compile(r"Starting in [5-8] minutes"))

    listener.get_by_text(reminded).click()
    expect(listener).to_have_url(re.compile(r"/calendar$"))


@pytest.mark.parametrize("reminder", ["None", "At time of event", "30 minutes before"])
def test_the_editor_shows_the_chosen_reminder_when_the_event_is_reopened(
    reminder, page_for, make_user
):
    owner = make_user()
    title = _title("Visit")
    day = dt.date.today().replace(day=20)
    page = page_for(owner)

    _make_in_editor(page, title, day, "10:00", "11:00", reminder=reminder)
    page.reload()
    _chip(page, title).click()

    editor = page.get_by_role("dialog")
    expect(_select(editor, "At time of event").locator("option:checked")).to_have_text(reminder)


# ---------------------------------------------------------------- zones


def test_a_shared_event_shows_to_the_reader_at_the_readers_time_and_day(page_for, make_user):
    tokyo, los_angeles = ZoneInfo("Asia/Tokyo"), ZoneInfo("America/Los_Angeles")
    owner, reader = make_user(), make_user()
    title = _title("Planning call")
    day = dt.datetime.now(tokyo).date().replace(day=15)
    owner_start = dt.datetime.combine(day, dt.time(8), tokyo)
    reader_start = owner_start.astimezone(los_angeles)
    assert reader_start.date() == day - dt.timedelta(days=1)
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        shared = client.post(
            "/api/v1/calendars/create", json={"name": "Remote team", "access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text

    owner_page = page_for(owner, timezone_id="Asia/Tokyo", locale="en-US")
    editor = _open_editor(owner_page)
    editor.get_by_placeholder("Event title").fill(title)
    _select(editor, "Remote team").select_option(label="Remote team")
    _fill_when(editor, day, "08:00", "09:00")
    _save(editor, "Create")
    expect(_chip(owner_page, title)).to_have_text(re.compile(rf"^\s*8:00\W*AM\s*{title}\s*$"))
    assert _days_showing(owner_page, title) == [day.day]

    reader_page = page_for(reader, timezone_id="America/Los_Angeles", locale="en-US")
    reader_page.goto("/calendar")
    expect(_chip(reader_page, title)).to_have_text(
        re.compile(rf"^\s*{reader_start.hour - 12}:00\W*PM\s*{title}\s*$")
    )
    assert _days_showing(reader_page, title) == [reader_start.day]


# ---------------------------------------------------------------- repeating events


def test_turning_a_daily_event_back_to_no_repeat_leaves_only_its_first_day(page_for, make_user):
    owner = make_user()
    title = _title("Walk")
    first_day = _month_grid(dt.date.today())[10]
    page = page_for(owner)

    _make_in_editor(page, title, first_day, "07:30", "08:00", repeat="Daily")
    expect(_chip(page, title).first).to_be_visible()
    assert len(_days_showing(page, title)) > 1

    _chip(page, title).nth(2).click()
    editor = page.get_by_role("dialog")
    expect(_select(editor, "Weekly").locator("option:checked")).to_have_text("Daily")
    _select(editor, "Weekly").select_option(label="No Repeat")
    _save(editor, "Save")

    expect(_chip(page, title)).to_have_count(1)
    assert _days_showing(page, title) == [first_day.day]


def test_deleting_a_repeating_event_from_one_occurrence_removes_every_occurrence(
    page_for, make_user
):
    owner = make_user()
    title = _title("Choir")
    start = dt.datetime.combine(_month_grid(dt.date.today())[3], dt.time(18)).astimezone()
    event_id = _event(owner, title, start, rrule="FREQ=WEEKLY")
    page = page_for(owner)
    page.goto("/calendar")
    expect(_chip(page, title)).to_have_count(6)

    _chip(page, title).nth(3).click()
    page.get_by_role("dialog").get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Confirm", exact=True).click()

    expect(_chip(page, title)).to_have_count(0)
    with owner.client() as client:
        assert client.get(f"/api/v1/calendars/events/{event_id}").status_code == 404


# ---------------------------------------------------------------- calendars and views


def test_an_event_put_on_a_new_calendar_in_the_editor_wears_its_colour(page_for, make_user):
    owner = make_user()
    name, title = _title("Garden"), _title("Planting")
    page = page_for(owner)
    page.goto("/calendar")

    page.get_by_title("New calendar").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Calendar name").fill(name)
    dialog.get_by_role("button", name="#ef4444").click()
    dialog.get_by_role("button", name="Create", exact=True).click()
    expect(page.get_by_role("button", name=name)).to_be_visible()

    _make_in_editor(page, title, dt.date.today().replace(day=18), "09:00", "10:00", calendar=name)

    expect(_chip(page, title)).to_be_visible()
    expect(_chip(page, title).locator("span").first).to_have_css(
        "background-color", "rgb(239, 68, 68)"
    )
    assert _by_title(owner, title, dt.datetime.now())["calendar_id"] != _default_calendar(owner)


def test_the_week_view_puts_an_event_in_the_column_of_the_day_it_starts(page_for, make_user):
    owner = make_user()
    late, early = _title("Late film"), _title("Early run")
    today = dt.date.today()
    sunday = today - dt.timedelta(days=(today.weekday() + 1) % 7)
    tuesday = sunday + dt.timedelta(days=2)
    _event(owner, late, dt.datetime.combine(tuesday, dt.time(23, 30)).astimezone())
    _event(
        owner,
        early,
        dt.datetime.combine(tuesday + dt.timedelta(days=1), dt.time(0, 30)).astimezone(),
    )
    page = page_for(owner)
    page.goto("/calendar")
    expect(page.get_by_role("button", name="Month", exact=True)).to_be_visible()
    page.get_by_role("button", name="Month", exact=True).click()
    page.get_by_role("button", name="Week", exact=True).click()
    expect(page.get_by_text("1 AM", exact=True)).to_be_visible()

    expect(_chip(page, late)).to_be_visible()
    expect(_chip(page, early)).to_be_visible()
    late_box, early_box = _chip(page, late).bounding_box(), _chip(page, early).bounding_box()
    tuesday_box = page.get_by_text("Tue", exact=True).bounding_box()
    wednesday_box = page.get_by_text("Wed", exact=True).bounding_box()
    assert late_box and early_box and tuesday_box and wednesday_box
    assert tuesday_box["x"] <= late_box["x"] < wednesday_box["x"]
    assert early_box["x"] >= wednesday_box["x"] - tuesday_box["width"]
    assert late_box["y"] > early_box["y"]
