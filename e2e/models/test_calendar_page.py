"""Journey: a person keeps their calendar on the calendar page.

* The header's Create opens the event editor; the title, day, times, location and description
  are saved as typed, in the browser's local time. An all-day event runs from midnight to the end
  of the day, and clicking an empty day starts an event at nine that day.
* Opening an event lets its owner change it or delete it, and the page shows the result.
* The month, week and day views each show the events of the days they cover, and the arrows
  step the day view to the next day.
* A repeating event made in the editor shows on every day its rule names: a weekly one every
  seventh day, a weekday one Monday to Friday in the browser's own zone, which the page hands to
  the server so the occurrences are expanded there.
* Hiding a calendar in the sidebar hides its events; an event the model made with its calendar
  tool shows at the local time it was given; another account sees an event only once its
  calendar is shared with it.
* Saving a later occurrence of a repeating event moved the whole series to that day, issue
  #30970, fixed by PR #30971.
* An event created from the header after 23:00 is saved ending a day late, issue
  open-webui/open-webui#31994. The editor starts from now until an hour later, which ends on the
  next day, and since PR #31303 (6fdbe3ab6) the end keeps that one day offset from whatever day
  is typed, though the editor shows no end date. The other editor tests open it at noon, so they
  do not depend on when they run.

Discriminates: in a frontend build with the editor dropping the location on create, the all-day
start taken from the time field, the edit saving the old title, the delete never sent, a clicked
day starting at eight, the week view placing events by hour alone, "Weekly" saved as every other
week, the page no longer sending the browser's zone and the sidebar toggle ignored, the matching
test went red and the rest stayed green; in a backend copy with the calendar tool reading times
as UTC the model's event test went red, and with calendars listed to their owners only the
sharing test went red at the reader. The occurrence test fails on dev 176d31d1d, before PR #30971.
The late evening test is red on dev ebc6add67 (open-webui/open-webui#31994).
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    create_event,
    default_calendar_id,
    events_between,
    set_timezone,
    to_ns,
)
from harness.tool_calls import run_tool

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOKYO = ZoneInfo("Asia/Tokyo")
IN_TOKYO = {"timezone_id": "Asia/Tokyo", "locale": "en-US"}


def _title(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def _this_month(day: int, hour: int, minute: int = 0) -> dt.datetime:
    today = dt.date.today()
    return dt.datetime(today.year, today.month, day, hour, minute).astimezone()


def _month_grid(today: dt.date) -> list[dt.date]:
    """The 42 days the month view shows, from the Sunday on or before the first."""
    first = today.replace(day=1)
    grid_start = first - dt.timedelta(days=(first.weekday() + 1) % 7)
    return [grid_start + dt.timedelta(days=offset) for offset in range(42)]


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


def _saved(owner: Actor, title: str, today: dt.date | None = None) -> dict:
    """The owner's event of that title in this month's view, as the page loads it."""
    grid = _month_grid(today or dt.date.today())
    start = to_ns(dt.datetime.combine(grid[0], dt.time()).astimezone())
    with owner.client() as client:
        listed = events_between(client, start, start + 43 * DAY_NS)
    matching = [event for event in listed if event["title"] == title]
    assert matching, f"no event titled {title!r} was saved"
    return matching[0]


def _chip(page: Page, title: str) -> Locator:
    """The event's own button, not the day or hour cell around it that shares its name."""
    return page.get_by_role("button", name=title).filter(has_not=page.get_by_role("button"))


def _days_showing(page: Page, title: str) -> list[int]:
    """The day numbers of the month-view cells holding the event, in grid order."""
    cells = page.get_by_role("button").filter(has=_chip(page, title))
    return [int(text.split()[0]) for text in cells.all_inner_texts()]


def _open_editor(page: Page, at: dt.time = dt.time(12)) -> Locator:
    """The header's editor, opened at `at` today: its new event starts then."""
    page.clock.set_fixed_time(dt.datetime.combine(dt.date.today(), at))
    page.goto("/calendar")
    page.get_by_role("button", name="Create", exact=True).click()
    return page.get_by_role("dialog")


def _fill_when(editor: Locator, day: dt.date, start: str, end: str) -> None:
    editor.locator('input[type="date"]').fill(day.isoformat())
    editor.locator('input[type="time"]').first.fill(start)
    editor.locator('input[type="time"]').last.fill(end)


def _repeat(editor: Locator, label: str) -> None:
    repeat = editor.locator("select").filter(has=editor.page.get_by_role("option", name="Weekly"))
    repeat.select_option(label=label)


def _save(editor: Locator, button: str) -> None:
    with editor.page.expect_response(lambda response: "/calendars/events/" in response.url):
        editor.get_by_role("button", name=button, exact=True).click()
    expect(editor).to_be_hidden()


def _switch_view(page: Page, current: str, target: str) -> None:
    page.get_by_role("button", name=current, exact=True).click()
    page.get_by_role("button", name=target, exact=True).click()
    # the menu closes, leaving only the switch that now names the view
    expect(page.get_by_role("button", name=target, exact=True)).to_have_count(1)


# ---------------------------------------------------------------- create, edit, delete


def test_an_event_created_from_the_header_is_saved_as_typed(page_for, make_user):
    owner = make_user()
    title = _title("Dentist")
    page = page_for(owner)

    editor = _open_editor(page)
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, _this_month(21, 0).date(), "10:30", "11:45")
    editor.get_by_placeholder("Add location").fill("Main St 4")
    editor.get_by_placeholder("Add description").fill("bring the x-rays")
    _save(editor, "Create")

    expect(_chip(page, title)).to_be_visible()
    assert _days_showing(page, title) == [21]
    saved = _saved(owner, title)
    assert (saved["start_at"], saved["end_at"], saved["all_day"]) == (
        to_ns(_this_month(21, 10, 30)),
        to_ns(_this_month(21, 11, 45)),
        False,
    )
    assert (saved["location"], saved["description"]) == ("Main St 4", "bring the x-rays")


def test_an_all_day_event_runs_from_midnight_to_the_end_of_the_day(page_for, make_user):
    owner = make_user()
    title = _title("Holiday")
    page = page_for(owner)

    editor = _open_editor(page)
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, _this_month(17, 0).date(), "10:00", "11:00")
    editor.get_by_label("All day").check()
    expect(editor.locator('input[type="time"]')).to_have_count(0)
    _save(editor, "Create")

    expect(_chip(page, title)).to_have_text(title)
    saved = _saved(owner, title)
    assert (saved["start_at"], saved["end_at"], saved["all_day"]) == (
        to_ns(_this_month(17, 0)),
        to_ns(_this_month(17, 23, 59)),
        True,
    )


def test_an_event_created_late_in_the_evening_ends_on_its_own_day(page_for, make_user):
    owner = make_user()
    title = _title("Breakfast")
    page = page_for(owner)

    editor = _open_editor(page, at=dt.time(23, 30))
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, _this_month(21, 0).date(), "10:30", "11:45")
    _save(editor, "Create")

    expect(_chip(page, title).first).to_be_visible()
    saved = _saved(owner, title)
    assert saved["end_at"] == to_ns(_this_month(21, 11, 45)), (
        "an event created after 23:00 was saved ending a day late, the day after the one typed: "
        "the editor's default end on the next day keeps its offset (open-webui/open-webui#31994)"
    )
    assert _days_showing(page, title) == [21]


def test_clicking_an_empty_day_starts_an_event_at_nine_that_day(page_for, make_user):
    owner = make_user()
    title = _title("Call")
    page = page_for(owner)
    page.goto("/calendar")

    # the sidebar's mini calendar comes first and has a "20" too
    page.get_by_role("button", name="20", exact=True).last.click()
    editor = page.get_by_role("dialog")
    editor.get_by_placeholder("Event title").fill(title)
    _save(editor, "Create")

    saved = _saved(owner, title)
    assert (saved["start_at"], saved["end_at"]) == (
        to_ns(_this_month(20, 9)),
        to_ns(_this_month(20, 10)),
    )


def test_an_edited_event_shows_its_new_title_and_keeps_its_time(page_for, make_user):
    owner = make_user()
    title, renamed = _title("Standup"), _title("Retro")
    event_id = _event(owner, title, _this_month(14, 9), location="Room 1")
    page = page_for(owner)
    page.goto("/calendar")

    _chip(page, title).click()
    editor = page.get_by_role("dialog")
    expect(editor.get_by_placeholder("Add location")).to_have_value("Room 1")
    editor.get_by_placeholder("Event title").fill(renamed)
    editor.get_by_placeholder("Add location").fill("Room 2")
    editor.get_by_placeholder("Add description").fill("bring notes")
    _save(editor, "Save")

    expect(_chip(page, renamed)).to_be_visible()
    expect(_chip(page, title)).to_have_count(0)
    with owner.client() as client:
        stored = client.get(f"/api/v1/calendars/events/{event_id}").json()
    assert (stored["title"], stored["location"], stored["description"]) == (
        renamed,
        "Room 2",
        "bring notes",
    )
    assert stored["start_at"] == to_ns(_this_month(14, 9))


def test_a_deleted_event_leaves_the_calendar(page_for, make_user):
    owner = make_user()
    title = _title("Lunch")
    event_id = _event(owner, title, _this_month(11, 12))
    page = page_for(owner)
    page.goto("/calendar")

    _chip(page, title).click()
    page.get_by_role("dialog").get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Confirm", exact=True).click()

    expect(_chip(page, title)).to_have_count(0)
    with owner.client() as client:
        assert client.get(f"/api/v1/calendars/events/{event_id}").status_code == 404


# ---------------------------------------------------------------- views


def test_the_week_and_day_views_show_only_their_own_days(page_for, make_user):
    owner = make_user()
    today_title, tomorrow_title = _title("Today"), _title("Tomorrow")
    today = dt.datetime.combine(dt.date.today(), dt.time(14)).astimezone()
    tomorrow = today + dt.timedelta(days=1)
    _event(owner, today_title, today)
    _event(owner, tomorrow_title, tomorrow)
    page = page_for(owner)
    page.goto("/calendar")
    expect(_chip(page, today_title)).to_be_visible()

    _switch_view(page, "Month", "Week")
    expect(page.get_by_text("1 AM", exact=True)).to_be_visible()
    expect(_chip(page, today_title)).to_have_count(1)

    _switch_view(page, "Week", "Day")
    expect(page.get_by_text(today.strftime("%a, %B %-d, %Y"), exact=True)).to_be_visible()
    expect(_chip(page, today_title)).to_be_visible()
    expect(_chip(page, tomorrow_title)).to_have_count(0)

    page.get_by_role("button", name="Next", exact=True).click()
    expect(page.get_by_text(tomorrow.strftime("%a, %B %-d, %Y"), exact=True)).to_be_visible()
    expect(_chip(page, tomorrow_title)).to_be_visible()
    expect(_chip(page, today_title)).to_have_count(0)


# ---------------------------------------------------------------- repeating events


def test_a_weekly_event_from_the_editor_shows_every_seventh_day(page_for, make_user):
    owner = make_user()
    title = _title("Choir")
    grid = _month_grid(dt.date.today())
    first_day = grid[3]
    page = page_for(owner)

    editor = _open_editor(page)
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, first_day, "18:00", "19:30")
    _repeat(editor, "Weekly")
    _save(editor, "Create")

    expect(_chip(page, title).first).to_be_visible()
    expect(_chip(page, title)).to_have_count(6)
    assert _days_showing(page, title) == [grid[3 + 7 * week].day for week in range(6)]


def test_a_weekday_event_stays_on_weekdays_in_a_browser_far_from_utc(page_for, make_user):
    owner = make_user()
    title = _title("Gym")
    today_in_tokyo = dt.datetime.now(TOKYO).date()
    grid = _month_grid(today_in_tokyo)
    first_monday = grid[1]
    page = page_for(owner, **IN_TOKYO)

    # six in the morning in Tokyo is still Sunday in UTC and in Europe
    editor = _open_editor(page)
    editor.get_by_placeholder("Event title").fill(title)
    _fill_when(editor, first_monday, "06:00", "07:00")
    _repeat(editor, "Monday – Friday")
    _save(editor, "Create")

    weekdays = [day for day in grid[1:] if day.weekday() < 5]
    expect(_chip(page, title)).to_have_count(len(weekdays))
    assert _days_showing(page, title) == [day.day for day in weekdays]
    saved = _saved(owner, title, today_in_tokyo)
    assert saved["start_at"] == to_ns(dt.datetime.combine(first_monday, dt.time(6), TOKYO))


@pytest.mark.regression
def test_saving_a_later_occurrence_keeps_the_series_start(page_for, make_user):
    owner = make_user()
    title, renamed = _title("Review"), _title("Design review")
    series_start = _this_month(1, 10)
    event_id = _event(owner, title, series_start, rrule="FREQ=WEEKLY")
    page = page_for(owner)
    page.goto("/calendar")

    expect(_chip(page, title).first).to_be_visible()
    _chip(page, title).nth(2).click()
    editor = page.get_by_role("dialog")
    editor.get_by_placeholder("Event title").fill(renamed)
    _save(editor, "Save")

    with owner.client() as client:
        stored = client.get(f"/api/v1/calendars/events/{event_id}").json()
    assert stored["start_at"] == to_ns(series_start), (
        "saving the third occurrence moved the series to its day and dropped the earlier ones"
        " (#30970, fix PR #30971)"
    )


# ---------------------------------------------------------------- calendars and sources


def test_hiding_a_calendar_in_the_sidebar_hides_its_events(page_for, make_user):
    owner = make_user()
    name, title = _title("Club"), _title("Match")
    page = page_for(owner)
    page.goto("/calendar")

    page.get_by_title("New calendar").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Calendar name").fill(name)
    dialog.get_by_role("button", name="Create", exact=True).click()
    expect(page.get_by_role("button", name=name)).to_be_visible()
    with owner.client() as client:
        calendars = client.get("/api/v1/calendars/").json()
    calendar_id = next(calendar["id"] for calendar in calendars if calendar["name"] == name)
    _event(owner, title, _this_month(19, 15), calendar_id=calendar_id)

    page.reload()
    expect(_chip(page, title)).to_be_visible()
    page.get_by_role("button", name=name).click()
    expect(_chip(page, title)).to_have_count(0)
    page.get_by_role("button", name=name).click()
    expect(_chip(page, title)).to_be_visible()


def test_an_event_the_model_made_shows_at_its_local_time(page_for, make_user, upstream):
    owner = make_user()
    title = _title("Pharmacy")
    tokyo_month = dt.datetime.now(TOKYO).date().replace(day=20)
    with owner.client() as client:
        set_timezone(client, "Asia/Tokyo")
        run_tool(
            client,
            upstream,
            "create_calendar_event",
            {"title": title, "start": f"{tokyo_month.isoformat()} 09:00"},
        )

    page = page_for(owner, **IN_TOKYO)
    page.goto("/calendar")

    expect(_chip(page, title)).to_have_text(re.compile(rf"^\s*9:00\W*AM\s*{title}\s*$"))
    assert _days_showing(page, title) == [20]


def test_another_account_sees_an_event_only_once_its_calendar_is_shared(page_for, make_user):
    owner, reader, stranger = make_user(), make_user(), make_user()
    title, own_title = _title("Offsite"), _title("Own")
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with owner.client() as client:
        shared = client.post(
            "/api/v1/calendars/create", json={"name": "Team", "access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text
    _event(owner, title, _this_month(22, 11), calendar_id=shared.json()["id"])
    _event(stranger, own_title, _this_month(22, 13))

    stranger_page = page_for(stranger)
    stranger_page.goto("/calendar")
    expect(_chip(stranger_page, own_title)).to_be_visible()
    expect(_chip(stranger_page, title)).to_have_count(0)

    reader_page = page_for(reader)
    reader_page.goto("/calendar")
    expect(_chip(reader_page, title)).to_be_visible()
