"""Calendar and automation schedule regressions fixed in 0.11.2, seen through the API.

* Moving an event to another date made it vanish: the range query behind
  `GET /api/v1/calendars/events` required `end_at > range_start` for any event with an end, and a
  move that rewrote `start_at` only left `end_at` on the old date. The query now also accepts
  `start_at >= range_start` (`1c5128c4a`, #29085, issue #29067).
* A DTSTART line inside an event's rule won over the event's own start in dateutil, so the
  listed occurrences ran on the rule's weekday, day of month and hour. The expansion now strips
  it (`a93c50803`, `8c0c7b3b6`).
* The same commits refuse a calendar rule that repeats more often than daily, on create and on
  update, with `ERROR_MESSAGES.CALENDAR_RRULE_TOO_FREQUENT` (422).
* The automation minimum interval measured a rule with its DTSTART left in, so a 29 February
  anchor made a yearly rule look four years apart and slip past a two-year minimum
  (`fd679e1da`).

Twin of unit/models/test_calendar_moves_and_recurrence.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the range query back to
`end_at > start`, the DTSTART kept in the expansion, the sub-daily check removed from the
calendar store, the DTSTART kept when measuring an automation's interval): the moved event is
missing from its new day, occurrences land on Thursdays at 17:00, the hourly rule is stored and
the leap-day automation is accepted.
"""

from __future__ import annotations

import datetime as dt
import time

import pytest

from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    create_event,
    default_calendar_id,
    events_between,
    set_timezone,
    to_ns,
    to_utc,
    update_event,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# A Tuesday at 09:30 UTC.
EVENT_START = dt.datetime(2026, 3, 10, 9, 30, tzinfo=dt.timezone.utc)
EVENT_START_NS = to_ns(EVENT_START)

# The Thursday before, at 17:00: a different weekday, day of month and hour.
FOREIGN_DTSTART = "DTSTART:20260305T170000"

TOO_FREQUENT = "more often than daily"
SUB_DAILY_RULES = (
    "RRULE:FREQ=HOURLY",
    "RRULE:FREQ=MINUTELY;INTERVAL=30",
    "RRULE:FREQ=HOURLY;INTERVAL=12",
    "RRULE:FREQ=SECONDLY;INTERVAL=45",
)
YEAR_SECONDS = 365 * 24 * 60 * 60


@pytest.fixture
def calendar(make_user):
    """A fresh account reading its calendar in UTC, with its own calendar id."""
    owner = make_user()
    client = owner.client()
    set_timezone(client, "UTC")
    yield client, default_calendar_id(client)
    client.close()


def _event_id(client, calendar_id: str, **fields) -> str:
    created = create_event(client, calendar_id, **fields)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _listed_ids(client, start_ns: int, end_ns: int) -> list[str]:
    return [event["id"] for event in events_between(client, start_ns, end_ns)]


def _occurrence_starts(client, calendar_id: str, rrule: str, days: int) -> list[dt.datetime]:
    event_id = _event_id(
        client, calendar_id, start_at=EVENT_START_NS, end_at=EVENT_START_NS + HOUR_NS, rrule=rrule
    )
    listed = events_between(client, EVENT_START_NS - DAY_NS, EVENT_START_NS + days * DAY_NS)
    return [to_utc(event["start_at"]) for event in listed if event["id"] == event_id]


def _days_from_now(days: int) -> int:
    return time.time_ns() + days * DAY_NS


# ---------------------------------------------------------------- narrow: moving an event


def test_an_event_moved_a_week_later_is_listed_on_its_new_day(calendar):
    client, calendar_id = calendar
    start = _days_from_now(1)
    event_id = _event_id(client, calendar_id, start_at=start, end_at=start + HOUR_NS)

    moved_start = start + 7 * DAY_NS
    moved = update_event(client, event_id, start_at=moved_start)
    assert moved.status_code == 200, moved.text

    assert event_id in _listed_ids(client, moved_start - HOUR_NS, moved_start + DAY_NS), (
        "the event vanished from the day it was moved to (#29067)"
    )


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize("end_offset_days", [-30, -7, -1, 0, 1])
def test_an_event_starting_inside_the_range_is_listed_whatever_its_end(calendar, end_offset_days):
    client, calendar_id = calendar
    start = _days_from_now(60)
    event_id = _event_id(
        client, calendar_id, start_at=start, end_at=start + end_offset_days * DAY_NS
    )

    assert event_id in _listed_ids(client, start - HOUR_NS, start + DAY_NS)


# ---------------------------------------------------------------- nearby


def test_an_event_outside_the_range_is_still_left_out(calendar):
    client, calendar_id = calendar
    start = _days_from_now(120)
    event_id = _event_id(client, calendar_id, start_at=start, end_at=start + HOUR_NS)

    assert event_id not in _listed_ids(client, start + 10 * DAY_NS, start + 20 * DAY_NS)
    assert event_id not in _listed_ids(client, start - 20 * DAY_NS, start - 10 * DAY_NS)


def test_an_event_that_began_before_the_range_and_is_still_running_is_listed(calendar):
    client, calendar_id = calendar
    start = _days_from_now(150)
    event_id = _event_id(client, calendar_id, start_at=start, end_at=start + 5 * DAY_NS)

    assert event_id in _listed_ids(client, start + 2 * DAY_NS, start + 3 * DAY_NS)


# ---------------------------------------------------------------- narrow: a DTSTART in the rule


def test_weekly_occurrences_fall_on_the_events_weekday_not_the_rules(calendar):
    client, calendar_id = calendar
    starts = _occurrence_starts(client, calendar_id, f"{FOREIGN_DTSTART}\nRRULE:FREQ=WEEKLY", 30)

    assert starts, "no occurrences listed"
    assert starts[0] == EVENT_START, f"the first occurrence is {starts[0]}, not the event start"
    assert {start.strftime("%a %H:%M") for start in starts} == {"Tue 09:30"}, (
        f"occurrences followed the rule's own DTSTART: {[str(start) for start in starts]}"
    )


def test_monthly_occurrences_keep_the_events_day_of_month(calendar):
    client, calendar_id = calendar
    starts = _occurrence_starts(client, calendar_id, f"{FOREIGN_DTSTART}\nRRULE:FREQ=MONTHLY", 70)

    assert starts[:3] == [EVENT_START, EVENT_START.replace(month=4), EVENT_START.replace(month=5)]


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize(
    ("freq", "days", "step"),
    [("DAILY", 6, dt.timedelta(days=1)), ("WEEKLY", 30, dt.timedelta(weeks=1))],
)
def test_every_frequency_steps_from_the_events_start(calendar, freq, days, step):
    client, calendar_id = calendar
    starts = _occurrence_starts(client, calendar_id, f"{FOREIGN_DTSTART}\nRRULE:FREQ={freq}", days)

    assert starts[0] == EVENT_START
    assert all(later - earlier == step for earlier, later in zip(starts, starts[1:]))


# ---------------------------------------------------------------- nearby


def test_a_rule_without_a_dtstart_is_unchanged(calendar):
    client, calendar_id = calendar
    starts = _occurrence_starts(client, calendar_id, "RRULE:FREQ=WEEKLY", 30)

    assert starts[0] == EVENT_START
    assert all(
        later - earlier == dt.timedelta(weeks=1) for earlier, later in zip(starts, starts[1:])
    )


def test_each_occurrence_keeps_the_events_duration(calendar):
    client, calendar_id = calendar
    event_id = _event_id(
        client,
        calendar_id,
        start_at=EVENT_START_NS,
        end_at=EVENT_START_NS + HOUR_NS,
        rrule=f"{FOREIGN_DTSTART}\nRRULE:FREQ=WEEKLY",
    )
    listed = events_between(client, EVENT_START_NS - DAY_NS, EVENT_START_NS + 30 * DAY_NS)

    durations = {event["end_at"] - event["start_at"] for event in listed if event["id"] == event_id}
    assert durations == {HOUR_NS}


# ---------------------------------------------------------------- narrow: sub-daily rules


def test_an_hourly_event_is_refused(calendar):
    client, calendar_id = calendar
    created = create_event(client, calendar_id, start_at=EVENT_START_NS, rrule="RRULE:FREQ=HOURLY")

    assert created.status_code == 422, f"an hourly event was stored: {created.text}"
    assert TOO_FREQUENT in created.text


def test_an_event_cannot_be_changed_to_repeat_hourly(calendar):
    client, calendar_id = calendar
    event_id = _event_id(client, calendar_id, start_at=EVENT_START_NS, rrule="RRULE:FREQ=DAILY")

    updated = update_event(client, event_id, rrule="RRULE:FREQ=HOURLY")

    assert updated.status_code == 422, f"an event was switched to hourly: {updated.text}"
    assert TOO_FREQUENT in updated.text
    stored = client.get(f"/api/v1/calendars/events/{event_id}").json()
    assert stored["rrule"] == "RRULE:FREQ=DAILY"


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize("rule", SUB_DAILY_RULES)
def test_every_sub_daily_rule_is_refused_on_create_and_update(calendar, rule):
    client, calendar_id = calendar
    created = create_event(client, calendar_id, start_at=EVENT_START_NS, rrule=rule)
    event_id = _event_id(client, calendar_id, start_at=EVENT_START_NS)
    updated = update_event(client, event_id, rrule=rule)

    assert (created.status_code, updated.status_code) == (422, 422), (created.text, updated.text)
    assert TOO_FREQUENT in created.text and TOO_FREQUENT in updated.text


# ---------------------------------------------------------------- nearby


@pytest.mark.parametrize(
    "rule",
    [
        "RRULE:FREQ=DAILY",
        "RRULE:FREQ=DAILY;INTERVAL=2",
        "RRULE:FREQ=WEEKLY;BYDAY=TU",
        "RRULE:FREQ=MONTHLY",
        None,
    ],
)
def test_daily_and_coarser_rules_are_accepted(calendar, rule):
    client, calendar_id = calendar
    created = create_event(client, calendar_id, start_at=EVENT_START_NS, rrule=rule)

    assert created.status_code == 200, created.text
    assert created.json()["rrule"] == rule


# ---------------------------------------------------------------- narrow: the automation minimum


@pytest.fixture
def automation_author(make_user, preserve, admin):
    """A user allowed to create automations, with the minimum interval left to the test."""
    preserve("permissions", "admin_config")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    author = make_user()
    created: list[tuple] = []
    yield author, created
    for actor, automation_id in created:
        with actor.client() as client:
            client.delete(f"/api/v1/automations/{automation_id}/delete")


def _set_min_interval(admin, seconds: int) -> None:
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        updated = client.post(
            "/api/v1/auths/admin/config", json={**config, "AUTOMATION_MIN_INTERVAL": seconds}
        )
    assert updated.status_code == 200, updated.text


def _create_automation(actor, created: list[tuple], rrule: str):
    with actor.client() as client:
        response = client.post(
            "/api/v1/automations/create",
            json={
                "name": "Yearly report",
                "data": {"prompt": "write the report", "model_id": "mock-model", "rrule": rrule},
                "is_active": False,
            },
        )
    if response.status_code == 200:
        created.append((actor, response.json()["id"]))
    return response


@pytest.mark.parametrize(
    ("rule", "minimum_years"),
    [
        ("DTSTART:20240229T090000\nRRULE:FREQ=YEARLY", 2),
        ("DTSTART:20240229T090000\nRRULE:FREQ=YEARLY;INTERVAL=2", 3),
    ],
    ids=["yearly", "every-two-years"],
)
def test_a_leap_day_anchor_does_not_stretch_a_rule_past_the_minimum(
    automation_author, admin, rule, minimum_years
):
    author, created = automation_author
    _set_min_interval(admin, minimum_years * YEAR_SECONDS)

    response = _create_automation(author, created, rule)

    assert response.status_code == 400, (
        f"a 29 February anchor made the rule look four years apart and it was accepted under "
        f"a {minimum_years}-year minimum: {response.text}"
    )
    assert "too frequent" in response.text


# ---------------------------------------------------------------- nearby


def test_a_rule_longer_than_the_minimum_is_still_accepted(automation_author, admin):
    author, created = automation_author
    _set_min_interval(admin, 2 * YEAR_SECONDS)

    response = _create_automation(author, created, "RRULE:FREQ=YEARLY;INTERVAL=5")

    assert response.status_code == 200, response.text


def test_an_admin_is_not_held_to_the_minimum(automation_author, admin):
    _, created = automation_author
    _set_min_interval(admin, 2 * YEAR_SECONDS)

    response = _create_automation(admin, created, "DTSTART:20240229T090000\nRRULE:FREQ=YEARLY")

    assert response.status_code == 200, response.text
