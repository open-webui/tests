"""Automation, calendar and search regressions fixed in 0.11.1, seen through the API.

* Recurring events were expanded in the server's timezone: the scan window and the rule's
  anchor came from a naive `datetime.fromtimestamp`, then each occurrence was stamped in the
  user's zone, so every occurrence `GET /api/v1/calendars/events` listed was shifted by the gap
  between the two zones (`d721b0d19`).
* `POST /api/v1/automations/create` accepted a rule with `COUNT=` and no `DTSTART`; with no
  anchor the count never ran out and the automation ran forever (`2ab0311b9`, #27781).
* The reminder poll compared a user-writable `meta.alert_minutes` to a number, so one event
  holding text there raised on every poll and no reminder went out for anyone (`abc69000b`,
  #28790). A sent reminder shows as `meta.alerted_at` on the event.
* Searching a JSON column matches the text a JSON encoder wrote, and the stdlib codec (the
  default) escapes non-ASCII while orjson (`ENABLE_ORJSON`) writes it raw. Automation search
  matched only the raw spelling and model tag search only the escaped one, so each found
  nothing under one of the two codecs (`189c14fc4`, #28399).

The timer and fork fixes in the same unit file stay there: their state lives on internal
chats that no endpoint lists.

Twin of unit/models/test_automations_and_calendar.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the expansion back on the
server's clock, the COUNT check removed, `alert_minutes` compared unchecked, the single-spelling
searches restored): occurrences move by the zone gap, a COUNT rule without DTSTART is stored, no
reminder is sent while a text window exists, and a CJK prompt or tag is missed under one codec.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from typing import Iterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from harness.actors import create_user
from harness.calendar_api import (
    DAY_NS,
    MINUTE_NS,
    create_event,
    default_calendar_id,
    events_between,
    set_timezone,
    to_ns,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# The reminder tests need a scheduler that polls every second, and the search tests need rows
# written by orjson; one extra instance serves both.
SECOND_INSTANCE_ENV = {"SCHEDULER_POLL_INTERVAL": "1", "ENABLE_ORJSON": "true"}

# Fixed-offset zones only: a DST fold would make "same wall clock" ambiguous.
FIXED_OFFSET_ZONES = ("Asia/Tokyo", "Asia/Kolkata", "UTC", "Pacific/Kiritimati", "Pacific/Honolulu")

# A Tuesday, far from any DST boundary in the zones above.
EVENT_START = dt.datetime(2026, 3, 10, 9, 30, tzinfo=dt.timezone.utc)
EVENT_START_NS = to_ns(EVENT_START)

CJK = "天気"
REMINDER_WAIT_SECONDS = 15


def _server_zone_differs(zone: str) -> bool:
    local_offset = EVENT_START.astimezone().utcoffset()
    return EVENT_START.astimezone(ZoneInfo(zone)).utcoffset() != local_offset


def _foreign_zone() -> str:
    """A zone whose offset differs from the server's, so a server-clock expansion shows."""
    return next(zone for zone in FIXED_OFFSET_ZONES if _server_zone_differs(zone))


def _daily_thrice_starts(client: httpx.Client) -> list[int]:
    created = create_event(
        client,
        default_calendar_id(client),
        start_at=EVENT_START_NS,
        end_at=EVENT_START_NS + 30 * MINUTE_NS,
        rrule="RRULE:FREQ=DAILY;COUNT=3",
    )
    assert created.status_code == 200, created.text
    listed = events_between(client, EVENT_START_NS - DAY_NS, EVENT_START_NS + 3 * DAY_NS)
    return [event["start_at"] for event in listed if event["id"] == created.json()["id"]]


# ---------------------------------------------------------------- narrow: the user's timezone


def test_a_recurring_event_is_listed_at_its_own_start_in_the_users_zone(make_user):
    zone = _foreign_zone()
    with make_user().client() as client:
        set_timezone(client, zone)
        starts = _daily_thrice_starts(client)

    assert starts, f"no occurrences listed for {zone}"
    drift_hours = (starts[0] - EVENT_START_NS) / (60 * MINUTE_NS)
    assert starts[0] == EVENT_START_NS, (
        f"the first occurrence is {drift_hours:+.2f}h off the event start for a user in {zone}"
    )


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize("zone", FIXED_OFFSET_ZONES)
def test_occurrences_start_at_the_event_and_step_a_day_in_every_zone(make_user, zone):
    with make_user().client() as client:
        set_timezone(client, zone)
        starts = _daily_thrice_starts(client)

    assert starts[0] == EVENT_START_NS, f"the listing in {zone} does not start at the event"
    assert [later - earlier for earlier, later in zip(starts, starts[1:])] == [DAY_NS, DAY_NS]


# ---------------------------------------------------------------- nearby


def test_a_user_without_a_zone_still_sees_the_event_start(make_user):
    with make_user().client() as client:
        starts = _daily_thrice_starts(client)

    assert starts[0] == EVENT_START_NS


def test_a_one_off_event_is_listed_unchanged(make_user):
    with make_user().client() as client:
        set_timezone(client, "Asia/Tokyo")
        created = create_event(
            client, default_calendar_id(client), start_at=EVENT_START_NS, end_at=None
        )
        listed = events_between(client, EVENT_START_NS - DAY_NS, EVENT_START_NS + DAY_NS)

    assert [event["start_at"] for event in listed if event["id"] == created.json()["id"]] == [
        EVENT_START_NS
    ]


# ---------------------------------------------------------------- narrow: COUNT without DTSTART


@pytest.fixture
def automations(admin):
    """Create automations as the admin (never due: inactive) and delete them afterwards."""
    client = admin.client()
    created: list[str] = []

    def create(rrule: str, prompt: str = "write the report") -> httpx.Response:
        response = client.post(
            "/api/v1/automations/create",
            json={
                "name": "Report",
                "data": {"prompt": prompt, "model_id": "mock-model", "rrule": rrule},
                "is_active": False,
            },
        )
        if response.status_code == 200:
            created.append(response.json()["id"])
        return response

    yield create
    for automation_id in created:
        client.delete(f"/api/v1/automations/{automation_id}/delete")
    client.close()


def test_a_count_rule_without_a_dtstart_is_refused(automations):
    response = automations("RRULE:FREQ=DAILY;COUNT=5")

    assert response.status_code == 400, (
        f"a COUNT rule with no anchor was stored and would never run out (#27781): {response.text}"
    )
    assert "DTSTART" in response.json()["detail"]


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize(
    "rule",
    [
        "RRULE:FREQ=WEEKLY;COUNT=2;BYDAY=MO",
        "RRULE:FREQ=MONTHLY;COUNT=12",
        "RRULE:FREQ=HOURLY;INTERVAL=6;COUNT=4",
        "rrule:freq=daily;count=5",
        "RRULE:FREQ=DAILY;INTERVAL=2;COUNT=3;WKST=MO",
    ],
)
def test_every_count_rule_needs_an_anchor(automations, rule):
    response = automations(rule)

    assert response.status_code == 400, response.text
    assert "DTSTART" in response.json()["detail"]


# ---------------------------------------------------------------- nearby


@pytest.mark.parametrize(
    "rule",
    [
        "DTSTART:20260101T090000\nRRULE:FREQ=DAILY;COUNT=5000",
        "DTSTART:20260101T090000\nRRULE:FREQ=WEEKLY;COUNT=500;BYDAY=MO",
        "RRULE:FREQ=DAILY",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE",
    ],
)
def test_anchored_and_unlimited_rules_are_still_accepted(automations, rule):
    assert automations(rule).status_code == 200


def test_an_exhausted_rule_is_still_refused_for_having_no_future_runs(automations):
    response = automations("DTSTART:20200101T090000\nRRULE:FREQ=DAILY;UNTIL=20200201T090000")

    assert response.status_code == 400
    assert "future" in response.json()["detail"]


# ---------------------------------------------------------------- the second instance


@pytest.fixture
def second_instance(instance_with):
    return instance_with(SECOND_INSTANCE_ENV)


@pytest.fixture
def remind(second_instance):
    """Create events on the second instance for fresh accounts, and delete them afterwards."""
    clients: list[tuple[httpx.Client, str]] = []

    def create(minutes_ahead: int, meta: dict | None = None) -> tuple[httpx.Client, str]:
        client = create_user(second_instance).client()
        start = time.time_ns() + minutes_ahead * MINUTE_NS
        created = create_event(client, default_calendar_id(client), start_at=start, meta=meta)
        assert created.status_code == 200, created.text
        clients.append((client, created.json()["id"]))
        return client, created.json()["id"]

    yield create
    for client, event_id in clients:
        client.delete(f"/api/v1/calendars/events/{event_id}/delete")
        client.close()


def _alerted(client: httpx.Client, event_id: str) -> bool:
    event = client.get(f"/api/v1/calendars/events/{event_id}").json()
    return bool((event.get("meta") or {}).get("alerted_at"))


def _wait_until_alerted(client: httpx.Client, event_id: str) -> bool:
    deadline = time.monotonic() + REMINDER_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _alerted(client, event_id):
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------- narrow: text in alert_minutes


# Each poll that sees the bystander's event also sees the one created before it.
@pytest.mark.parametrize("window", ["10", ["10"], {"minutes": 10}, "not a number"])
def test_a_non_numeric_alert_window_does_not_stop_everyones_reminders(remind, window):
    junk_client, junk_event = remind(5, {"alert_minutes": window})
    bystander_client, plain_event = remind(5)

    assert _wait_until_alerted(bystander_client, plain_event), (
        f"no reminder went out within {REMINDER_WAIT_SECONDS}s while another user's event held "
        f"alert_minutes={window!r} (#28790)"
    )
    assert _alerted(junk_client, junk_event), "the junk window should fall back to the default one"


# ---------------------------------------------------------------- nearby


def test_a_numeric_alert_window_is_still_honoured(remind):
    later_client, later_event = remind(20, {"alert_minutes": 10})
    soon_client, soon_event = remind(5, {"alert_minutes": 10})

    assert _wait_until_alerted(soon_client, soon_event)
    assert not _alerted(later_client, later_event), "an event outside its own window was reminded"


def test_a_negative_alert_window_still_means_no_reminder(remind):
    muted_client, muted_event = remind(5, {"alert_minutes": -1})
    control_client, control_event = remind(5)

    assert _wait_until_alerted(control_client, control_event)
    assert not _alerted(muted_client, muted_event)


# ---------------------------------------------------------------- narrow: both JSON spellings


@pytest.fixture(params=["stdlib-codec", "orjson-codec"])
def any_codec_admin(request, instance, instance_with) -> Iterator[httpx.Client]:
    """The admin of an instance writing JSON with the stdlib codec, or with orjson."""
    chosen = instance if request.param == "stdlib-codec" else instance_with(SECOND_INSTANCE_ENV)
    with chosen.client() as client:
        yield client


def test_automation_search_finds_a_non_ascii_prompt(any_codec_admin):
    created = any_codec_admin.post(
        "/api/v1/automations/create",
        json={
            "name": f"forecast {uuid.uuid4().hex[:6]}",
            "data": {
                "prompt": f"{CJK} report",
                "model_id": "mock-model",
                "rrule": "RRULE:FREQ=DAILY",
            },
            "is_active": False,
        },
    )
    assert created.status_code == 200, created.text
    automation_id = created.json()["id"]
    try:
        found = any_codec_admin.get("/api/v1/automations/list", params={"query": CJK}).json()
    finally:
        any_codec_admin.delete(f"/api/v1/automations/{automation_id}/delete")

    assert automation_id in [item["id"] for item in found["items"]], (
        "searching an automation's non-ASCII prompt missed the row this codec wrote"
        " (#28399, on Postgres #31422)"
    )


def _tagged_model(client: httpx.Client, tag: str) -> str:
    model_id = f"tagged-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": "mock-model",
            "name": model_id,
            "meta": {"tags": [{"name": tag}]},
            "params": {},
        },
    )
    assert created.status_code == 200, created.text
    return model_id


def _models_tagged(client: httpx.Client, tag: str) -> list[str]:
    listed = client.get("/api/v1/models/list", params={"tag": tag})
    assert listed.status_code == 200, listed.text
    return [item["id"] for item in listed.json()["items"]]


def test_model_tag_search_finds_a_non_ascii_tag(any_codec_admin):
    model_id = _tagged_model(any_codec_admin, CJK)
    try:
        found = _models_tagged(any_codec_admin, CJK)
    finally:
        any_codec_admin.post("/api/v1/models/model/delete", json={"id": model_id})

    assert model_id in found, (
        "a non-ASCII model tag missed the row this codec wrote (#28399, on Postgres #31422)"
    )


# ---------------------------------------------------------------- nearby


def test_an_ascii_tag_still_matches_whole_tags_case_insensitively(any_codec_admin):
    model_id = _tagged_model(any_codec_admin, "Weather")
    try:
        found = _models_tagged(any_codec_admin, "weather")
        unrelated = _models_tagged(any_codec_admin, "weathervane")
    finally:
        any_codec_admin.post("/api/v1/models/model/delete", json={"id": model_id})

    assert model_id in found
    assert model_id not in unrelated
