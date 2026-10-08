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
* Searching a JSON column matches the text a JSON encoder wrote, and a row holds non-ASCII raw
  or escaped: stdlib json escaped it until #31615 (321a24dfe) had every codec write it raw, so
  rows saved before an upgrade keep the escapes. Automation search matched only the raw spelling
  and model tag search only the escaped one, so each missed the rows of the other spelling
  (`189c14fc4`, #28399). The escaped rows are rewritten in the database as stdlib json left them.
* Forking a waiting timer chat copied its `meta`, and the scheduler claimed timers by `meta`,
  so the fork was a second claim target and the timer fired twice. Timers now hang off the
  `chat.timer_at` column, which a fork does not carry (`16c2a9eda`, #27663, issues
  #27622/#27745). The model sets a real timer through the `timer` tool (behind
  ENABLE_SUBAGENTS); no endpoint lists the internal timer chat, so its id is read from the
  instance's database, and the owner forks it over the API.

The timer whose chat completion raised (`f5a5a434b`, #27785) is pinned by
integration/chat/test_failed_timer_error.py: the reply it leaves carries the error.

Twin of unit/models/test_automations_and_calendar.py.

`test_a_forked_timer_chat_does_not_fire_a_second_time` is red on dev 62f70a844: since de73bb830 a
chat request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the expansion back on the
server's clock, the COUNT check removed, `alert_minutes` compared unchecked, the single-spelling
searches restored): occurrences move by the zone gap, a COUNT rule without DTSTART is stored, no
reminder is sent while a text window exists, and a CJK prompt or tag is missed in one spelling. On
dev ef67cc3fa, claiming timers by `meta.timer_at` again fires the forked timer a second time.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from typing import Iterator
from zoneinfo import ZoneInfo

import httpx
import pytest
import sqlalchemy

from harness import backends
from harness import upstream as reply
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
from harness.chat import ask
from harness.json_codecs import stored_text

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# The reminder tests need a scheduler that polls every second.
SECOND_INSTANCE_ENV = {"SCHEDULER_POLL_INTERVAL": "1"}

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


@pytest.fixture(params=["raw", "escaped"])
def spelling(request) -> str:
    """How a row holds non-ASCII: raw as written now, or escaped as stdlib json saved it."""
    return request.param


@pytest.fixture
def admin_client(admin) -> Iterator[httpx.Client]:
    with admin.client() as client:
        yield client


def _respell(instance, spelling: str, table: str, column: str, row_id: str) -> None:
    """Rewrite a stored JSON column in `spelling`; stdlib json's default escapes non-ASCII."""
    if spelling == "raw":
        return
    stored = json.loads(stored_text(instance, table, column, row_id))
    backends.write_rows(
        instance,
        f'UPDATE "{table}" SET "{column}" = :value WHERE id = :row_id',
        [{"value": json.dumps(stored), "row_id": row_id}],
    )
    assert "\\u" in stored_text(instance, table, column, row_id)


def test_automation_search_finds_a_non_ascii_prompt(instance, admin_client, spelling):
    created = admin_client.post(
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
        _respell(instance, spelling, "automation", "data", automation_id)
        found = admin_client.get("/api/v1/automations/list", params={"query": CJK}).json()
    finally:
        admin_client.delete(f"/api/v1/automations/{automation_id}/delete")

    assert automation_id in [item["id"] for item in found["items"]], (
        f"searching an automation's non-ASCII prompt missed a row stored {spelling}"
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


def test_model_tag_search_finds_a_non_ascii_tag(instance, admin_client, spelling):
    model_id = _tagged_model(admin_client, CJK)
    try:
        _respell(instance, spelling, "model", "meta", model_id)
        found = _models_tagged(admin_client, CJK)
    finally:
        admin_client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert model_id in found, (
        f"a non-ASCII model tag missed a row stored {spelling} (#28399, on Postgres #31422)"
    )


# ---------------------------------------------------------------- nearby


def test_an_ascii_tag_still_matches_whole_tags_case_insensitively(admin_client):
    model_id = _tagged_model(admin_client, "Weather")
    try:
        found = _models_tagged(admin_client, "weather")
        unrelated = _models_tagged(admin_client, "weathervane")
    finally:
        admin_client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert model_id in found
    assert model_id not in unrelated


# ---------------------------------------------------------------- narrow: a forked timer chat

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "The bread is out of the oven."


@pytest.fixture
def timers_enabled(preserve, admin):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def _set_timer(client: httpx.Client, upstream, at: str) -> None:
    upstream.queue(reply.tool_call("timer", {"prompt": TIMER_PROMPT, "at": at}), reply.text("Set."))
    _turn, message = ask(client, "tell me when the bread is done")
    assert message["content"].endswith("Set."), message


def _pending_timer_ids(instance, owner_id: str) -> list[str]:
    """The owner's waiting timer chats: no endpoint lists these internal chats."""
    engine = sqlalchemy.create_engine(instance.database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.text(
                    "SELECT id FROM chat WHERE user_id = :owner AND timer_at IS NOT NULL"
                ),
                {"owner": owner_id},
            )
            return [row[0] for row in rows]
    finally:
        engine.dispose()


def _timer_prompts_sent(upstream) -> int:
    last_messages = [request["messages"][-1] for request in upstream.chat_requests()]
    return sum(TIMER_PROMPT in str(message.get("content")) for message in last_messages)


def _wait_for_timer_prompts(upstream, count: int, within: float) -> int:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline and _timer_prompts_sent(upstream) < count:
        time.sleep(0.2)
    return _timer_prompts_sent(upstream)


def test_a_forked_timer_chat_does_not_fire_a_second_time(
    timers_enabled, instance, make_user, upstream
):
    owner = make_user()
    with owner.client() as client:
        _set_timer(client, upstream, at="4s")
        [timer_id] = _pending_timer_ids(instance, owner.id)
        forked = client.post(f"/api/v1/chats/{timer_id}/fork")
        assert forked.status_code == 200, forked.text

    assert _wait_for_timer_prompts(upstream, 1, within=20) == 1, "the timer never fired"
    # polled every second, so a second claim would fire within a few seconds
    assert _wait_for_timer_prompts(upstream, 2, within=5) == 1, (
        "the fork of a waiting timer chat was claimed as a second timer and fired again "
        "(#27622, #27745)"
    )


# ---------------------------------------------------------------- nearby


def test_a_timer_not_yet_due_does_not_fire(timers_enabled, instance, make_user, upstream):
    owner = make_user()
    with owner.client() as client:
        _set_timer(client, upstream, at="1h")

    assert _pending_timer_ids(instance, owner.id), "the timer was not stored as waiting"
    assert _wait_for_timer_prompts(upstream, 1, within=4) == 0, "a timer fired an hour early"
