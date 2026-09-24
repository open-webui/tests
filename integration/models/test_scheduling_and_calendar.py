"""Calendar invitation and schedule regressions fixed in 0.11.0, seen through the API.

* `set_attendees` took each attendee's RSVP from the organiser's form and rewrote the whole
  attendee list, so an organiser could accept an invitation on the invitee's behalf and every
  edit reset an answer already given. The range listing also kept a declined invitation on the
  invitee's calendar (`9b635d8f3`, #27007).
* The schedule parser read FREQ out of the rule text case-sensitively, so a rule with lowercase
  keys was never snapped to clock boundaries, and it accepted rules it cannot run (`EXRULE`, two
  `RRULE` lines, `INTERVAL=0`) (`c4ae8c8`, `2d928df`, #27470). A far-past DTSTART is pinned by
  integration/security/test_recurrence_rule_parsing.py.
* The next runs were computed from the server's clock while the user's clock decided which were
  in the future, so for a user in a zone behind the server a quarter-hourly schedule's next run
  lay hours ahead (`b3aead2`, #26954).

Automations are created switched off, so the scheduler never runs them.

Twin of unit/models/test_scheduling_and_calendar.py.

Discriminates: passes on dev bbfa876af; fails with each fix reverted (the form's status stored,
the declined filter dropped, FREQ scraped case-sensitively, the three refusals removed, the rule
anchored on the server's clock): an organiser's "accepted" sticks, an edit resets the RSVP, the
declined event stays listed, a lowercase rule runs off the clock, the unsupported rules are
stored or refused for another reason and the next run lies hours ahead.
"""

from __future__ import annotations

import datetime as dt
import time
from zoneinfo import ZoneInfo

import httpx
import pytest

from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    SECOND_NS,
    create_event,
    default_calendar_id,
    events_between,
    set_timezone,
    update_event,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# UTC-11 all year, so behind the clock of practically every server.
ZONE_BEHIND = "Pacific/Pago_Pago"


@pytest.fixture
def invitation(make_user):
    """An event an organiser created an hour from now, and the invitee's client."""
    organiser, invitee = make_user(), make_user()
    start = time.time_ns() + HOUR_NS
    with organiser.client() as organiser_client, invitee.client() as invitee_client:
        created = create_event(
            organiser_client,
            default_calendar_id(organiser_client),
            start_at=start,
            end_at=start + HOUR_NS,
            attendees=[{"user_id": invitee.id}],
        )
        assert created.status_code == 200, created.text
        yield organiser_client, invitee_client, invitee.id, created.json()["id"]


def _listed(client: httpx.Client) -> dict[str, dict]:
    now = time.time_ns()
    return {event["id"]: event for event in events_between(client, now - DAY_NS, now + DAY_NS)}


def _status_of(client: httpx.Client, event_id: str, user_id: str) -> str | None:
    """The attendee's RSVP as their own calendar lists it."""
    attendees = _listed(client)[event_id]["attendees"]
    return next((entry["status"] for entry in attendees if entry["user_id"] == user_id), None)


def _rsvp(client: httpx.Client, event_id: str, status: str) -> None:
    answered = client.post(f"/api/v1/calendars/events/{event_id}/rsvp", json={"status": status})
    assert answered.status_code == 200, answered.text


# ---------------------------------------------------------------- narrow: RSVP is the invitee's


def test_an_organiser_cannot_accept_on_the_invitees_behalf(make_user):
    organiser, invitee = make_user(), make_user()
    start = time.time_ns() + HOUR_NS
    with organiser.client() as client:
        created = create_event(
            client,
            default_calendar_id(client),
            start_at=start,
            attendees=[{"user_id": invitee.id, "status": "accepted"}],
        )
    assert created.status_code == 200, created.text

    with invitee.client() as client:
        status = _status_of(client, created.json()["id"], invitee.id)
    assert status == "pending", f"the organiser answered the invitation for the invitee: {status}"


def test_an_edit_keeps_the_answer_the_invitee_gave(invitation, make_user):
    organiser_client, invitee_client, invitee_id, event_id = invitation
    _rsvp(invitee_client, event_id, "accepted")

    edited = update_event(
        organiser_client,
        event_id,
        attendees=[{"user_id": invitee_id}, {"user_id": make_user().id}],
    )

    assert edited.status_code == 200, edited.text
    assert _status_of(invitee_client, event_id, invitee_id) == "accepted", (
        "editing the attendee list reset an RSVP the invitee had already given"
    )


def test_a_declined_invitation_leaves_the_invitees_calendar(invitation):
    organiser_client, invitee_client, _, event_id = invitation
    assert event_id in _listed(invitee_client)

    _rsvp(invitee_client, event_id, "declined")

    assert event_id not in _listed(invitee_client), (
        "a declined invitation is still listed on the invitee's calendar"
    )
    assert event_id in _listed(organiser_client), "declining removed the organiser's event"


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize("sent", ["accepted", "declined", "tentative", "pending"])
def test_no_status_the_organiser_sends_replaces_the_invitees(invitation, sent):
    organiser_client, invitee_client, invitee_id, event_id = invitation
    given = "accepted" if sent == "tentative" else "tentative"
    _rsvp(invitee_client, event_id, given)

    edited = update_event(
        organiser_client, event_id, attendees=[{"user_id": invitee_id, "status": sent}]
    )

    assert edited.status_code == 200, edited.text
    assert _status_of(invitee_client, event_id, invitee_id) == given


@pytest.mark.parametrize("status", ["pending", "accepted", "tentative"])
def test_every_answer_but_declined_keeps_the_event_listed(invitation, status):
    _, invitee_client, _, event_id = invitation

    _rsvp(invitee_client, event_id, status)

    assert event_id in _listed(invitee_client)


# ---------------------------------------------------------------- nearby


def test_attendee_meta_is_still_the_organisers_to_write(make_user):
    organiser, invitee = make_user(), make_user()
    with organiser.client() as client:
        created = create_event(
            client,
            default_calendar_id(client),
            start_at=time.time_ns() + HOUR_NS,
            attendees=[{"user_id": invitee.id, "meta": {"role": "chair"}}],
        )
    assert created.status_code == 200, created.text
    assert [(entry["status"], entry["meta"]) for entry in created.json()["attendees"]] == [
        ("pending", {"role": "chair"})
    ]


# ---------------------------------------------------------------- schedules


@pytest.fixture
def scheduler(make_user, preserve, admin):
    """`scheduler(rule, zone)` creates a switched-off automation as a fresh user in `zone`."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    created: list[tuple[httpx.Client, str]] = []

    def create(rule: str, zone: str = "UTC") -> httpx.Response:
        client = make_user().client()
        set_timezone(client, zone)
        response = client.post(
            "/api/v1/automations/create",
            json={
                "name": "schedule",
                "is_active": False,
                "data": {"prompt": "ping", "model_id": MOCK_MODEL_ID, "rrule": rule},
            },
        )
        if response.status_code == 200:
            created.append((client, response.json()["id"]))
        else:
            client.close()
        return response

    yield create
    for client, automation_id in created:
        client.delete(f"/api/v1/automations/{automation_id}/delete")
        client.close()


def _next_runs(response: httpx.Response, zone: str) -> list[dt.datetime]:
    assert response.status_code == 200, response.text
    return [
        dt.datetime.fromtimestamp(run / SECOND_NS, ZoneInfo(zone))
        for run in response.json()["next_runs"]
    ]


@pytest.mark.parametrize(
    ("rule", "step"),
    [
        ("RRULE:freq=MINUTELY;interval=5", dt.timedelta(minutes=5)),
        ("RRULE:freq=SECONDLY;interval=10", dt.timedelta(seconds=10)),
        ("RRULE:freq=HOURLY;interval=6", dt.timedelta(hours=6)),
        ("RRULE:FREQ=MINUTELY;INTERVAL=20", dt.timedelta(minutes=20)),
    ],
    ids=["minutely-lowercase", "secondly-lowercase", "hourly-lowercase", "minutely-uppercase"],
)
def test_a_sub_daily_rule_runs_on_clock_boundaries_in_either_case(scheduler, rule, step):
    runs = _next_runs(scheduler(rule), "UTC")
    midnight = runs[0].replace(hour=0, minute=0, second=0, microsecond=0)

    off_boundary = [run for run in runs if (run - midnight) % step]
    assert not off_boundary, f"{rule!r} runs off its {step} clock boundaries: {off_boundary}"
    assert {later - earlier for earlier, later in zip(runs, runs[1:])} == {step}


def test_a_quarter_hourly_rule_runs_next_within_the_quarter_hour_in_the_users_zone(scheduler):
    now = dt.datetime.now(ZoneInfo(ZONE_BEHIND))
    # the server shares this process's clock
    assert now.utcoffset() < dt.datetime.now().astimezone().utcoffset()

    runs = _next_runs(scheduler("RRULE:FREQ=MINUTELY;INTERVAL=15", ZONE_BEHIND), ZONE_BEHIND)

    # a few seconds of slack for a quarter-hour turning while the request is served
    assert dt.timedelta(0) < runs[0] - now <= dt.timedelta(minutes=15, seconds=30), (
        f"the next run is {runs[0] - now} away for a user in {ZONE_BEHIND}: the schedule was "
        "anchored on the server's clock"
    )
    assert (runs[0].minute % 15, runs[0].second) == (0, 0)
    assert {later - earlier for earlier, later in zip(runs, runs[1:])} == {dt.timedelta(minutes=15)}


@pytest.mark.parametrize(
    ("rule", "refusal"),
    [
        ("RRULE:FREQ=DAILY\nEXRULE:FREQ=WEEKLY", "EXRULE is not supported"),
        ("RRULE:FREQ=DAILY\nRRULE:FREQ=HOURLY", "only one RRULE is supported"),
        ("RRULE:FREQ=MINUTELY;INTERVAL=0", "INTERVAL must be a positive integer"),
        ("RRULE:FREQ=SECONDLY;INTERVAL=-1", "INTERVAL must be a positive integer"),
    ],
    ids=["exrule", "two-rrules", "zero-interval", "negative-interval"],
)
def test_a_rule_the_scheduler_cannot_run_is_refused_by_name(scheduler, rule, refusal):
    response = scheduler(rule)

    assert response.status_code == 400, f"{rule!r} was stored: {response.text}"
    assert refusal in response.json()["detail"]


def test_a_daily_rule_still_steps_a_day(scheduler):
    runs = _next_runs(scheduler("RRULE:FREQ=DAILY"), "UTC")

    assert {later - earlier for earlier, later in zip(runs, runs[1:])} == {dt.timedelta(days=1)}
