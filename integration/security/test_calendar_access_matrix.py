"""Journey: who may see, fill, change, share and delete a shared calendar and its events.

The owner shares a calendar with a reader (read) and a writer (read and write), directly or
through a group. A stranger is refused every route; the reader sees the calendar and its events
but cannot create, change or delete one; the writer and the admin may do all of that. Changing
the grants and deleting the calendar are the owner's and the admin's, so the writer is refused
those. After a refused write the owner still sees the same calendar, grants and events. Moving an
event needs write access on the calendar it moves into as well.

Discriminates: in a backend copy, asking for `read` instead of `write` in the event update
handler's calendar check turns the `/events/{event_id}/update` rows red (the reader gets 200 and
the title changes), dropping the owner-only check on grants from the calendar update handler
turns the grant row red (the writer gets 200 and the grants are gone), and dropping the
destination check from the event update handler turns the foreign-calendar moves red (200, the
event moves).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import ROLES, Shareable, attempts, cast, grant
from harness.actors import Actor
from harness.calendar_api import (
    DAY_NS,
    HOUR_NS,
    create_event,
    events_between,
    to_utc,
    update_event,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CALENDAR = Shareable(
    create_path="/api/v1/calendars/create",
    create_body=lambda: {"name": f"shared {uuid.uuid4().hex[:8]}"},
    access_path="/api/v1/calendars/{id}/update",
)
START_NS = 1_900_000_000 * 1_000_000_000  # a fixed day in 2030, clear of any real schedule

OK = 200
REFUSED = 403
READ = {"owner", "reader", "writer", "admin"}
WRITE = {"owner", "writer", "admin"}
OWNER = {"owner", "admin"}


def _event_in(owner: Actor, calendar_id: str) -> dict:
    with owner.client() as client:
        created = create_event(client, calendar_id, title="Planning", start_at=START_NS)
    assert created.status_code == 200, created.text
    return {"event_id": created.json()["id"]}


def _new_event(actor: Actor, fields: dict) -> dict:
    return {"calendar_id": fields["id"], "title": "Added", "start_at": START_NS + HOUR_NS}


def _fresh_name(actor: Actor, fields: dict) -> dict:
    return CALENDAR.create_body()


def _owner_view(client: httpx.Client, fields: dict) -> dict | None:
    calendar = client.get(f"/api/v1/calendars/{fields['id']}")
    if calendar.status_code != 200:
        return None
    listed = client.get(
        "/api/v1/calendars/events",
        params={
            "start": to_utc(START_NS - DAY_NS).isoformat(),
            "end": to_utc(START_NS + DAY_NS).isoformat(),
            "calendar_ids": fields["id"],
        },
    )
    assert listed.status_code == 200, listed.text
    return {
        "name": calendar.json()["name"],
        "grants": sorted(
            (entry["principal_id"], entry["permission"])
            for entry in calendar.json()["access_grants"]
        ),
        "events": sorted((event["id"], event["title"]) for event in listed.json()),
    }


# method, path, body, setup, who gets through; everyone else is answered 403
MATRIX = [
    ("GET", "/api/v1/calendars/{id}", None, None, READ),
    ("POST", "/api/v1/calendars/{id}/update", _fresh_name, None, WRITE),
    ("POST", "/api/v1/calendars/{id}/update", {"access_grants": []}, None, OWNER),
    ("DELETE", "/api/v1/calendars/{id}/delete", None, None, OWNER),
    ("POST", "/api/v1/calendars/events/create", _new_event, None, WRITE),
    ("GET", "/api/v1/calendars/events/{event_id}", None, _event_in, READ),
    ("POST", "/api/v1/calendars/events/{event_id}/update", {"title": "Moved"}, _event_in, WRITE),
    ("DELETE", "/api/v1/calendars/events/{event_id}/delete", None, _event_in, WRITE),
]
MATRIX_IDS = [
    "GET calendar",
    "rename calendar",
    "clear calendar grants",
    "DELETE calendar",
    "create event",
    "GET event",
    "update event",
    "DELETE event",
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize("method, path, body, setup, allowed", MATRIX, ids=MATRIX_IDS)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, setup, allowed, via, admin, make_user
):
    accounts = cast(CALENDAR, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, setup=setup, look=_owner_view)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        role: OK if role in allowed else REFUSED for role in ROLES
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != OK:
            assert attempt.after == attempt.before, f"a refused {role} changed the calendar"


def test_a_reader_sees_the_events_and_a_stranger_does_not(admin, make_user):
    accounts = cast(CALENDAR, admin, make_user)
    with accounts.owner.client() as client:
        calendar_id = client.post(CALENDAR.create_path, json=CALENDAR.create_body()).json()["id"]
        client.post(
            CALENDAR.access_path.format(id=calendar_id), json={"access_grants": accounts.grants}
        ).raise_for_status()
    event_id = _event_in(accounts.owner, calendar_id)["event_id"]

    seen = {}
    for role in ROLES:
        with accounts.actor(role).client() as client:
            listed = events_between(client, START_NS - DAY_NS, START_NS + DAY_NS)
        seen[role] = event_id in [event["id"] for event in listed]

    # the listing holds the admin's own calendars only
    assert seen == {
        "owner": True,
        "stranger": False,
        "reader": True,
        "writer": True,
        "admin": False,
    }


@pytest.mark.parametrize("permission", [None, "read"])
def test_an_event_cannot_move_into_a_calendar_the_mover_cannot_write(permission, make_user):
    mover, other = make_user(), make_user()
    grants = [grant("user", mover.id, permission)] if permission else []
    with other.client() as client:
        created = client.post(
            CALENDAR.create_path, json={**CALENDAR.create_body(), "access_grants": grants}
        )
    assert created.status_code == 200, created.text
    foreign_calendar = created.json()["id"]
    with mover.client() as client:
        own_calendar = client.post(CALENDAR.create_path, json=CALENDAR.create_body()).json()["id"]
        event_id = _event_in(mover, own_calendar)["event_id"]

        moved = update_event(client, event_id, calendar_id=foreign_calendar)
        stored = client.get(f"/api/v1/calendars/events/{event_id}")

    assert moved.status_code == REFUSED, moved.text
    assert stored.json()["calendar_id"] == own_calendar
    with other.client() as client:
        assert _owner_view(client, {"id": foreign_calendar})["events"] == []


def test_an_event_moves_into_a_calendar_the_mover_may_write(make_user):
    mover, other = make_user(), make_user()
    grants = [grant("user", mover.id, "read"), grant("user", mover.id, "write")]
    with other.client() as client:
        created = client.post(
            CALENDAR.create_path, json={**CALENDAR.create_body(), "access_grants": grants}
        )
    assert created.status_code == 200, created.text
    shared_calendar = created.json()["id"]
    with mover.client() as client:
        own_calendar = client.post(CALENDAR.create_path, json=CALENDAR.create_body()).json()["id"]
        event_id = _event_in(mover, own_calendar)["event_id"]

        moved = update_event(client, event_id, calendar_id=shared_calendar)

    assert moved.status_code == OK, moved.text
    assert moved.json()["calendar_id"] == shared_calendar
