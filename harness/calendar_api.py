"""Calendar and timezone calls the way the calendar page makes them, times in epoch nanoseconds."""

from __future__ import annotations

import datetime as dt

import httpx

SECOND_NS = 1_000_000_000
MINUTE_NS = 60 * SECOND_NS
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS


def to_ns(moment: dt.datetime) -> int:
    return int(moment.timestamp()) * SECOND_NS


def to_utc(ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ns / SECOND_NS, dt.timezone.utc)


def set_timezone(client: httpx.Client, timezone: str) -> None:
    updated = client.post("/api/v1/auths/update/timezone", json={"timezone": timezone})
    assert updated.status_code == 200, updated.text


def default_calendar_id(client: httpx.Client) -> str:
    """The account's own calendar, created on first listing."""
    calendars = client.get("/api/v1/calendars/")
    assert calendars.status_code == 200, calendars.text
    return next(calendar["id"] for calendar in calendars.json() if calendar["is_default"])


def create_event(client: httpx.Client, calendar_id: str, **fields) -> httpx.Response:
    return client.post(
        "/api/v1/calendars/events/create",
        json={"calendar_id": calendar_id, "title": "Standup", **fields},
    )


def update_event(client: httpx.Client, event_id: str, **fields) -> httpx.Response:
    return client.post(f"/api/v1/calendars/events/{event_id}/update", json=fields)


def events_between(client: httpx.Client, start_ns: int, end_ns: int) -> list[dict]:
    """The events (recurring ones expanded) the calendar page loads for that range."""
    listed = client.get(
        "/api/v1/calendars/events",
        params={"start": to_utc(start_ns).isoformat(), "end": to_utc(end_ns).isoformat()},
    )
    assert listed.status_code == 200, listed.text
    return listed.json()
