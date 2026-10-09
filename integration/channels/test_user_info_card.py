"""Journey: the info a viewer gets for another person carries their timezone and last activity.

The card in a channel reads GET /api/v1/users/{id}/info. The web client stores the account's
timezone with POST /api/v1/auths/update/timezone when it loads; the info another account gets
carries that timezone as set (null while none was stored) and a `last_active_at` in epoch seconds
that is recent for someone who just used the API.

Discriminates: passes on dev 538f9c909; in a backend copy, removing `last_active_at` and
`timezone` from UserInfoResponse turns every test in this module red.
"""

from __future__ import annotations

import time

import pytest

from harness.calendar_api import set_timezone

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

RECENT_SECONDS = 300


def _info_seen_by(viewer, person) -> dict:
    with viewer.client() as client:
        info = client.get(f"/api/v1/users/{person.id}/info")
    assert info.status_code == 200, info.text
    return info.json()


def test_the_info_a_viewer_gets_carries_the_timezone_the_person_set(make_user):
    person, viewer = make_user(), make_user()
    with person.client() as client:
        set_timezone(client, "Asia/Tokyo")

    assert _info_seen_by(viewer, person)["timezone"] == "Asia/Tokyo"


def test_the_timezone_follows_the_latest_one_the_person_set(make_user):
    person, viewer = make_user(), make_user()
    with person.client() as client:
        set_timezone(client, "Asia/Tokyo")
        set_timezone(client, "America/Sao_Paulo")

    assert _info_seen_by(viewer, person)["timezone"] == "America/Sao_Paulo"


def test_the_timezone_is_empty_while_the_person_never_set_one(make_user):
    person, viewer = make_user(), make_user()

    assert _info_seen_by(viewer, person)["timezone"] is None


def test_the_info_a_viewer_gets_carries_a_recent_last_active_time(make_user):
    person, viewer = make_user(), make_user()
    with person.client() as client:
        client.get("/api/v1/auths/").raise_for_status()
    # the activity stamp is written in the background after the request
    deadline = time.time() + 10
    last_active_at = None
    while time.time() < deadline and not last_active_at:
        last_active_at = _info_seen_by(viewer, person)["last_active_at"]
        time.sleep(0.2)

    assert last_active_at is not None
    assert abs(time.time() - last_active_at) < RECENT_SECONDS
