"""Journey: an admin hides single controls from users, and every signed-in account reads the map.

Admin Settings > Interface > Visible controls saves a map of control ids to false through
`POST /api/v1/configs/interface/visibility`; `/api/config` hands the saved map to every signed-in
account under `ui.visibility`, and the web client hides those controls from everyone but admins.
A later save replaces the whole map, a user cannot save one, a value other than true or false is
refused and leaves the map alone, a signed-out visitor is not given the map, and `UI_VISIBILITY`
from the environment is the map a fresh instance starts from until an admin saves another.

Discriminates: passes on dev 948432b43; in a backend copy, with the save route open to any
verified user the refusal test fails (the user's map is saved), with the form taking `bool` in
place of `StrictBool` the non-boolean test fails ("false" is saved as false), and with
`ui.visibility` left out of `/api/config` the read back and environment tests fail.
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness.actors import create_user

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

VISIBILITY = "/api/v1/configs/interface/visibility"
HIDDEN = {"chat.attach_webpage": False, "chat.archive": False}
FROM_ENV = {"chat.fork": False, "settings.widescreen": False}


def visibility_seen_by(client: httpx.Client) -> dict | None:
    config = client.get("/api/config")
    assert config.status_code == 200, config.text
    return config.json().get("ui", {}).get("visibility")


@pytest.fixture
def admin_client(admin):
    """The admin's client; the instance's own visibility map is put back afterwards."""
    with admin.client() as client:
        before = visibility_seen_by(client) or {}
        yield client
        client.post(VISIBILITY, json={"visibility": before}).raise_for_status()


def test_a_saved_map_reaches_admins_and_users_through_the_config(admin_client, make_user):
    saved = admin_client.post(VISIBILITY, json={"visibility": HIDDEN})
    assert saved.status_code == 200, saved.text
    assert saved.json() == {"visibility": HIDDEN}

    assert visibility_seen_by(admin_client) == HIDDEN
    with make_user().client() as client:
        assert visibility_seen_by(client) == HIDDEN, "a user is not given the saved map"


def test_a_later_save_replaces_the_whole_map(admin_client):
    admin_client.post(VISIBILITY, json={"visibility": HIDDEN}).raise_for_status()

    saved = admin_client.post(VISIBILITY, json={"visibility": {"chat.tags": False}})
    assert saved.status_code == 200, saved.text

    assert visibility_seen_by(admin_client) == {"chat.tags": False}


def test_a_user_cannot_save_a_map(admin_client, make_user):
    admin_client.post(VISIBILITY, json={"visibility": HIDDEN}).raise_for_status()

    with make_user().client() as client:
        refused = client.post(VISIBILITY, json={"visibility": {}})

    assert refused.status_code in (401, 403), refused.text
    assert visibility_seen_by(admin_client) == HIDDEN, "a user's save changed the map"


@pytest.mark.parametrize("value", ["false", 0, None, "hidden"])
def test_a_value_other_than_true_or_false_is_refused(admin_client, value):
    admin_client.post(VISIBILITY, json={"visibility": HIDDEN}).raise_for_status()

    refused = admin_client.post(VISIBILITY, json={"visibility": {"chat.fork": value}})

    assert refused.status_code == 422, f"{value!r} was taken: {refused.status_code} {refused.text}"
    assert visibility_seen_by(admin_client) == HIDDEN, f"saving {value!r} changed the map"


def test_a_signed_out_visitor_is_not_given_the_map(admin_client, instance):
    admin_client.post(VISIBILITY, json={"visibility": HIDDEN}).raise_for_status()

    with httpx.Client(base_url=instance.base_url, timeout=30.0) as anonymous:
        assert visibility_seen_by(anonymous) is None


@pytest.mark.slow
def test_the_environment_map_is_where_an_instance_starts(instance_with):
    launched = instance_with({"UI_VISIBILITY": json.dumps(FROM_ENV)})

    with create_user(launched).client() as client:
        assert visibility_seen_by(client) == FROM_ENV, "UI_VISIBILITY did not reach users"

    with launched.client() as admin_client:
        saved = admin_client.post(VISIBILITY, json={"visibility": HIDDEN})
        assert saved.status_code == 200, saved.text
        try:
            assert visibility_seen_by(admin_client) == HIDDEN, "the saved map lost to the env"
        finally:
            admin_client.post(VISIBILITY, json={"visibility": FROM_ENV}).raise_for_status()
