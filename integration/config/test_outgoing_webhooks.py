"""Journey: the webhooks Open WebUI calls out to, a user's notification targets and the admin's.

A user's notification targets exist only while the admin has user webhooks switched on (off,
the routes answer 404 as if absent) and only for a user holding `features.webhooks`. A target's
test button delivers the test payload to the target's URL. The fetch guard refuses a target on
a loopback address, which is where the local listener lives, so delivery runs on an instance
booted with local fetching allowed, as the web loader tests do.

The admin's event webhooks are admin-only on every route. One subscribed to `user.created`
receives the event when the admin adds an account, with the event's envelope: its name, the
actor, the new account as subject and its role.

Discriminates: in a backend copy, `_check_notifications_access` skipping the switch turns the
switched-off test red (HTTP 200) and skipping the permission turns the permission test red,
`_normalize_target` skipping `validate_url` turns the loopback test red (the target is saved),
`test_target` sending an empty payload turns the delivery test red, and `add_user` publishing
`user.updated` in place of `user.created` turns the event delivery test red.
"""

from __future__ import annotations

import time

import pytest

from harness.actors import admin_of, create_user
from harness.listener import json_answer
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PERMISSIONS = "/api/v1/users/default/permissions"
TARGETS = "/api/v1/notifications/targets"
EVENT_WEBHOOKS = "/api/events/webhooks"
REFUSED = {401, 403}


def _wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _save(client, path: str, **changes) -> None:
    current = client.get(path)
    current.raise_for_status()
    saved = client.post(path, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def _allow_user_webhooks(client, switched_on: bool = True, permitted: bool = True) -> None:
    _save(client, ADMIN_CONFIG, ENABLE_USER_WEBHOOKS=switched_on)
    features = client.get(PERMISSIONS).json()["features"]
    _save(client, PERMISSIONS, features={**features, "webhooks": permitted})


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def user_webhooks(admin, preserve):
    """`user_webhooks(switched_on, permitted)` sets user webhooks on the shared instance."""
    preserve("admin_config", "permissions")

    def configure(switched_on: bool = True, permitted: bool = True) -> None:
        with admin.client() as client:
            _allow_user_webhooks(client, switched_on, permitted)

    return configure


# a user's notification targets


@pytest.mark.parametrize("role", ["user", "admin"])
def test_switched_off_user_webhooks_answer_404(user_webhooks, make_user, role):
    user_webhooks(switched_on=False)

    with make_user(role=role).client() as client:
        listed = client.get(TARGETS)

    assert listed.status_code == 404, listed.text


def test_a_user_without_webhooks_is_refused_and_an_admin_is_not(user_webhooks, make_user):
    user_webhooks(permitted=False)

    with make_user().client() as client:
        refused = client.get(TARGETS)
    with make_user(role="admin").client() as client:
        listed = client.get(TARGETS)

    assert refused.status_code == 403, refused.text
    assert listed.status_code == 200 and listed.json() == {"targets": []}, listed.text


def test_a_target_on_a_loopback_address_is_refused(user_webhooks, listener, make_user):
    user_webhooks()

    with make_user().client() as client:
        created = client.post(TARGETS, json={"config": {"url": f"{listener.base_url}/hook"}})
        listed = client.get(TARGETS)

    assert created.status_code == 400, created.text
    assert listed.json() == {"targets": []}
    assert listener.received == []


def test_the_test_button_delivers_the_test_payload(fetching_instance, preserve, listener):
    preserve("admin_config", "permissions", on=fetching_instance)
    with admin_of(fetching_instance).client() as client:
        _allow_user_webhooks(client)
    listener.route("POST", "/hook", json_answer({}))
    account = create_user(fetching_instance)

    with account.client() as client:
        created = client.post(
            TARGETS, json={"id": "ops", "config": {"url": f"{listener.base_url}/hook"}}
        )
        assert created.status_code == 200, created.text
        tested = client.post(f"{TARGETS}/ops/test")

    assert created.json()["is_default"] is True
    assert "url" not in created.json()["config"], "the target's URL was handed back unmasked"
    assert tested.status_code == 200 and tested.json() == {"ok": True}, tested.text
    [delivered] = listener.requests_to("/hook")
    assert delivered.json() == {"action": "test", "user_id": account.id}


# the admin's event webhooks


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", EVENT_WEBHOOKS),
        ("POST", EVENT_WEBHOOKS),
        ("PUT", f"{EVENT_WEBHOOKS}/any"),
        ("DELETE", f"{EVENT_WEBHOOKS}/any"),
    ],
)
def test_a_user_is_refused_every_event_webhook_route(make_user, method, path):
    with make_user().client() as client:
        refused = client.request(method, path, json={"url": "https://example.com/hook"})

    assert refused.status_code in REFUSED, refused.text


def test_an_admin_creates_updates_and_deletes_an_event_webhook(fetching_instance, listener):
    with admin_of(fetching_instance).client() as client:
        created = client.post(
            EVENT_WEBHOOKS,
            json={"name": "Audit", "url": f"{listener.base_url}/events", "events": ["user.*"]},
        )
        assert created.status_code == 200, created.text
        webhook_id = created.json()["id"]
        updated = client.put(f"{EVENT_WEBHOOKS}/{webhook_id}", json={"enabled": False})
        listed = client.get(EVENT_WEBHOOKS).json()
        deleted = client.delete(f"{EVENT_WEBHOOKS}/{webhook_id}")
        remaining = client.get(EVENT_WEBHOOKS).json()

    assert updated.status_code == 200, updated.text
    [stored] = [webhook for webhook in listed if webhook["id"] == webhook_id]
    assert (stored["name"], stored["events"], stored["enabled"]) == ("Audit", ["user.*"], False)
    assert deleted.status_code == 200, deleted.text
    assert webhook_id not in [webhook["id"] for webhook in remaining]


def test_a_new_account_is_delivered_to_an_event_webhook(fetching_instance, listener):
    listener.route("POST", "/events", json_answer({}))
    admin = admin_of(fetching_instance)
    with admin.client() as client:
        created = client.post(
            EVENT_WEBHOOKS,
            json={"url": f"{listener.base_url}/events", "events": ["user.created"]},
        )
        assert created.status_code == 200, created.text
        try:
            account = create_user(fetching_instance)
            assert _wait_for(lambda: listener.requests_to("/events")), "no event was delivered"
        finally:
            client.delete(f"{EVENT_WEBHOOKS}/{created.json()['id']}")

    [delivered] = listener.requests_to("/events")
    event = delivered.json()
    assert (event["event"], event["resource"], event["operation"]) == (
        "user.created",
        "user",
        "created",
    )
    assert event["subject"] == {"type": "user", "id": account.id}
    assert event["actor"]["id"] == admin.id and event["actor"]["role"] == "admin"
    assert event["source"] == "admin"
    assert event["data"] == {"role": "user"}
