"""Regression: a user the admin denied webhooks could still save a webhook destination.

open-webui 0.11.0, fix `af629177f` (the maintainer's version of the closed PR #27297):
`POST /api/v1/users/user/settings/update` dropped `ui.toolServers` without
`features.direct_tool_servers` but never looked at `features.webhooks`. The interface only hid the
field, so a direct call stored `ui.notifications.webhook_url`, or a top-level `notifications`
block, for a denied user. The fix drops both on save unless the caller is an admin or holds the
permission. Here the admin denies the features through the default permissions and each save is
read back through `GET /api/v1/users/user/settings`, what the settings page loads.

Twin of unit/security/test_webhook_permission_enforcement.py.

Discriminates: passes on bbfa876af, fails with af629177f reverted (both webhook saves come back
verbatim and the combined save keeps them); the other tests pass on both.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

WEBHOOK_URL = "https://attacker.example/hook"
TOOL_SERVERS = [{"url": "https://attacker.example/openapi.json", "key": "k"}]

DENIED_WEBHOOK_SAVES = {
    "ui.notifications.webhook_url": {
        "ui": {"notifications": {"webhook_url": WEBHOOK_URL, "enabled": True}}
    },
    "notifications": {"ui": {}, "notifications": {"webhook_url": WEBHOOK_URL}},
}


def _set_permissions(admin, section: str, **flags: bool) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions[section].update(flags)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def _save_and_reload(actor, settings: dict) -> dict:
    """Save as the settings page does, then read what a reload shows."""
    with actor.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json=settings)
        assert saved.status_code == 200, saved.text
        stored = client.get("/api/v1/users/user/settings")
    assert stored.status_code == 200, stored.text
    return stored.json() or {}


@pytest.fixture
def gated_features_denied(admin, preserve):
    preserve("permissions")
    _set_permissions(admin, "features", webhooks=False, direct_tool_servers=False)


# narrow: both shapes of a webhook save from a denied user


@pytest.mark.parametrize("field", sorted(DENIED_WEBHOOK_SAVES))
def test_denied_user_cannot_store_a_webhook(gated_features_denied, make_user, field):
    stored = _save_and_reload(make_user(), DENIED_WEBHOOK_SAVES[field])

    assert WEBHOOK_URL not in json.dumps(stored), (
        f"a user denied features.webhooks stored a webhook through `{field}` (#27297): {stored}"
    )


# broad: every permission-gated field is enforced in one save, the rest is kept


def test_one_save_drops_every_gated_field_and_keeps_the_rest(gated_features_denied, make_user):
    stored = _save_and_reload(
        make_user(),
        {
            "ui": {
                "theme": "dark",
                "toolServers": TOOL_SERVERS,
                "notifications": {"webhook_url": WEBHOOK_URL, "enabled": True},
            },
            "notifications": {"webhook_url": WEBHOOK_URL},
        },
    )

    assert "attacker.example" not in json.dumps(stored), f"a gated field was stored: {stored}"
    assert stored["ui"]["theme"] == "dark"
    assert stored["ui"]["notifications"] == {"enabled": True}


# nearby: admins and permitted users keep the field, clearing always works


def test_admin_can_store_a_webhook_while_users_are_denied(gated_features_denied, make_user):
    stored = _save_and_reload(
        make_user(role="admin"), DENIED_WEBHOOK_SAVES["ui.notifications.webhook_url"]
    )

    assert stored["ui"]["notifications"]["webhook_url"] == WEBHOOK_URL


def test_permitted_user_can_store_a_webhook(admin, preserve, make_user):
    preserve("permissions")
    _set_permissions(admin, "features", webhooks=True)

    stored = _save_and_reload(make_user(), DENIED_WEBHOOK_SAVES["ui.notifications.webhook_url"])

    assert stored["ui"]["notifications"]["webhook_url"] == WEBHOOK_URL


def test_user_can_clear_a_webhook_after_losing_the_permission(admin, preserve, make_user):
    preserve("permissions")
    _set_permissions(admin, "features", webhooks=True)
    account = make_user()
    _save_and_reload(account, DENIED_WEBHOOK_SAVES["ui.notifications.webhook_url"])
    _set_permissions(admin, "features", webhooks=False)

    stored = _save_and_reload(account, {"ui": {"notifications": {"webhook_url": ""}}})

    assert WEBHOOK_URL not in json.dumps(stored)


def test_denied_interface_settings_are_dropped_and_the_rest_kept(admin, preserve, make_user):
    preserve("permissions")
    _set_permissions(admin, "settings", interface=False)

    stored = _save_and_reload(make_user(), {"ui": {"theme": "dark", "showUsername": True}})

    assert "showUsername" not in stored["ui"], "a denied Interface setting was stored"
    assert stored["ui"]["theme"] == "dark"
