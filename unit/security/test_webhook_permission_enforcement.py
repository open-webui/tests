"""Regression: the webhook permission was advisory, so a denied user could still
save a webhook destination.

open-webui 0.11.0 fix (PR #27297): `update_user_settings_by_session_user` in
`routers/users.py` stripped `ui.toolServers` when `features.direct_tool_servers`
was denied, but nothing looked at `features.webhooks`. The frontend merely hid
the notification-webhook field, so a direct POST to `/api/v1/users/user/settings/update`
persisted `ui.notifications.webhook_url` for a user the admin had explicitly
denied webhooks, and the notification path then delivered to that URL. The fix
drops the `notifications` block and the `webhook_url` key on save unless the
caller is an admin or holds `features.webhooks`.

Discriminates: passes on v0.11.0, fails on v0.10.2 (the denied webhook URL lands
in the stored settings verbatim).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

WEBHOOK_URL = "https://attacker.example/hook"
TOOL_SERVERS = [{"url": "https://attacker.example/openapi.json", "key": "k"}]


@pytest.fixture(scope="session")
def users_router(owui_module):
    return owui_module("open_webui.routers.users")


def _permissions(**feature_overrides) -> dict:
    features = {"webhooks": True, "direct_tool_servers": True, **feature_overrides}
    return {"features": features, "settings": {"interface": True}}


async def _save_settings(
    users_router,
    settings: dict,
    permissions: dict,
    role: str = "user",
    stored: dict | None = None,
) -> dict:
    """Drive the real endpoint and return the settings that actually landed."""
    persisted = dict(stored or {})

    async def update_user_settings_by_id(id, updated, db=None):
        persisted.update(updated)
        return SimpleNamespace(id=id, settings=persisted)

    config_get = AsyncMock(
        side_effect=lambda key, default=None: permissions if key == "user.permissions" else default
    )
    with (
        patch.object(users_router.Config, "get", config_get),
        patch.object(users_router.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
        patch.object(users_router.Users, "update_user_settings_by_id", update_user_settings_by_id),
        patch.object(users_router, "publish_event", AsyncMock()),
    ):
        await users_router.update_user_settings_by_session_user(
            request=None,
            form_data=users_router.UserSettings(**settings),
            user=SimpleNamespace(id="alice", role=role),
            db=None,
        )
    return persisted


# --- narrow: the denied webhook must not survive the save ------------------


@pytest.mark.asyncio
async def test_denied_user_cannot_persist_a_webhook_url(users_router):
    persisted = await _save_settings(
        users_router,
        {"ui": {"notifications": {"webhook_url": WEBHOOK_URL}}},
        _permissions(webhooks=False),
    )
    assert persisted["ui"]["notifications"].get("webhook_url") is None, (
        "a user denied the webhooks permission still saved a notification webhook, so "
        "the server delivers their notifications to an address the admin forbade (#27297)"
    )


@pytest.mark.asyncio
async def test_denied_user_cannot_persist_a_top_level_notifications_block(users_router):
    """`UserSettings` allows extra keys, so the block can arrive outside `ui` too."""
    persisted = await _save_settings(
        users_router,
        {"ui": {}, "notifications": {"webhook_url": WEBHOOK_URL}},
        _permissions(webhooks=False),
    )
    assert "notifications" not in persisted, (
        "the notifications block bypassed the webhooks permission by being sent at the "
        "top level of the settings payload instead of under `ui` (#27297)"
    )


# --- broad: every permission the UI reflects must be checked on save -------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "settings", "gated_path"),
    [
        (
            "webhooks",
            {"ui": {"notifications": {"webhook_url": WEBHOOK_URL}}},
            ("ui", "notifications", "webhook_url"),
        ),
        ("webhooks", {"ui": {}, "notifications": {"webhook_url": WEBHOOK_URL}}, ("notifications",)),
        ("direct_tool_servers", {"ui": {"toolServers": TOOL_SERVERS}}, ("ui", "toolServers")),
    ],
)
async def test_every_permission_gated_settings_field_is_stripped_on_save(
    users_router, feature, settings, gated_path
):
    persisted = await _save_settings(users_router, settings, _permissions(**{feature: False}))

    node = persisted
    for key in gated_path[:-1]:
        node = node.get(key, {})
    assert gated_path[-1] not in node, (
        f"settings field {'.'.join(gated_path)} is gated by features.{feature} in the "
        "interface but not enforced on save, so a direct API call sets it anyway (#27297)"
    )


# --- nearby: behaviour that must hold on both refs -------------------------


@pytest.mark.asyncio
async def test_permitted_user_can_save_a_webhook(users_router):
    persisted = await _save_settings(
        users_router,
        {"ui": {"notifications": {"webhook_url": WEBHOOK_URL}}},
        _permissions(webhooks=True),
    )
    assert persisted["ui"]["notifications"]["webhook_url"] == WEBHOOK_URL


@pytest.mark.asyncio
async def test_admin_can_save_a_webhook_regardless_of_the_permission(users_router):
    persisted = await _save_settings(
        users_router,
        {"ui": {"notifications": {"webhook_url": WEBHOOK_URL}}},
        _permissions(webhooks=False),
        role="admin",
    )
    assert persisted["ui"]["notifications"]["webhook_url"] == WEBHOOK_URL


@pytest.mark.asyncio
@pytest.mark.parametrize("webhooks_allowed", [True, False])
async def test_clearing_an_existing_webhook_always_works(users_router, webhooks_allowed):
    """Removing a webhook is not using one, so it must not be blocked."""
    persisted = await _save_settings(
        users_router,
        {"ui": {"notifications": {"webhook_url": ""}}},
        _permissions(webhooks=webhooks_allowed),
        stored={"ui": {"notifications": {"webhook_url": WEBHOOK_URL}}},
    )
    assert not persisted["ui"]["notifications"].get("webhook_url")


@pytest.mark.asyncio
async def test_unrelated_settings_survive_a_denied_webhook(users_router):
    """The strip must be surgical, not a rejection of the whole payload."""
    persisted = await _save_settings(
        users_router,
        {"ui": {"theme": "dark", "notifications": {"webhook_url": WEBHOOK_URL, "enabled": True}}},
        _permissions(webhooks=False),
    )
    assert persisted["ui"]["theme"] == "dark"
    assert persisted["ui"]["notifications"]["enabled"] is True


@pytest.mark.asyncio
async def test_settings_without_a_notifications_block_are_untouched(users_router):
    persisted = await _save_settings(
        users_router, {"ui": {"theme": "dark"}}, _permissions(webhooks=False)
    )
    assert persisted["ui"] == {"theme": "dark"}


@pytest.mark.asyncio
async def test_interface_permission_still_rejects_the_whole_save(users_router):
    from fastapi import HTTPException

    permissions = _permissions()
    permissions["settings"]["interface"] = False
    with pytest.raises(HTTPException) as excinfo:
        await _save_settings(users_router, {"ui": {"theme": "dark"}}, permissions)
    assert excinfo.value.status_code == 403
