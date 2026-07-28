"""Regression: the folder-sharing permission survives an admin save.

open-webui 0.11.0 fix `3cf3f8e18` (#27296, issue #27120): `sharing.folders` was
present in `DEFAULT_USER_PERMISSIONS` but had no field on the `SharingPermissions`
schema. The default-permissions and group-permissions endpoints validate through
that schema, and pydantic drops unknown keys, so an admin enabling folder sharing
had the flag silently discarded on save and the setting never took effect. The
fix adds `folders: bool = False`, restoring parity with the config defaults.

Discriminates: passes on v0.11.0, fails on v0.10.2 (`folders` vanishes from the
serialized permissions, and the effective `sharing.folders` falls back to the
config default False even though the admin turned it on).
"""

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

# Present in DEFAULT_USER_PERMISSIONS and in the admin UI but still missing from
# SharingPermissions at 0.11.0; the same class of bug as #27120, not yet fixed.
UNMODELLED_SHARING_KEYS = {"open_chats"}


@pytest.fixture(scope="session")
def users_router_module(owui_module):
    return owui_module("open_webui.routers.users")


def _sharing_defaults() -> dict:
    from open_webui.config import DEFAULT_USER_PERMISSIONS

    return DEFAULT_USER_PERMISSIONS["sharing"]


def _saved_permissions(users_module, **sharing_overrides) -> dict:
    """What `POST /users/default/permissions` persists for a given form."""
    from open_webui.config import DEFAULT_USER_PERMISSIONS as defaults

    form = users_module.UserPermissions(
        workspace=users_module.WorkspacePermissions(**defaults["workspace"]),
        sharing=users_module.SharingPermissions(**{**defaults["sharing"], **sharing_overrides}),
        access_grants=users_module.AccessGrantsPermissions(**defaults["access_grants"]),
        chat=users_module.ChatPermissions(**defaults["chat"]),
        features=users_module.FeaturesPermissions(**defaults["features"]),
        settings=users_module.SettingsPermissions(**defaults["settings"]),
    )
    return form.model_dump(by_alias=True)


def test_sharing_permissions_round_trips_folders(users_router_module):
    saved = _saved_permissions(users_router_module, folders=True)

    assert saved["sharing"].get("folders") is True, (
        "the admin enabled folder sharing and the setting was dropped on save, so "
        "the toggle silently reverts (#27120)"
    )


@pytest.mark.asyncio
async def test_enabled_folder_sharing_reaches_the_permission_layer(
    users_router_module, access_control_module
):
    """The consequence: what an admin saved must be what `has_permission` reads."""
    saved = _saved_permissions(users_router_module, folders=True)

    with patch.object(
        access_control_module.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
    ):
        allowed = await access_control_module.has_permission("alice", "sharing.folders", saved)

    assert allowed is True, (
        "folder sharing reads as disabled for a user even though the admin enabled "
        "it, because the saved permissions never carried the flag (#27120)"
    )


def test_every_default_sharing_toggle_has_a_model_field(users_router_module):
    """The invariant: a toggle absent from the schema is a toggle that cannot be saved."""
    modelled = set(users_router_module.SharingPermissions.model_fields)
    missing = set(_sharing_defaults()) - modelled - UNMODELLED_SHARING_KEYS

    assert missing == set(), (
        f"sharing toggles {sorted(missing)} exist in DEFAULT_USER_PERMISSIONS but have "
        "no SharingPermissions field, so admins cannot persist them (#27120)"
    )


def test_no_sharing_model_field_is_missing_from_the_defaults(users_router_module):
    """The other direction: a schema field with no default is never populated."""
    orphaned = set(users_router_module.SharingPermissions.model_fields) - set(_sharing_defaults())

    assert orphaned == set(), (
        f"SharingPermissions fields {sorted(orphaned)} have no entry in "
        "DEFAULT_USER_PERMISSIONS, so they always fall back to the schema default"
    )


@pytest.mark.parametrize("toggle", ["folders", "notes", "public_notes", "public_chats"])
@pytest.mark.parametrize("enabled", [True, False])
def test_each_sharing_toggle_round_trips_both_ways(users_router_module, toggle, enabled):
    """Neighbouring toggles must persist in both directions, not just the default."""
    if toggle not in users_router_module.SharingPermissions.model_fields:
        pytest.skip(f"{toggle} is not modelled on this ref")

    saved = _saved_permissions(users_router_module, **{toggle: enabled})

    assert saved["sharing"][toggle] is enabled


def test_unknown_stored_sharing_keys_are_tolerated(users_router_module):
    """A config written by a newer release must not break the admin page."""
    sharing = users_router_module.SharingPermissions(
        **{**_sharing_defaults(), "some_future_toggle": True}
    )

    assert "some_future_toggle" not in sharing.model_dump()


def test_missing_stored_sharing_section_falls_back_to_schema_defaults(users_router_module):
    """`get_default_user_permissions` builds from `.get('sharing', {})`."""
    sharing = users_router_module.SharingPermissions().model_dump()

    assert set(sharing) == set(users_router_module.SharingPermissions.model_fields)
