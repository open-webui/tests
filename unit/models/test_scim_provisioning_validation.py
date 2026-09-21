"""Regression: SCIM provisioning accepted nonsense and touched accounts for nothing.

open-webui 0.11.4 commit `ad9da9816` (a `refac`): the SCIM PATCH handler
dispatched on the operation's path with no else branch, so an attribute Open
WebUI does not support (any unknown path) and an operation with an unsupported
`op` both passed in silence as a successful no-op, and a wrong value type (a
string for `active`, a number for `userName`) was written through to the user
row. A sync that rewrote the same values also went through the full update:
`patch_user` re-updated every field the directory sent even when identical, and
`update_user_scim_by_id` stamped `updated_at` on every sync call, marking the
account as touched. The fix validates every operation up front and returns a
SCIM 400 error for an unsupported op, an unsupported path, or a wrong value
type, prunes fields whose value equals the stored one before persisting, and
`update_user_scim_by_id` only writes when the scim payload actually changed.

Discriminates: passes on 344ea5306 (0.11.4), fails on ad9da9816^ (an unsupported
path or op is a silent success, a wrong value type reaches the update, and a
no-op sync re-stamps the row).
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def scim_router(owui_module):
    return owui_module("open_webui.routers.scim")


@pytest.fixture(scope="module")
def users_model(owui_module):
    return owui_module("open_webui.models.users")


def _request():
    return SimpleNamespace(base_url="http://test/")


def _patch(scim_router, *operations):
    return scim_router.SCIMPatchRequest(Operations=list(operations))


def _op(scim_router, op: str, path: str | None, value=None):
    return scim_router.SCIMPatchOperation(op=op, path=path, value=value)


def _user(users_model, **overrides):
    base = dict(
        id="u-1",
        email="old@example.com",
        name="Old Name",
        role="user",
        active=True,
        profile_image_url="/user.png",
        scim={"testprov": {"external_id": "abc"}},
        created_at=1,
        updated_at=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


async def _patch_user(scim_router, user, request_data):
    """Drive the real handler with the user row stubbed at the I/O boundary."""
    updates = []

    async def record_update(user_id, data, db=None):
        updates.append(data)
        updated = _user_from(scim_router, user, data)
        return updated

    with (
        patch.object(scim_router.Users, "get_scim_user_by_id", AsyncMock(return_value=user)),
        patch.object(scim_router.Users, "update_user_by_id", side_effect=record_update),
        patch.object(scim_router, "publish_event", AsyncMock()),
    ):
        result = await scim_router.patch_user(
            user.id, request_data["request"], request_data["patch"], _=True, db=None
        )
    return result, updates


def _user_from(scim_router, user, data):
    fields = dict(vars(user))
    fields.update(data)
    return SimpleNamespace(**fields)


def _run(request, patch):
    return {"request": request, "patch": patch}


# --- narrow: unsupported requests come back as errors -----------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "path": "nickName", "value": "bob"},
        {"op": "replace", "path": "phoneNumbers[primary eq true].value", "value": "123"},
        {"op": "delete", "path": "displayName"},
    ],
)
async def test_an_unsupported_attribute_is_rejected(scim_router, users_model, operation):
    """Pre-fix an unknown path fell through every branch and the request reported success."""
    response, updates = await _patch_user(
        scim_router,
        _user(users_model),
        _run(
            _request(),
            _patch(scim_router, _op(scim_router, **operation)),
        ),
    )

    assert response.status_code == 400, (
        f"a PATCH for the unsupported attribute {operation['path']!r} was accepted in "
        "silence, so the directory believed a change Open WebUI never made (ad9da9816)"
    )
    assert updates == []


@pytest.mark.asyncio
async def test_an_unsupported_operation_is_rejected(scim_router, users_model):
    """Pre-fix any op other than 'replace' fell through the if-chain as a no-op success."""
    response, updates = await _patch_user(
        scim_router,
        _user(users_model),
        _run(
            _request(),
            _patch(scim_router, _op(scim_router, "delete", "displayName", None)),
        ),
    )

    assert response.status_code == 400
    assert updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,value",
    [
        ("active", "yes"),
        ("userName", 123),
        ("displayName", ["New"]),
        ("name.formatted", 5),
    ],
)
async def test_a_wrong_value_type_is_rejected(scim_router, users_model, path, value):
    """Pre-fix the value was written through with no type check, corrupting the row."""
    response, updates = await _patch_user(
        scim_router,
        _user(users_model),
        _run(_request(), _patch(scim_router, _op(scim_router, "replace", path, value))),
    )

    assert response.status_code == 400, (
        f"a PATCH writing {value!r} into {path!r} was accepted, so the wrong kind of "
        "value reached the user row (ad9da9816)"
    )
    assert updates == []


# --- narrow: a no-op sync does not touch the account ------------------------------


@pytest.mark.asyncio
async def test_a_sync_that_changes_nothing_neither_updates_nor_publishes(
    scim_router, users_model
):
    """Pre-fix the identical displayName went through update_user_by_id with a fresh
    updated_at, and the USER_UPDATED event fired for a change that changed nothing."""
    user = _user(users_model)
    published = []

    async def record_update(user_id, data, db=None):
        published.append(("update", data))
        return user

    with (
        patch.object(scim_router.Users, "get_scim_user_by_id", AsyncMock(return_value=user)),
        patch.object(scim_router.Users, "update_user_by_id", side_effect=record_update),
        patch.object(
            scim_router,
            "publish_event",
            AsyncMock(side_effect=lambda *a, **k: published.append(("event", k.get("data")))),
        ),
    ):
        await scim_router.patch_user(
            "u-1",
            _request(),
            _patch(scim_router, _op(scim_router, "replace", "displayName", "Old Name")),
            _=True,
            db=None,
        )

    assert published == [], (
        "a sync that rewrote the same values updated the row and published USER_UPDATED, "
        "marking the account as touched for nothing (ad9da9816)"
    )


@pytest.mark.asyncio
async def test_a_noop_externalid_sync_leaves_the_row_untouched(scim_router, users_model):
    """Same contract through the externalId path, where the scim payload equals the stored one."""
    user = _user(users_model)
    updates = []

    async def record_update(user_id, data, db=None):
        updates.append(data)
        return user

    with (
        patch.object(scim_router.Users, "get_scim_user_by_id", AsyncMock(return_value=user)),
        patch.object(scim_router.Users, "update_user_by_id", side_effect=record_update),
        patch.object(scim_router.Users, "update_user_scim_by_id", side_effect=record_update),
        patch.object(scim_router, "publish_event", AsyncMock()),
        patch.object(scim_router, "get_scim_provider", lambda: "testprov"),
    ):
        await scim_router.patch_user(
            "u-1",
            _request(),
            _patch(scim_router, _op(scim_router, "replace", "externalId", "abc")),
            _=True,
            db=None,
        )

    assert updates == [], (
        "a directory sync that re-sent the same externalId rewrote the scim payload, "
        "re-stamping updated_at on every sync (ad9da9816)"
    )


# --- broad: a changed value still goes through, with the timestamp ---------------


@pytest.mark.asyncio
async def test_a_changed_field_is_written_and_published(scim_router, users_model):
    user = _user(users_model)
    updates = []
    events = []

    async def record_update(user_id, data, db=None):
        updates.append(data)
        return _user_from(scim_router, user, data)

    with (
        patch.object(scim_router.Users, "get_scim_user_by_id", AsyncMock(return_value=user)),
        patch.object(scim_router.Users, "update_user_by_id", side_effect=record_update),
        patch.object(
            scim_router,
            "publish_event",
            AsyncMock(side_effect=lambda *a, **k: events.append(k.get("data"))),
        ),
    ):
        await scim_router.patch_user(
            "u-1",
            _request(),
            _patch(
                scim_router,
                _op(scim_router, "replace", "displayName", "New Name"),
                _op(scim_router, "replace", "userName", "new@example.com"),
            ),
            _=True,
            db=None,
        )

    assert len(updates) == 1, "a changed value must still be persisted once"
    assert updates[0]["name"] == "New Name"
    assert updates[0]["email"] == "new@example.com"
    assert "updated_at" in updates[0], "a real change must stamp updated_at"
    assert any("name" in (event or {}).get("updated_fields", []) for event in events)


# --- nearby: the model layer's half of the no-op guard ----------------------------


@pytest.mark.asyncio
async def test_update_user_scim_by_id_skips_the_write_for_an_identical_payload(
    owui_module,
):
    """`ad9da9816` also fixed `UsersTable.update_user_scim_by_id`: pre-fix it assigned the
    scim dict and re-stamped updated_at even when the payload was identical."""
    users_module = owui_module("open_webui.models.users")
    db_module = owui_module("open_webui.internal.db")
    owui_module("open_webui.config")  # migrated scratch database

    user_id = f"scim-{uuid.uuid4().hex[:16]}"
    stored_at = int(time.time()) - 100
    async with db_module.get_async_db() as session:
        session.add(
            users_module.User(
                id=user_id,
                email=f"{user_id}@example.test",
                name="Directory User",
                role="user",
                profile_image_url="/user.png",
                oauth={"oidc": {"sub": "x"}},
                scim={"testprov": {"external_id": "abc"}},
                created_at=stored_at,
                updated_at=stored_at,
                last_active_at=stored_at,
            )
        )
        await session.commit()

    synced = await users_module.Users.update_user_scim_by_id(user_id, "testprov", "abc")
    assert synced.updated_at == stored_at, (
        "a sync that re-sent the same externalId re-stamped the account as touched (ad9da9816)"
    )

    changed = await users_module.Users.update_user_scim_by_id(user_id, "testprov", "new-id")
    assert changed.updated_at > stored_at, "a changed externalId must still stamp the row"
