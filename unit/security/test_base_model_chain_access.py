"""Regression: a shared workspace model must not become a ladder to a base
model the caller could not use directly.

open-webui 0.11.0 fix `fe4b31942` (#26905, issue #26900): `has_base_model_access`
walked the `base_model_id` chain and, on reaching a model with no row in the
`model` table (a raw provider model), did `break`, treating "unregistered" as
"unrestricted". Everywhere else unregistered models are admin-only:
`get_filtered_models` hides them from non-admins and `check_model_access` rejects
them outright. So an admin could share a thin preset whose base was a restricted
raw provider model, and any non-admin with read access to the preset reached the
underlying model through it. The fix returns `user_role == 'admin'` at that hop.

Discriminates: passes on v0.11.0, fails on v0.10.2 (`break` → True for everyone).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

UNREGISTERED = "raw-provider-model"


def _model(id: str, user_id: str = "owner", base_model_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=id, user_id=user_id, base_model_id=base_model_id)


def _registry(*models: SimpleNamespace) -> AsyncMock:
    """Stand in for `Models.get_model_by_id`; anything not registered is None."""
    by_id = {m.id: m for m in models}
    return AsyncMock(side_effect=lambda id, db=None: by_id.get(id))


def _patch_registry(get_model_by_id: AsyncMock, has_access: AsyncMock):
    """`has_base_model_access` imports these inside its body, so they must be
    patched on their defining modules rather than on access_control."""
    import open_webui.models.access_grants as ag
    import open_webui.models.models as mm

    return (
        patch.object(mm.Models, "get_model_by_id", get_model_by_id),
        patch.object(ag.AccessGrants, "has_access", has_access),
    )


@pytest.mark.asyncio
async def test_unregistered_base_model_denied_for_non_admin(access_control_module):
    """The bug: preset the user owns, base model not in the `model` table."""
    preset = _model("shared-preset", user_id="alice", base_model_id=UNREGISTERED)
    registry, grants = _patch_registry(_registry(preset), AsyncMock(return_value=False))
    with registry, grants:
        allowed = await access_control_module.has_base_model_access(
            "alice", preset, user_role="user"
        )
    assert allowed is False, (
        "a non-admin reached an unregistered (admin-only) base model through a "
        "preset they can read; #26900 is back"
    )


@pytest.mark.asyncio
async def test_unregistered_base_model_allowed_for_admin(access_control_module):
    """Admins may use unregistered models directly, so the chain must not
    block them either, otherwise the fix breaks admin presets."""
    preset = _model("shared-preset", user_id="alice", base_model_id=UNREGISTERED)
    registry, grants = _patch_registry(_registry(preset), AsyncMock(return_value=False))
    with registry, grants:
        allowed = await access_control_module.has_base_model_access(
            "alice", preset, user_role="admin"
        )
    assert allowed is True


# The three checks below are role-independent, so they omit `user_role` and hold
# on both sides of the fix; they exist to prove the fix didn't over-deny.


@pytest.mark.asyncio
async def test_registered_base_model_owned_by_caller_is_allowed(access_control_module):
    """Sanity (positive path): every hop registered and owned → allowed."""
    base = _model("base", user_id="alice")
    preset = _model("shared-preset", user_id="alice", base_model_id="base")
    registry, grants = _patch_registry(_registry(preset, base), AsyncMock(return_value=False))
    with registry, grants:
        allowed = await access_control_module.has_base_model_access("alice", preset)
    assert allowed is True


@pytest.mark.asyncio
async def test_registered_base_model_without_grant_is_denied(access_control_module):
    """A registered base owned by someone else with no read grant stays denied."""
    base = _model("base", user_id="bob")
    preset = _model("shared-preset", user_id="bob", base_model_id="base")
    registry, grants = _patch_registry(_registry(preset, base), AsyncMock(return_value=False))
    with registry, grants:
        allowed = await access_control_module.has_base_model_access("alice", preset)
    assert allowed is False


@pytest.mark.asyncio
async def test_chain_cycle_terminates(access_control_module):
    """A model whose base points back at itself must not spin forever."""
    a = _model("a", user_id="alice", base_model_id="b")
    b = _model("b", user_id="alice", base_model_id="a")
    registry, grants = _patch_registry(_registry(a, b), AsyncMock(return_value=False))
    with registry, grants:
        allowed = await access_control_module.has_base_model_access("alice", a)
    assert allowed is True


@pytest.mark.asyncio
async def test_check_model_access_denies_the_chain_end_to_end(access_control_module):
    """The same denial through the real entry point callers use, not just the
    walk in isolation. This is the assertion that fails on v0.10.2, where the
    walk returned True and no 403 was raised."""
    mod = access_control_module
    preset = _model("shared-preset", user_id="alice", base_model_id=UNREGISTERED)
    user = SimpleNamespace(id="alice", role="user")

    from fastapi import HTTPException

    registry, grants = _patch_registry(_registry(preset), AsyncMock(return_value=False))
    with (
        registry,
        grants,
        patch.object(mod.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await mod.check_model_access(user, preset)
    assert excinfo.value.status_code == 403
