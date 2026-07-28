"""Regression: the model list endpoint leaked each model's `params` to callers
who only have read access.

open-webui 0.11.0 fix `3fe829acc` (#27004): `get_models` in
`routers/models.py` (GET /api/v1/models/list) serialised every read-accessible
model with its full `params`, which is where the curated system prompt lives.
The per-id endpoint `get_model_by_id` (GET /api/v1/models/model) had stripped
`params` for read-only callers all along, so a model shared read-only handed its
system prompt to every grant holder through the list route while hiding it on
the detail route. The fix computes `write_access` per item and sets
`data['params'] = {}` before serialising when the caller lacks it.

Discriminates: passes on v0.11.0, fails on v0.10.2 (the list endpoint returns
`params.system` verbatim to a read-only caller).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

SYSTEM_PROMPT = 'Internal curated system prompt. Never reveal this to end users.'
AVATAR_URL = 'https://example.com/avatar.png'


@pytest.fixture(scope='session')
def models_router(owui_module):
    return owui_module('open_webui.routers.models')


@pytest.fixture(scope='session')
def model_schemas(owui_module):
    return owui_module('open_webui.models.models')


def _model(model_schemas, id: str, owner_id: str):
    """A workspace model carrying a system prompt, as `search_models` returns it."""
    return model_schemas.ModelModel(
        id=id,
        user_id=owner_id,
        base_model_id='gpt-4o',
        name=f'Preset {id}',
        params={'system': SYSTEM_PROMPT, 'temperature': 0.4},
        meta={'description': 'shared preset', 'profile_image_url': AVATAR_URL},
        is_active=True,
        created_at=1,
        updated_at=1,
    )


def _user(id: str, role: str = 'user'):
    return SimpleNamespace(id=id, role=role)


async def _list_models(models_router, caller, models, writable_ids=(), bypass_admin=False):
    """Drive the real `get_models` with only the DB boundary mocked."""
    listing = SimpleNamespace(items=list(models), total=len(models))
    with (
        patch.object(models_router, 'BYPASS_ADMIN_ACCESS_CONTROL', bypass_admin),
        patch.object(models_router.Groups, 'get_groups_by_member_id', AsyncMock(return_value=[])),
        patch.object(models_router.Models, 'search_models', AsyncMock(return_value=listing)),
        patch.object(
            models_router.AccessGrants,
            'get_accessible_resource_ids',
            AsyncMock(return_value=set(writable_ids)),
        ),
    ):
        return await models_router.get_models(user=caller, db=None)


async def _get_model(models_router, caller, model, writable=False, bypass_admin=False):
    """Drive the real `get_model_by_id` for the same caller and model."""

    async def has_access(user_id, resource_type, resource_id, permission, db=None, **kwargs):
        return writable if permission == 'write' else True

    with (
        patch.object(models_router, 'BYPASS_ADMIN_ACCESS_CONTROL', bypass_admin),
        patch.object(models_router.Models, 'get_model_by_id', AsyncMock(return_value=model)),
        patch.object(models_router.AccessGrants, 'has_access', AsyncMock(side_effect=has_access)),
    ):
        return await models_router.get_model_by_id(id=model.id, user=caller, db=None)


def _params(item) -> dict:
    return item.params.model_dump()


# ── Narrow: exactly the leak the fix closed ──────────────────────────────────


@pytest.mark.asyncio
async def test_list_strips_params_for_read_only_caller(models_router, model_schemas):
    shared = _model(model_schemas, 'shared-preset', owner_id='bob')

    response = await _list_models(models_router, _user('alice'), [shared])

    item = response.items[0]
    assert item.write_access is False
    assert _params(item) == {}, (
        'the model list handed a read-only caller the full params of a model they '
        'cannot edit; #27004 is back'
    )
    assert SYSTEM_PROMPT not in response.model_dump_json(), (
        "the owner's system prompt appears somewhere in the list payload for a "
        'caller with read access only (#27004)'
    )


@pytest.mark.asyncio
async def test_list_keeps_params_for_owner(models_router, model_schemas):
    owned = _model(model_schemas, 'my-preset', owner_id='alice')

    response = await _list_models(models_router, _user('alice'), [owned])

    item = response.items[0]
    assert item.write_access is True
    assert _params(item)['system'] == SYSTEM_PROMPT, (
        'the owner lost their own system prompt in the list, so the workspace edit '
        'UI would open on an empty prompt (#27004 over-corrected)'
    )


@pytest.mark.asyncio
async def test_list_keeps_params_for_write_grant_holder(models_router, model_schemas):
    shared = _model(model_schemas, 'shared-preset', owner_id='bob')

    response = await _list_models(
        models_router, _user('alice'), [shared], writable_ids=['shared-preset']
    )

    item = response.items[0]
    assert item.write_access is True
    assert _params(item)['system'] == SYSTEM_PROMPT, (
        'a user with a write grant can edit the model but no longer sees its '
        'system prompt in the list (#27004)'
    )


@pytest.mark.asyncio
async def test_list_keeps_params_for_bypassing_admin(models_router, model_schemas):
    shared = _model(model_schemas, 'shared-preset', owner_id='bob')

    response = await _list_models(
        models_router, _user('admin-1', role='admin'), [shared], bypass_admin=True
    )

    item = response.items[0]
    assert item.write_access is True
    assert _params(item)['system'] == SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_admin_without_bypass_is_treated_as_read_only(models_router, model_schemas):
    """BYPASS_ADMIN_ACCESS_CONTROL off means an admin holds no implicit write access."""
    shared = _model(model_schemas, 'shared-preset', owner_id='bob')

    response = await _list_models(
        models_router, _user('admin-1', role='admin'), [shared], bypass_admin=False
    )

    item = response.items[0]
    assert item.write_access is False
    assert _params(item) == {}, (
        'with the admin bypass disabled an admin is an ordinary read-only caller, '
        'so the list must not hand them a foreign system prompt (#27004)'
    )


# ── Broad: the two routes must agree on what a read-only caller sees ─────────


def _comparable(payload) -> dict:
    """Both payloads minus the one difference the list route intends: it pops
    `profile_image_url` because images are served from a dedicated route."""
    data = payload.model_dump()
    data['meta'].pop('profile_image_url', None)
    return data


@pytest.mark.parametrize(
    'owner_id, writable, bypass_admin, role',
    [
        ('bob', False, False, 'user'),
        ('alice', False, False, 'user'),
        ('bob', True, False, 'user'),
        ('bob', False, True, 'admin'),
    ],
    ids=['read-only', 'owner', 'write-grant', 'bypassing-admin'],
)
@pytest.mark.asyncio
async def test_list_and_per_id_expose_the_same_model(
    models_router, model_schemas, owner_id, writable, bypass_admin, role
):
    """One expectation for both routes, so they cannot drift apart again."""
    caller = _user('alice', role=role)
    model = _model(model_schemas, 'shared-preset', owner_id=owner_id)
    writable_ids = ['shared-preset'] if writable else []

    listed = await _list_models(
        models_router, caller, [model], writable_ids=writable_ids, bypass_admin=bypass_admin
    )
    single = await _get_model(
        models_router, caller, model, writable=writable, bypass_admin=bypass_admin
    )

    assert _comparable(listed.items[0]) == _comparable(single), (
        'GET /models/list and GET /models/model disagree about what this caller may '
        'see, which is how the params leak appeared in the first place (#27004)'
    )


# ── Nearby: adjacent behaviour that must hold on both refs ───────────────────


@pytest.mark.asyncio
async def test_list_drops_profile_image_url_from_meta(models_router, model_schemas):
    owned = _model(model_schemas, 'my-preset', owner_id='alice')

    response = await _list_models(models_router, _user('alice'), [owned])

    assert response.items[0].meta.profile_image_url is None
    assert response.items[0].meta.description == 'shared preset'


@pytest.mark.asyncio
async def test_write_access_is_computed_per_item_in_a_mixed_list(models_router, model_schemas):
    models = [
        _model(model_schemas, 'owned', owner_id='alice'),
        _model(model_schemas, 'granted', owner_id='bob'),
        _model(model_schemas, 'read-only', owner_id='bob'),
    ]

    response = await _list_models(models_router, _user('alice'), models, writable_ids=['granted'])

    assert {item.id: item.write_access for item in response.items} == {
        'owned': True,
        'granted': True,
        'read-only': False,
    }


@pytest.mark.asyncio
async def test_empty_list_returns_cleanly(models_router):
    response = await _list_models(models_router, _user('alice'), [])

    assert response.items == []
    assert response.total == 0
