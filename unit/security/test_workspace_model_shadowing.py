"""Regression: a non-admin could publish a workspace model that takes over the
identity of a model served by a connected provider.

open-webui 0.11.1 fix `ea55d3879`: the write paths in `routers/models.py`
(`create_new_model`, `import_models`, `update_model_by_id`) let any user holding
the `workspace.models` permission store an entry whose `id` equals a real
provider model id and whose `base_model_id` is empty. A workspace entry with no
base model is not a preset layered over something else, it *is* the model of
that id, so the entry replaced the provider's model in the picker for everyone
on the instance: the shadowing user's system prompt, params and knowledge were
silently applied to every chat anyone started with what looked like the stock
model.

The fix makes a non-admin write always carry a `base_model_id` and rejects an id
that belongs to a non-preset entry of `request.app.state.MODELS` (with the
ollama tag stripped, so `llama3` cannot shadow `llama3:latest`). Create and
update raise 401; import skips the offending entry and leaves it out of the
reported imported ids. Admins keep the unrestricted path.

Discriminates: passes on v0.11.1, fails on v0.11.0 (all three routes happily
persist the shadowing entry).
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression


@pytest.fixture(scope='session')
def models_router(owui_module):
    return owui_module('open_webui.routers.models')


@pytest.fixture(scope='session')
def model_schemas(owui_module):
    return owui_module('open_webui.models.models')


# ── the world the routes see ─────────────────────────────────────────────────

# What the connected providers serve, in the shape `get_all_models` leaves in
# `app.state.MODELS`: real models plus the workspace presets built on them.
PROVIDER_MODELS = {
    'gpt-4o': {'id': 'gpt-4o', 'name': 'GPT-4o', 'owned_by': 'openai'},
    'llama3:latest': {'id': 'llama3:latest', 'name': 'Llama 3', 'owned_by': 'ollama'},
    'team-assistant': {
        'id': 'team-assistant',
        'name': 'Team Assistant',
        'owned_by': 'openai',
        'preset': True,
    },
}


def _user(id: str, role: str = 'user'):
    return SimpleNamespace(id=id, role=role)


def _request():
    state = SimpleNamespace(MODELS=dict(PROVIDER_MODELS))
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _form(model_schemas, id: str, base_model_id: str | None = None, name: str | None = None):
    """A workspace model payload as the create/update routes receive it.

    `base_model_id` is always passed explicitly so it lands in `model_fields_set`,
    which is how the update route tells "left alone" from "cleared".
    """
    return model_schemas.ModelForm(
        id=id,
        base_model_id=base_model_id,
        name=name or f'Preset {id}',
        meta={},
        params={},
    )


def _stored(model_schemas, id: str, owner_id: str, base_model_id: str | None):
    """An existing row, as `Models.get_model_by_id` returns it."""
    return model_schemas.ModelModel(
        id=id,
        user_id=owner_id,
        base_model_id=base_model_id,
        name=f'Preset {id}',
        params={},
        meta={'description': 'workspace model'},
        is_active=True,
        created_at=1,
        updated_at=1,
    )


@contextmanager
def _boundaries(models_router, **overrides):
    """Mock only the boundaries (config, events, model discovery) so the route's
    own authorisation logic runs for real."""
    stubs = {
        'has_permission': AsyncMock(return_value=True),
        'filter_allowed_access_grants': AsyncMock(return_value=[]),
        'publish_event': AsyncMock(),
    }
    stubs.update(overrides)

    with patch.object(models_router.Config, 'get', AsyncMock(return_value={})):
        with patch.multiple(models_router, **stubs):
            yield stubs


# ── drivers: one per write path, each reporting what it persisted ────────────


class Outcome(SimpleNamespace):
    """What a write path did: the returned model, the raised HTTPException (if
    any), and the ids it actually wrote to the database."""

    @property
    def denied(self) -> bool:
        return self.error is not None

    @property
    def status_code(self) -> int | None:
        return self.error.status_code if self.error else None


async def _create(models_router, model_schemas, caller, form, existing=None) -> Outcome:
    insert = AsyncMock(return_value=_stored(model_schemas, form.id, caller.id, form.base_model_id))
    result, error = None, None

    with _boundaries(models_router):
        with (
            patch.object(models_router.Models, 'get_model_by_id', AsyncMock(return_value=existing)),
            patch.object(models_router.Models, 'insert_new_model', insert),
        ):
            try:
                result = await models_router.create_new_model(
                    request=_request(), form_data=form, user=caller, db=None
                )
            except HTTPException as e:
                error = e

    written = [call.args[0].id for call in insert.await_args_list]
    return Outcome(result=result, error=error, written=written, reported=written)


async def _import(models_router, model_schemas, caller, payloads, existing=()) -> Outcome:
    insert = AsyncMock(return_value=None)
    update = AsyncMock(return_value=None)
    publish = AsyncMock()
    result, error = None, None
    form = models_router.ModelsImportForm(models=list(payloads))

    with _boundaries(models_router, publish_event=publish):
        with (
            patch.object(
                models_router.Models, 'get_models_by_ids', AsyncMock(return_value=list(existing))
            ),
            patch.object(models_router.Models, 'insert_new_model', insert),
            patch.object(models_router.Models, 'update_model_by_id', update),
            patch.object(
                models_router.Groups, 'get_groups_by_member_id', AsyncMock(return_value=[])
            ),
            patch.object(
                models_router.AccessGrants,
                'get_accessible_resource_ids',
                AsyncMock(return_value=set()),
            ),
        ):
            try:
                result = await models_router.import_models(
                    request=_request(), user=caller, form_data=form, db=None
                )
            except HTTPException as e:
                error = e

    written = [call.kwargs['form_data'].id for call in insert.await_args_list]
    written += [call.args[0] for call in update.await_args_list]
    reported = list(publish.await_args.kwargs['data']['model_ids']) if publish.await_args else []
    return Outcome(result=result, error=error, written=written, reported=reported)


async def _update(models_router, model_schemas, caller, form, existing, has_access=True) -> Outcome:
    update = AsyncMock(return_value=_stored(model_schemas, form.id, caller.id, form.base_model_id))
    result, error = None, None

    with _boundaries(models_router):
        with (
            patch.object(models_router.Models, 'get_model_by_id', AsyncMock(return_value=existing)),
            patch.object(models_router.Models, 'update_model_by_id', update),
            patch.object(
                models_router.AccessGrants, 'has_access', AsyncMock(return_value=has_access)
            ),
        ):
            try:
                result = await models_router.update_model_by_id(
                    request=_request(), form_data=form, user=caller, db=None
                )
            except HTTPException as e:
                error = e

    written = [call.args[0] for call in update.await_args_list]
    return Outcome(result=result, error=error, written=written, reported=written)


# ── Narrow: the three writes that let a user hijack a provider model ─────────


@pytest.mark.asyncio
async def test_create_refuses_non_admin_id_that_shadows_a_provider_model(
    models_router, model_schemas
):
    """POST /models/create with id `gpt-4o` and no base model: the exact hijack."""
    form = _form(model_schemas, 'gpt-4o', base_model_id=None)

    outcome = await _create(models_router, model_schemas, _user('mallory'), form)

    assert outcome.written == [], (
        'a non-admin stored a workspace model under a provider model id, so everyone '
        'picking `gpt-4o` now gets their system prompt instead (ea55d3879 is back)'
    )
    assert outcome.status_code == 401


@pytest.mark.asyncio
async def test_create_refuses_the_shadowing_id_even_with_a_base_model_set(
    models_router, model_schemas
):
    """Carrying a base model does not license taking the real model's id."""
    form = _form(model_schemas, 'gpt-4o', base_model_id='gpt-4o-2024-08-06')

    outcome = await _create(models_router, model_schemas, _user('mallory'), form)

    assert outcome.written == []
    assert outcome.status_code == 401


@pytest.mark.asyncio
async def test_create_refuses_an_ollama_id_with_its_tag_stripped(models_router, model_schemas):
    """`llama3` resolves to `llama3:latest`, so the untagged id shadows it too."""
    form = _form(model_schemas, 'llama3', base_model_id='llama3:latest')

    outcome = await _create(models_router, model_schemas, _user('mallory'), form)

    assert outcome.written == [], (
        'a non-admin claimed the untagged ollama id `llama3`, which resolves to the '
        "provider's `llama3:latest` (ea55d3879 is back)"
    )
    assert outcome.status_code == 401


@pytest.mark.asyncio
async def test_import_skips_a_non_admin_entry_that_shadows_a_provider_model(
    models_router, model_schemas
):
    """POST /models/import is the same hijack with a JSON file instead of the form."""
    outcome = await _import(
        models_router,
        model_schemas,
        _user('mallory'),
        [
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'base_model_id': 'gpt-4o-2024-08-06'},
            {'id': 'mallory-helper', 'name': 'Helper', 'base_model_id': 'gpt-4o'},
        ],
    )

    assert 'gpt-4o' not in outcome.written, (
        'importing a model file let a non-admin store an entry under a provider model '
        'id, hijacking it instance-wide (ea55d3879 is back)'
    )
    assert 'gpt-4o' not in outcome.reported
    assert outcome.written == [
        'mallory-helper'
    ], 'the harmless entry alongside it in the same file must still import'


@pytest.mark.asyncio
async def test_import_skips_a_non_admin_entry_with_no_base_model(models_router, model_schemas):
    """An entry with no base model *is* a model of that id, so a user cannot import one."""
    outcome = await _import(
        models_router,
        model_schemas,
        _user('mallory'),
        [{'id': 'brand-new-model', 'name': 'Brand New'}],
    )

    assert outcome.written == [], (
        'a non-admin imported a standalone model entry (no base model), which is how an '
        'id gets claimed ahead of the provider that serves it (ea55d3879 is back)'
    )
    assert outcome.reported == []


@pytest.mark.asyncio
async def test_update_refuses_a_non_admin_clearing_the_base_model(models_router, model_schemas):
    """Editing a preset into a standalone entry is the third route to the same hijack."""
    existing = _stored(model_schemas, 'gpt-4o', owner_id='mallory', base_model_id='gpt-4o-mini')
    form = _form(model_schemas, 'gpt-4o', base_model_id=None)

    outcome = await _update(models_router, model_schemas, _user('mallory'), form, existing)

    assert outcome.written == [], (
        'a non-admin edited their preset into a base-model-less entry, which takes the '
        'id over for everyone on the instance (ea55d3879 is back)'
    )
    assert outcome.status_code == 401


@pytest.mark.asyncio
async def test_import_skips_clearing_the_base_model_of_an_existing_entry(
    models_router, model_schemas
):
    """The import route's update branch must refuse that same edit."""
    existing = _stored(model_schemas, 'gpt-4o', owner_id='mallory', base_model_id='gpt-4o-mini')

    outcome = await _import(
        models_router,
        model_schemas,
        _user('mallory'),
        [{'id': 'gpt-4o', 'name': 'GPT-4o', 'base_model_id': None}],
        existing=[existing],
    )

    assert outcome.written == [], (
        're-importing an owned preset with an empty base model cleared it, turning the '
        'entry into a shadow of the provider model (ea55d3879 is back)'
    )


# ── Broad: one invariant every non-admin write path must hold ────────────────


SHADOWING_WRITES = [
    ('create', 'gpt-4o', None),
    ('create', 'gpt-4o', 'gpt-4o-2024-08-06'),
    ('create', 'llama3', 'llama3:latest'),
    ('create', 'llama3:latest', 'gpt-4o'),
    ('create', 'brand-new', None),
    ('import', 'gpt-4o', 'gpt-4o-2024-08-06'),
    ('import', 'llama3', 'gpt-4o'),
    ('import', 'llama3:latest', 'gpt-4o'),
    ('import', 'brand-new', None),
]


@pytest.mark.parametrize('route, model_id, base_model_id', SHADOWING_WRITES)
@pytest.mark.asyncio
async def test_no_non_admin_write_path_persists_a_shadowing_model(
    models_router, model_schemas, route, model_id, base_model_id
):
    """Create and import must agree: a non-admin never stores an entry claiming a
    provider model id, and never stores one without a base model."""
    caller = _user('mallory')

    if route == 'create':
        form = _form(model_schemas, model_id, base_model_id=base_model_id)
        outcome = await _create(models_router, model_schemas, caller, form)
    else:
        outcome = await _import(
            models_router,
            model_schemas,
            caller,
            [{'id': model_id, 'name': model_id, 'base_model_id': base_model_id}],
        )

    assert (
        outcome.written == []
    ), f'{route} persisted the shadowing entry {model_id!r} (ea55d3879 is back)'
    assert outcome.reported == [], f'{route} reported {model_id!r} as written although it was not'


@pytest.mark.parametrize('route', ['create', 'import', 'update'])
@pytest.mark.asyncio
async def test_admin_writes_are_never_restricted_by_the_shadowing_guard(
    models_router, model_schemas, route
):
    """The guard is about privilege, so an admin keeps every one of these writes."""
    admin = _user('root', role='admin')

    if route == 'create':
        form = _form(model_schemas, 'gpt-4o', base_model_id=None)
        outcome = await _create(models_router, model_schemas, admin, form)
    elif route == 'import':
        outcome = await _import(
            models_router, model_schemas, admin, [{'id': 'gpt-4o', 'name': 'GPT-4o'}]
        )
    else:
        existing = _stored(model_schemas, 'gpt-4o', owner_id='someone', base_model_id='gpt-4o-mini')
        form = _form(model_schemas, 'gpt-4o', base_model_id=None)
        # No grant and not the owner, so the admin role is the only thing admitting the write.
        outcome = await _update(
            models_router, model_schemas, admin, form, existing, has_access=False
        )

    assert not outcome.denied, f'the shadowing guard fired on an admin {route} (ea55d3879)'
    assert outcome.written == ['gpt-4o']


# ── Nearby: the ordinary workspace-model flows must keep working ─────────────


@pytest.mark.asyncio
async def test_non_admin_can_still_create_a_preset_on_a_free_id(models_router, model_schemas):
    form = _form(model_schemas, 'mallory-helper', base_model_id='gpt-4o')

    outcome = await _create(models_router, model_schemas, _user('mallory'), form)

    assert outcome.written == ['mallory-helper'], (
        'the shadowing guard rejected an ordinary preset on an unused id, so users can '
        'no longer create workspace models at all (ea55d3879 over-corrected)'
    )
    assert outcome.result.id == 'mallory-helper'


@pytest.mark.asyncio
async def test_non_admin_create_on_a_taken_workspace_id_is_still_the_db_check(
    models_router, model_schemas
):
    """`preset` entries in app.state.MODELS are workspace models, not provider ones, so
    the collision for those stays the pre-existing row lookup."""
    form = _form(model_schemas, 'team-assistant', base_model_id='gpt-4o')
    existing = _stored(model_schemas, 'team-assistant', owner_id='bob', base_model_id='gpt-4o')

    outcome = await _create(models_router, model_schemas, _user('mallory'), form, existing=existing)

    assert outcome.written == []
    assert outcome.status_code == 401


@pytest.mark.asyncio
async def test_non_admin_can_still_edit_a_preset_that_keeps_its_base_model(
    models_router, model_schemas
):
    existing = _stored(model_schemas, 'mallory-helper', owner_id='mallory', base_model_id='gpt-4o')
    form = _form(model_schemas, 'mallory-helper', base_model_id='gpt-4o', name='Renamed')

    outcome = await _update(models_router, model_schemas, _user('mallory'), form, existing)

    assert outcome.written == ['mallory-helper']
    assert outcome.result is not None


@pytest.mark.asyncio
async def test_non_admin_edit_of_an_entry_that_never_had_a_base_model_still_works(
    models_router, model_schemas
):
    """An admin-made standalone entry the user has write access to: the guard fires on
    *clearing* a base model, and there is none to clear."""
    existing = _stored(model_schemas, 'house-model', owner_id='root', base_model_id=None)
    form = _form(model_schemas, 'house-model', base_model_id=None, name='Renamed')

    outcome = await _update(models_router, model_schemas, _user('mallory'), form, existing)

    assert outcome.written == ['house-model']
    assert outcome.result is not None


@pytest.mark.asyncio
async def test_import_still_updates_a_preset_the_user_owns(models_router, model_schemas):
    existing = _stored(model_schemas, 'mallory-helper', owner_id='mallory', base_model_id='gpt-4o')

    outcome = await _import(
        models_router,
        model_schemas,
        _user('mallory'),
        [{'id': 'mallory-helper', 'name': 'Renamed', 'base_model_id': 'gpt-4o'}],
        existing=[existing],
    )

    assert outcome.written == ['mallory-helper']
    assert outcome.reported == ['mallory-helper']
