"""Regression: deleting someone else's chat must not cancel their generation.

open-webui 0.11.0 fix `4f93c3e36` (#27006): `DELETE /api/v1/chats/{id}` called
`stop_item_tasks(redis, id)` as its very first statement, before checking whether
the caller was an admin, held `chat.delete`, or owned the target chat. Any verified
user who knew a chat id (discoverable through a shared chat or a shared folder)
could therefore kill that chat's in-flight streaming response or title/tag
generation; the delete was refused afterwards, but the damage was already done.
The fix authorizes first and only then runs the cancellation and the delete.

Discriminates: passes on v0.11.0, fails on v0.10.2 (`stop_item_tasks` runs for a
non-owner before the 404/401).
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

VICTIM_CHAT_ID = 'victim-chat-id'

# Every recorded step that mutates state or is observable outside the request.
SIDE_EFFECTS = frozenset(
    {'stop_item_tasks', 'delete_orphan_tags', 'delete_as_admin', 'delete_as_owner', 'publish_event'}
)
AUTHORIZATION = frozenset({'check_permission', 'lookup_as_admin', 'lookup_as_owner'})


def _user(id: str, role: str = 'user') -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role, email=f'{id}@example.com', name=id)


def _chat(owner_id: str, tags: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=VICTIM_CHAT_ID, user_id=owner_id, meta={'tags': tags or []})


def _request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))


@contextmanager
def _recorded(
    chats_module,
    *,
    admin_lookup=None,
    owner_lookup=None,
    has_delete_permission=True,
    children=(),
):
    """Patch the router's collaborators and record the order they run in.

    `admin_lookup`/`owner_lookup` are what the two ownership queries return, so a
    non-owner is modelled by `owner_lookup=None` exactly as the DB would.
    """
    steps: list[str] = []

    def record(name, result=None):
        async def _call(*args, **kwargs):
            steps.append(name)
            return result(*args, **kwargs) if callable(result) else result

        return _call

    chats_model = chats_module.Chats
    with (
        patch.object(chats_module, 'stop_item_tasks', record('stop_item_tasks')),
        patch.object(chats_module, 'publish_event', record('publish_event')),
        patch.object(
            chats_module, 'has_permission', record('check_permission', has_delete_permission)
        ),
        patch.object(chats_module.Config, 'get', record('read_permission_config', {})),
        patch.object(chats_model, 'get_chat_by_id', record('lookup_as_admin', admin_lookup)),
        patch.object(
            chats_model, 'get_chat_by_id_and_user_id', record('lookup_as_owner', owner_lookup)
        ),
        patch.object(
            chats_model, 'delete_orphan_tags_for_user', record('delete_orphan_tags', True)
        ),
        patch.object(chats_model, 'delete_chat_by_id', record('delete_as_admin', True)),
        patch.object(chats_model, 'delete_chat_by_id_and_user_id', record('delete_as_owner', True)),
        patch.object(
            chats_model,
            'get_internal_chat_ids_by_parent_id',
            record('lookup_children', list(children)),
        ),
    ):
        yield steps


async def _delete(chats_module, user):
    return await chats_module.delete_chat_by_id(
        request=_request(), id=VICTIM_CHAT_ID, user=user, db=None
    )


@pytest.fixture(scope='module')
def chats_router(owui_module):
    return owui_module('open_webui.routers.chats')


# --- Narrow: the bug itself ---


@pytest.mark.asyncio
async def test_non_owner_delete_does_not_cancel_the_owners_tasks(chats_router):
    """The bug: a stranger's DELETE reached `stop_item_tasks` before the 404."""
    with _recorded(chats_router, owner_lookup=None) as steps:
        with pytest.raises(HTTPException) as excinfo:
            await _delete(chats_router, _user('intruder'))

    assert excinfo.value.status_code == 404
    assert 'stop_item_tasks' not in steps, (
        "a user who does not own the chat cancelled its in-flight generation: "
        f"knowing a chat id is enough to cut off another user's reply (#27006). "
        f"Steps ran: {steps}"
    )


@pytest.mark.asyncio
async def test_caller_without_delete_permission_does_not_cancel_tasks(chats_router):
    """Same hole via the other refusal: `chat.delete` revoked group-wide."""
    with _recorded(
        chats_router, owner_lookup=_chat('intruder'), has_delete_permission=False
    ) as steps:
        with pytest.raises(HTTPException) as excinfo:
            await _delete(chats_router, _user('intruder'))

    assert excinfo.value.status_code == 401
    assert 'stop_item_tasks' not in steps, (
        'a caller whose chat.delete permission was revoked still cancelled the chat tasks '
        '(#27006). '
        f'Steps ran: {steps}'
    )


# --- Broad: the invariant the fix is an instance of ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'role,admin_lookup,owner_lookup',
    [('user', None, None), ('admin', None, None)],
    ids=['non_owner', 'admin_missing_chat'],
)
async def test_refused_delete_has_no_side_effects_at_all(
    chats_router, role, admin_lookup, owner_lookup
):
    """Not just cancellation: no tag cleanup, no delete, no event either."""
    with _recorded(chats_router, admin_lookup=admin_lookup, owner_lookup=owner_lookup) as steps:
        with pytest.raises(HTTPException):
            await _delete(chats_router, _user('caller', role=role))

    assert not SIDE_EFFECTS.intersection(steps), (
        f'a refused delete still mutated state or emitted an event: '
        f'{sorted(SIDE_EFFECTS.intersection(steps))} (#27006)'
    )


@pytest.mark.asyncio
async def test_authorization_completes_before_the_first_side_effect(chats_router):
    """Ordering property, asserted on the path that does reach the side effects."""
    with _recorded(chats_router, owner_lookup=_chat('owner')) as steps:
        await _delete(chats_router, _user('owner'))

    last_authorization = max(index for index, step in enumerate(steps) if step in AUTHORIZATION)
    first_side_effect = min(index for index, step in enumerate(steps) if step in SIDE_EFFECTS)
    assert last_authorization < first_side_effect, (
        f'an ownership check runs after a side effect, so the side effect is reachable '
        f'unauthorized (#27006). Steps ran: {steps}'
    )


@pytest.mark.asyncio
async def test_archive_endpoint_authorizes_before_cancelling_tasks(chats_router):
    """Sibling endpoint with the same shape: archiving also cancels tasks."""
    steps: list[str] = []

    async def refuse_lookup(*args, **kwargs):
        steps.append('lookup_as_owner')
        return None

    async def cancel(*args, **kwargs):
        steps.append('stop_item_tasks')

    with (
        patch.object(chats_router.Chats, 'get_chat_by_id_and_user_id', refuse_lookup),
        patch.object(chats_router, 'stop_item_tasks', cancel),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await chats_router.archive_chat_by_id(
                request=_request(), id=VICTIM_CHAT_ID, user=_user('intruder'), db=None
            )

    assert excinfo.value.status_code == 401
    assert steps == ['lookup_as_owner'], (
        f'POST /chats/{{id}}/archive cancelled a non-owner chat tasks before refusing '
        f'(#27006). Steps ran: {steps}'
    )


@pytest.mark.asyncio
async def test_bulk_delete_is_scoped_to_the_caller(chats_router):
    """`DELETE /chats/` takes no id, so it must only ever target the caller."""
    deleted_for: list[str] = []

    async def delete_chats(user_id, db=None):
        deleted_for.append(user_id)
        return True

    async def allow(*args, **kwargs):
        return True

    with (
        patch.object(chats_router.Chats, 'delete_chats_by_user_id', delete_chats),
        patch.object(chats_router, 'has_permission', allow),
        patch.object(chats_router.Config, 'get', allow),
        patch.object(chats_router, 'publish_event', allow),
    ):
        await chats_router.delete_all_user_chats(request=_request(), user=_user('caller'), db=None)

    assert deleted_for == ['caller'], (
        f'bulk delete targeted a user other than the caller: {deleted_for}'
    )


# --- Nearby: behaviour that is correct on both refs and must stay that way ---


@pytest.mark.asyncio
async def test_owner_delete_still_cancels_their_own_tasks(chats_router):
    with _recorded(chats_router, owner_lookup=_chat('owner', tags=['work'])) as steps:
        result = await _delete(chats_router, _user('owner'))

    assert result is True
    assert 'stop_item_tasks' in steps
    assert 'delete_as_owner' in steps
    assert 'publish_event' in steps


@pytest.mark.asyncio
async def test_admin_delete_of_another_users_chat_succeeds(chats_router):
    with _recorded(chats_router, admin_lookup=_chat('owner')) as steps:
        result = await _delete(chats_router, _user('root', role='admin'))

    assert result is True
    assert 'lookup_as_owner' not in steps, (
        'the admin path must not require the admin to own the chat'
    )
    assert 'stop_item_tasks' in steps
    assert 'delete_as_admin' in steps


@pytest.mark.asyncio
async def test_deleting_a_missing_chat_is_refused_cleanly(chats_router):
    with _recorded(chats_router, owner_lookup=None):
        with pytest.raises(HTTPException) as excinfo:
            await _delete(chats_router, _user('owner'))

    assert excinfo.value.status_code == 404
