"""Regression: marking a chat as read cancelled every user's timers on that chat.

open-webui 0.11.0 fix `e140d8f3c` (#27472): `cancel_timers_for_chat` selected timer
rows on the internal marker, the type, the parent chat id and the status, but never
on the owner, so it matched rows belonging to any user. The `events:chat`
`last_read_at` socket handler compounded it by discarding the boolean returned by the
ownership-checked `update_chat_last_read_at_by_id` and cancelling regardless, so a
user who knew someone else's chat id could silently cancel that owner's scheduled
prompt, and in ordinary use two people holding timers on one shared chat cancelled
each other's. The fix makes `user_id` a required parameter, filters on
`Chat.user_id`, and returns early when the caller does not own the chat.

Tests run the real query against a real SQLite session, so the assertions are on the
rows the production SQL actually touched.

0.11.1 `16c2a9eda` (#27663) rewrote the query to key off a new `Chat.timer_at` column
instead of the internal/type meta keys, keeping the `Chat.user_id` filter. Seeding sets
that column when the model has it, so the tests run against both shapes.

0.11.2 `d7674c517` (#28835) routed every socket handler through `get_socket_session_user`,
which reads only Socket.IO's local session store, so the socket tests stub that store.

Discriminates: passes on v0.11.0 through v0.11.3, fails on v0.10.2 (`open_webui.utils.timers`
does not exist on v0.10.2, so the timer feature and its fix both landed inside the 0.11.0
cycle and these tests skip there; they fail behaviourally on `e140d8f3c^`, the last
commit carrying the bug, where the unscoped query cancels the other user's timers).
"""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.regression

CANCEL_TARGET = 'cancel_timers_for_chat'


@pytest.fixture(scope='module')
def timers(owui_module):
    # Importing config runs the alembic migrations, so the `chat` table exists.
    owui_module('open_webui.config')
    return owui_module('open_webui.utils.timers')


@pytest.fixture
def ids():
    """Unique id prefix so tests sharing the scratch database cannot collide."""
    return uuid4().hex[:12]


def _chat(timers, chat_id, owner_id):
    now = int(time.time())
    return timers.Chat(
        id=chat_id,
        user_id=owner_id,
        title=f'Chat: {chat_id}',
        chat={},
        meta={},
        created_at=now,
        updated_at=now,
    )


def _timer(timers, timer_id, owner_id, parent_chat_id, cancel_on=('chat.read',), status='pending'):
    now = int(time.time())
    due_at = time.time_ns()
    row = timers.Chat(
        id=timer_id,
        user_id=owner_id,
        title=f'Timer: {timer_id}',
        chat={},
        meta={
            'internal': True,
            'type': 'timer',
            'parent_chat_id': parent_chat_id,
            'status': status,
            'cancel_on': list(cancel_on),
            'timer_at': due_at,
        },
        created_at=now,
        updated_at=now,
    )
    # 0.11.1 moved the due time onto a real column the queries key off; claiming clears it.
    if 'timer_at' in timers.Chat.__table__.columns:
        row.timer_at = due_at if status == 'pending' else None
    return row


async def _seed(timers, rows):
    async with timers.get_async_db() as db:
        for row in rows:
            db.add(row)
        await db.commit()


async def _statuses(timers, timer_ids):
    async with timers.get_async_db() as db:
        return {
            timer_id: (await db.get(timers.Chat, timer_id)).meta.get('status')
            for timer_id in timer_ids
        }


async def _cancelled(timers, timer_ids):
    statuses = await _statuses(timers, timer_ids)
    return {timer_id for timer_id, status in statuses.items() if status == 'cancelled'}


def _socket_session(sessions):
    """0.11.2 `d7674c517` (#28835) made the socket handlers read the acting user only from
    Socket.IO's own local session store, so stub that store rather than `SESSION_POOL`."""

    async def get_session(sid, namespace=None):
        if sid not in sessions:
            raise KeyError(sid)
        return {'user': sessions[sid]}

    return get_session


async def _read_chat_as(timers, parent_chat_id, event, user_id):
    """Tolerate the pre-fix two-argument signature so the assertion lands on which rows
    were cancelled rather than on a TypeError."""
    if 'user_id' in inspect.signature(timers.cancel_timers_for_chat).parameters:
        await timers.cancel_timers_for_chat(parent_chat_id, event, user_id)
    else:
        await timers.cancel_timers_for_chat(parent_chat_id, event)


# --- Narrow: the bug itself ---


@pytest.mark.asyncio
async def test_marking_a_chat_read_cancels_only_the_callers_timers(timers, ids):
    """Two users hold a pending timer on one shared chat; only the reader's may go."""
    parent_chat_id = f'{ids}-shared-chat'
    alice_timer = f'{ids}-alice'
    bob_timer = f'{ids}-bob'
    await _seed(
        timers,
        [
            _timer(timers, alice_timer, 'alice', parent_chat_id),
            _timer(timers, bob_timer, 'bob', parent_chat_id),
        ],
    )

    await _read_chat_as(timers, parent_chat_id, 'chat.read', 'alice')

    assert await _cancelled(timers, [alice_timer, bob_timer]) == {alice_timer}, (
        "alice reading a chat cancelled bob's pending timer on the same chat: bob's scheduled "
        'prompt silently never fires and he is never told (#27472)'
    )


@pytest.mark.asyncio
async def test_socket_last_read_at_from_a_non_owner_cancels_nothing(timers, owui_module, ids):
    """The attack path: a stranger marks a chat read over their own socket session."""
    socket_main = owui_module('open_webui.socket.main')

    parent_chat_id = f'{ids}-victim-chat'
    await _seed(
        timers,
        [
            _chat(timers, parent_chat_id, 'victim'),
            _timer(timers, f'{ids}-victim-timer', 'victim', parent_chat_id),
        ],
    )

    async def emit(*args, **kwargs):
        return None

    cancel_calls = []

    async def record_cancel(*args, **kwargs):
        cancel_calls.append((args, kwargs))

    # The handler imports the helper at call time, so patch it on its own module.
    with (
        patch.object(socket_main.sio, 'emit', emit),
        patch.object(timers, CANCEL_TARGET, record_cancel),
        patch.object(
            socket_main.sio, 'get_session', _socket_session({'intruder-sid': {'id': 'intruder'}})
        ),
    ):
        await socket_main.chat_events(
            'intruder-sid', {'chat_id': parent_chat_id, 'data': {'type': 'last_read_at'}}
        )

    assert cancel_calls == [], (
        'the socket handler ran the timer cancellation for a user who does not own the chat: '
        'only the ownership filter inside the query then stands between a stranger and the '
        'owner pending timers (#27472)'
    )


# --- Broad: every caller-reachable bulk cancel must carry the user scope ---


@pytest.mark.asyncio
@pytest.mark.parametrize('event', ['chat.read', 'chat.user_message'])
async def test_both_cancel_events_are_owner_scoped(timers, ids, event):
    """`chat.user_message` is the other trigger and takes the same query."""
    parent_chat_id = f'{ids}-shared-chat'
    caller_timer = f'{ids}-caller'
    stranger_timer = f'{ids}-stranger'
    await _seed(
        timers,
        [
            _timer(timers, caller_timer, 'caller', parent_chat_id, cancel_on=(event,)),
            _timer(timers, stranger_timer, 'stranger', parent_chat_id, cancel_on=(event,)),
        ],
    )

    await _read_chat_as(timers, parent_chat_id, event, 'caller')

    assert await _cancelled(timers, [caller_timer, stranger_timer]) == {caller_timer}, (
        f'the {event} trigger cancelled a timer belonging to another user (#27472)'
    )


def test_every_call_site_passes_the_acting_user(timers):
    """Guards the next unscoped sweep: both callers must name the acting user."""
    backend_root = Path(timers.__file__).resolve().parent.parent
    call_sites = 0
    unscoped = []
    for relative_path in ('main.py', 'socket/main.py'):
        path = backend_root / relative_path
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, 'id', None)
            )
            if called != CANCEL_TARGET:
                continue
            call_sites += 1
            passes_user = len(node.args) >= 3 or any(
                keyword.arg == 'user_id' for keyword in node.keywords
            )
            if not passes_user:
                unscoped.append(f'{relative_path}:{node.lineno}')

    assert not unscoped, (
        f'{CANCEL_TARGET} is called without the acting user at {unscoped}: that call cancels the '
        'pending timers of every user on the chat, not just the caller (#27472)'
    )
    assert call_sites == 2, (
        f'expected the two known {CANCEL_TARGET} call sites, found {call_sites}: a caller moved or '
        'was added elsewhere and is no longer covered by this check (#27472)'
    )


# --- Nearby: adjacent behaviour that must keep working ---


@pytest.mark.asyncio
async def test_the_caller_own_timer_is_still_cancelled(timers, ids):
    parent_chat_id = f'{ids}-own-chat'
    own_timer = f'{ids}-own'
    await _seed(timers, [_timer(timers, own_timer, 'alice', parent_chat_id)])

    await _read_chat_as(timers, parent_chat_id, 'chat.read', 'alice')

    async with timers.get_async_db() as db:
        meta = (await db.get(timers.Chat, own_timer)).meta
    assert meta['status'] == 'cancelled', 'scoping the clear to the caller broke the feature itself'
    assert meta['timer_cancelled_by'] == 'chat.read'
    assert meta['timer_cancelled_at'] > 0


@pytest.mark.asyncio
async def test_reading_a_chat_with_no_timers_is_a_no_op(timers, ids):
    await _read_chat_as(timers, f'{ids}-empty-chat', 'chat.read', 'alice')


@pytest.mark.asyncio
async def test_the_callers_timers_on_other_chats_are_untouched(timers, ids):
    read_chat_id = f'{ids}-read-chat'
    other_chat_id = f'{ids}-other-chat'
    read_timer = f'{ids}-read'
    other_timer = f'{ids}-other'
    await _seed(
        timers,
        [
            _timer(timers, read_timer, 'alice', read_chat_id),
            _timer(timers, other_timer, 'alice', other_chat_id),
        ],
    )

    await _read_chat_as(timers, read_chat_id, 'chat.read', 'alice')

    assert await _cancelled(timers, [read_timer, other_timer]) == {read_timer}, (
        'reading one chat cancelled the same user timers on an unrelated chat'
    )


@pytest.mark.asyncio
async def test_a_timer_not_subscribed_to_the_event_survives(timers, ids):
    parent_chat_id = f'{ids}-mixed-chat'
    read_timer = f'{ids}-on-read'
    message_timer = f'{ids}-on-message'
    await _seed(
        timers,
        [
            _timer(timers, read_timer, 'alice', parent_chat_id, cancel_on=('chat.read',)),
            _timer(
                timers, message_timer, 'alice', parent_chat_id, cancel_on=('chat.user_message',)
            ),
        ],
    )

    await _read_chat_as(timers, parent_chat_id, 'chat.read', 'alice')

    assert await _cancelled(timers, [read_timer, message_timer]) == {read_timer}, (
        'a timer that did not opt into chat.read was cancelled by a read'
    )


@pytest.mark.asyncio
async def test_a_running_timer_is_not_cancelled(timers, ids):
    parent_chat_id = f'{ids}-running-chat'
    running_timer = f'{ids}-running'
    await _seed(timers, [_timer(timers, running_timer, 'alice', parent_chat_id, status='running')])

    await _read_chat_as(timers, parent_chat_id, 'chat.read', 'alice')

    assert (await _statuses(timers, [running_timer]))[running_timer] == 'running', (
        'a timer already claimed for execution was cancelled out from under the runner'
    )


@pytest.mark.asyncio
async def test_socket_last_read_at_from_the_owner_still_cancels_their_timer(
    timers, owui_module, ids
):
    """Positive control for the socket path: the early return did not break reading."""
    socket_main = owui_module('open_webui.socket.main')

    parent_chat_id = f'{ids}-owned-chat'
    owner_timer = f'{ids}-owner-timer'
    await _seed(
        timers,
        [
            _chat(timers, parent_chat_id, 'owner'),
            _timer(timers, owner_timer, 'owner', parent_chat_id),
        ],
    )

    async def emit(*args, **kwargs):
        return None

    async def unread_counts(user_id):
        return {}

    with (
        patch.object(socket_main.sio, 'emit', emit),
        patch.object(socket_main, 'get_folder_unread_counts', unread_counts),
        patch.object(
            socket_main.sio, 'get_session', _socket_session({'owner-sid': {'id': 'owner'}})
        ),
    ):
        await socket_main.chat_events(
            'owner-sid', {'chat_id': parent_chat_id, 'data': {'type': 'last_read_at'}}
        )

    assert await _cancelled(timers, [owner_timer]) == {owner_timer}, (
        'the owner marking their own chat as read no longer cancels their chat.read timer'
    )
