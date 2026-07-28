"""Regression: channel write access must not double as a moderation capability.

open-webui 0.11.0 fix `c609ec411` (#27197): `update_message_by_id` and
`delete_message_by_id` in `open_webui/routers/channels.py` gated the standard
(non group, non dm) branch on a single OR condition, so holding write access on
the channel satisfied the check on its own. Any member who could post could
therefore edit or delete a message authored by someone else, and because the
model layer never rewrites `message.user_id`, an edited message kept the
original author's attribution. The fix splits the condition into two gates:
channel write access first, then authorship.

Both endpoints are driven for real; only the model layer, the access helper and
the event/socket sinks are mocked at the boundary.

Discriminates: passes on v0.11.0, fails on v0.10.2 (a non-author member with
write access is allowed through instead of getting 403).
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

REQUEST = SimpleNamespace()

MUTATION = {'update': 'update_message_by_id', 'delete': 'delete_message_by_id'}


@pytest.fixture(scope='module')
def channels_router(owui_module):
    return owui_module('open_webui.routers.channels')


class FakeUser:
    def __init__(self, id: str, role: str = 'user'):
        self.id = id
        self.name = id.title()
        self.role = role

    def model_dump(self):
        return {'id': self.id, 'name': self.name, 'role': self.role}


class FakeChannel:
    def __init__(self, type: str = 'channel', id: str = 'channel-1'):
        self.id = id
        self.type = type

    def model_dump(self):
        return {'id': self.id, 'type': self.type}


AUTHOR = FakeUser('alice')
INTRUDER = FakeUser('mallory')
ADMIN = FakeUser('root', role='admin')


def _message(channels_router, user_id: str = AUTHOR.id, channel_id: str = 'channel-1'):
    """Shaped like `Messages.get_message_by_id`, which returns a MessageResponse."""
    return channels_router.MessageResponse(
        id='message-1',
        user_id=user_id,
        channel_id=channel_id,
        content='original text',
        created_at=1,
        updated_at=1,
        latest_reply_at=None,
        reply_count=0,
        reactions=[],
    )


def _message_layer(message):
    return SimpleNamespace(
        get_message_by_id=AsyncMock(return_value=message),
        update_message_by_id=AsyncMock(return_value=True),
        delete_message_by_id=AsyncMock(return_value=True),
        update_is_pinned_by_id=AsyncMock(return_value=True),
    )


@contextmanager
def _boundary(mod, channel, messages, is_member, has_write):
    channels = SimpleNamespace(
        get_channel_by_id=AsyncMock(return_value=channel),
        is_user_channel_member=AsyncMock(return_value=is_member),
    )
    replacements = {
        'Messages': messages,
        'Channels': channels,
        'Users': SimpleNamespace(get_user_by_id=AsyncMock(return_value=AUTHOR)),
        'check_channels_access': AsyncMock(),
        'channel_has_access': AsyncMock(return_value=has_write),
        'publish_event': AsyncMock(),
        'sio': SimpleNamespace(emit=AsyncMock()),
    }
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(mod, name, replacement))
        yield


async def _invoke(mod, action, caller, channel, messages, *, is_member=True, has_write=True):
    message = messages.get_message_by_id.return_value
    with _boundary(mod, channel, messages, is_member, has_write):
        if action == 'update':
            return await mod.update_message_by_id(
                REQUEST,
                channel.id,
                message.id,
                mod.MessageForm(content='rewritten by someone else'),
                caller,
                None,
            )
        return await mod.delete_message_by_id(REQUEST, channel.id, message.id, caller, None)


async def _attempt(mod, action, caller, channel, messages, **kwargs):
    """Returns (status_code or None, whether the message row was mutated)."""
    status_code = None
    try:
        await _invoke(mod, action, caller, channel, messages, **kwargs)
    except HTTPException as e:
        status_code = e.status_code
    return status_code, getattr(messages, MUTATION[action]).called


# --- Narrow: exactly the reported bug ------------------------------------


@pytest.mark.asyncio
async def test_write_access_does_not_allow_editing_another_members_message(channels_router):
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(channels_router, 'update', INTRUDER, FakeChannel(), messages)
    assert outcome == (403, False), (
        'a member holding only write access rewrote another member\'s message, which stays '
        'attributed to the original author (#27197)'
    )


@pytest.mark.asyncio
async def test_write_access_does_not_allow_deleting_another_members_message(channels_router):
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(channels_router, 'delete', INTRUDER, FakeChannel(), messages)
    assert outcome == (403, False), (
        'a member holding only write access deleted another member\'s message (#27197)'
    )


# --- Broad: the invariant, across both endpoints and both branches -------


@pytest.mark.asyncio
@pytest.mark.parametrize('action', sorted(MUTATION))
@pytest.mark.parametrize('channel_type', ['channel', 'group', 'dm'])
async def test_non_author_non_admin_cannot_mutate_a_message(channels_router, action, channel_type):
    """Full channel standing (member and write access) never grants moderation."""
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(channels_router, action, INTRUDER, FakeChannel(channel_type), messages)
    assert outcome == (403, False), (
        f'{action} on a {channel_type} channel let a non-author with full channel standing '
        f'mutate someone else\'s message; author-or-admin is the rule everywhere (#27197)'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('action', sorted(MUTATION))
async def test_author_without_write_access_is_refused(channels_router, action):
    """The other half of the split: authorship must not stand in for write access either."""
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(
        channels_router, action, AUTHOR, FakeChannel(), messages, has_write=False
    )
    assert outcome == (403, False), (
        f'{action} skipped the channel write-access gate for the message author (#27197)'
    )


# --- Nearby: behaviour that must survive the fix -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('action', sorted(MUTATION))
@pytest.mark.parametrize('channel_type', ['channel', 'group', 'dm'])
async def test_author_can_mutate_own_message(channels_router, action, channel_type):
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(channels_router, action, AUTHOR, FakeChannel(channel_type), messages)
    assert outcome == (None, True), f'the author lost the ability to {action} their own message'


@pytest.mark.asyncio
@pytest.mark.parametrize('action', sorted(MUTATION))
@pytest.mark.parametrize('channel_type', ['channel', 'group', 'dm'])
async def test_admin_can_mutate_any_message(channels_router, action, channel_type):
    """Admins hold no write access here, so only the role carries them past both gates."""
    messages = _message_layer(_message(channels_router))
    outcome = await _attempt(
        channels_router, action, ADMIN, FakeChannel(channel_type), messages, has_write=False
    )
    assert outcome == (None, True), f'an admin can no longer {action} another user\'s message'


@pytest.mark.asyncio
@pytest.mark.parametrize('action', sorted(MUTATION))
async def test_missing_message_is_not_found(channels_router, action):
    messages = _message_layer(_message(channels_router))
    messages.get_message_by_id = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as excinfo:
        with _boundary(channels_router, FakeChannel(), messages, True, True):
            if action == 'update':
                await channels_router.update_message_by_id(
                    REQUEST,
                    'channel-1',
                    'gone',
                    channels_router.MessageForm(content='x'),
                    AUTHOR,
                    None,
                )
            else:
                await channels_router.delete_message_by_id(
                    REQUEST, 'channel-1', 'gone', AUTHOR, None
                )
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_pinning_stays_open_to_any_member_with_write_access(channels_router):
    """Pinning is deliberately not an authorship-gated action, so the fix must not reach it."""
    messages = _message_layer(_message(channels_router))
    with _boundary(channels_router, FakeChannel(), messages, True, True):
        await channels_router.pin_channel_message(
            REQUEST,
            'channel-1',
            'message-1',
            channels_router.PinMessageForm(is_pinned=True),
            INTRUDER,
            None,
        )
    assert messages.update_is_pinned_by_id.called, (
        'pinning someone else\'s message is intended behaviour and was over-corrected'
    )
