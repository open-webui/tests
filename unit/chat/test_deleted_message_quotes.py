"""Regression: a deleted channel message leaves no readable quote on its replies.

open-webui 0.11.4 PR #30314 / issue #30313: deleting a channel message left the
people looking at the channel pointing at it. Replies kept showing a quote of the
deleted message, and a reply already being composed was still sent addressed to
it. The fix (`b988f06ce`) clears those references in the channel and thread
components when the `message:delete` event arrives.

The fix itself is frontend-only; there is no backend commit in the PR. What the
backend contributes, and what the frontend clearing leans on, is the contract
that once the target row is gone no backend read surface hands out its content:
every read path resolves a dangling `reply_to_id` to `reply_to_message: None`,
so a reload clears the quotes the live view kept. This file pins that contract
against a real scratch database. The narrow tests here pass on both refs by
design: they are the invariant guard (layer 3), and the discriminating
behaviour of #30314 lives in the Svelte components pinned in
`test_stats_window_origin`'s sibling frontend files, not in Python.

Discriminates: this file is an invariant guard; the pinned backend behaviour
(dangling quote resolves to None on every read) passes on both 344ea5306 and
b988f06ce^. A regression here means the reload itself would start showing
quotes of deleted messages again.
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

FUTURE_EPOCH = 4_000_000_000


@pytest.fixture(scope="module")
def owui(owui_module):
    """Backend modules plus a migrated scratch database."""
    owui_module("open_webui.config")  # runs alembic upgrade head against DATA_DIR
    messages = owui_module("open_webui.models.messages")
    return SimpleNamespace(
        db=owui_module("open_webui.internal.db"),
        messages=messages,
        Messages=messages.Messages,
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


async def _add(owui, *rows):
    async with owui.db.get_async_db() as session:
        session.add_all(rows)
        await session.commit()


def _message(owui, channel_id: str, *, reply_to_id: str | None = None, content: str = "text"):
    message_id = _new_id("msg")
    now = FUTURE_EPOCH + int(time.time())
    return owui.messages.Message(
        id=message_id,
        user_id=_new_id("user"),
        channel_id=channel_id,
        reply_to_id=reply_to_id,
        content=content,
        created_at=now,
        updated_at=now,
    )


async def _quoted_pair(owui):
    """A parent message with one reply quoting it, in a fresh channel."""
    channel_id = _new_id("channel")
    parent = _message(owui, channel_id, content="the quoted text")
    reply = _message(owui, channel_id, reply_to_id=parent.id, content="the reply")
    await _add(owui, parent, reply)
    return channel_id, parent, reply


# --- narrow: every read surface drops the quote once the target is gone ----------


@pytest.mark.asyncio(loop_scope="module")
async def test_deleting_the_quoted_message_clears_the_channel_listing_quote(owui):
    channel_id, parent, reply = await _quoted_pair(owui)

    assert await owui.Messages.delete_message_by_id(parent.id)
    listing = await owui.Messages.get_messages_by_channel_id(channel_id)

    assert [message.id for message in listing] == [reply.id]
    assert listing[0].reply_to_message is None, (
        "the channel listing still resolved the reply's quote to the deleted message, so a "
        "reload would keep showing its text and author (#30313)"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_fetching_the_reply_directly_resolves_no_quote(owui):
    _channel_id, parent, reply = await _quoted_pair(owui)
    await owui.Messages.delete_message_by_id(parent.id)

    fetched = await owui.Messages.get_message_by_id(reply.id)

    assert fetched is not None
    assert fetched.reply_to_message is None, (
        "reading the reply by id still carried the deleted message's content as its quote (#30313)"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_thread_listing_resolves_no_quote_for_a_deleted_target(owui):
    channel_id, parent, reply = await _quoted_pair(owui)
    child = _message(owui, channel_id, reply_to_id=parent.id, content="a thread answer")
    thread_root = _message(owui, channel_id, content="thread root")
    child.parent_id = thread_root.id
    await _add(owui, child)
    await owui.Messages.delete_message_by_id(parent.id)

    thread = await owui.Messages.get_thread_replies_by_message_id(thread_root.id)

    quotes = [message.reply_to_message for message in thread]
    assert quotes == [None], (
        "the thread listing still resolved quotes pointing at the deleted message (#30313)"
    )


# --- nearby: the delete itself and the still-live quote path ---------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_a_live_quote_still_resolves(owui):
    """Control: a reply whose target exists keeps its quote; the None above is the
    dangling case, not a blanket clearing."""
    channel_id, parent, reply = await _quoted_pair(owui)

    fetched = await owui.Messages.get_message_by_id(reply.id)

    assert fetched.reply_to_message is not None
    assert fetched.reply_to_message.id == parent.id
    assert fetched.reply_to_message.content == "the quoted text"


@pytest.mark.asyncio(loop_scope="module")
async def test_deleting_a_message_with_no_quotes_deletes_only_the_row(owui):
    channel_id, parent, _reply = await _quoted_pair(owui)
    orphan = _message(owui, channel_id, reply_to_id=_new_id("never-existed"))

    assert await owui.Messages.delete_message_by_id(parent.id)
    fetched = await owui.Messages.get_message_by_id(orphan.id)
    assert fetched is None or fetched.reply_to_message is None
