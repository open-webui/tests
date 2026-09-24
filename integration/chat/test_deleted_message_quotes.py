"""Invariant: once a channel message is deleted, no read of its replies hands out its content.

open-webui issue #30313 (PR #30314, commit b988f06ce): deleting a channel message left its quote
on every reply in the open views. That fix is frontend-only; what it relies on, and what makes a
reload show the replies without the quote, is that every channel read resolves the quote of a
deleted message to `reply_to_message: None`. This pins that contract over the channel API, for
the channel listing, a single message and a thread listing.

Twin of unit/chat/test_deleted_message_quotes.py.

Discriminates: an invariant guard, so it passes on bbfa876af and on b988f06ce^ alike; with
`Messages.delete_message_by_id` leaving the message row in place, every quote still resolves to
the deleted text and the three deletion tests go red.
"""

from __future__ import annotations

import pytest

from harness.channel_quotes import delete_message, enable_channels, group_channel, post_message

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def channel(admin, preserve, make_user):
    """A group channel of two fresh accounts: its id, the author and the reader."""
    preserve("admin_config")
    enable_channels(admin)
    author, reader = make_user(), make_user()
    return group_channel(author, reader), author, reader


def read(actor, path: str):
    with actor.client() as client:
        response = client.get(path)
    response.raise_for_status()
    return response.json()


def test_the_channel_listing_drops_the_quote_of_a_deleted_message(channel):
    channel_id, author, reader = channel
    quoted_id = post_message(author, channel_id, "the quoted text")
    reply_id = post_message(reader, channel_id, "the reply", reply_to_id=quoted_id)

    delete_message(author, channel_id, quoted_id)
    listing = read(reader, f"/api/v1/channels/{channel_id}/messages")

    assert [(message["id"], message["reply_to_message"]) for message in listing] == [
        (reply_id, None)
    ]


def test_a_reply_read_by_id_carries_no_quote_of_a_deleted_message(channel):
    channel_id, author, reader = channel
    quoted_id = post_message(author, channel_id, "the quoted text")
    reply_id = post_message(reader, channel_id, "the reply", reply_to_id=quoted_id)

    delete_message(author, channel_id, quoted_id)
    reply = read(reader, f"/api/v1/channels/{channel_id}/messages/{reply_id}")

    assert reply["content"] == "the reply"
    assert reply["reply_to_message"] is None


def test_the_thread_listing_drops_the_quote_of_a_deleted_message(channel):
    channel_id, author, reader = channel
    root_id = post_message(author, channel_id, "a thread starter")
    quoted_id = post_message(author, channel_id, "the quoted text", parent_id=root_id)
    reply_id = post_message(
        reader, channel_id, "the reply", parent_id=root_id, reply_to_id=quoted_id
    )

    delete_message(author, channel_id, quoted_id)
    thread = read(reader, f"/api/v1/channels/{channel_id}/messages/{root_id}/thread")

    quotes = {message["id"]: message["reply_to_message"] for message in thread}
    assert quoted_id not in quotes
    assert quotes[reply_id] is None


def test_a_quote_of_a_message_that_still_exists_resolves(channel):
    channel_id, author, reader = channel
    quoted_id = post_message(author, channel_id, "the quoted text")
    reply_id = post_message(reader, channel_id, "the reply", reply_to_id=quoted_id)

    quote = read(reader, f"/api/v1/channels/{channel_id}/messages/{reply_id}")["reply_to_message"]

    assert (quote["id"], quote["content"]) == (quoted_id, "the quoted text")
