"""Journey: the model searches and reads the channels an account belongs to.

With channels switched on, a chat is offered four channel tools: `search_channels` matches a
channel's name or description, `search_channel_messages` finds messages and thread replies,
`view_channel_message` reads one message and `view_channel_thread` a whole thread in the order
it was written. Each sees only the channels the account is a member of; another channel's
message is refused by id. With channels off, none of the four is offered.

The scripted model calls each tool and the test reads the result it was sent back.

Discriminates: in a backend copy, `view_channel_message` skipping its membership check turned
the stranger test red; `view_channel_thread` listing replies newest first turned the thread order
test red; `search_channels` ignoring descriptions turned the description test red; and the
channel tools offered without the channels switch turned the switched-off test red.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness.channel_chat import enable_channels
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHANNEL_TOOLS = {
    "search_channels",
    "search_channel_messages",
    "view_channel_message",
    "view_channel_thread",
}


def call(actor, upstream, tool: str, **arguments):
    """The JSON a builtin tool returned when the model called it for `actor`."""
    with actor.client() as client:
        return json.loads(run_tool(client, upstream, tool, arguments))


def post(actor, channel_id: str, content: str, parent_id: str | None = None) -> str:
    with actor.client() as client:
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post",
            json={"content": content, "parent_id": parent_id},
        )
    assert posted.status_code == 200, posted.text
    return posted.json()["id"]


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        enable_channels(client)


@pytest.fixture
def birdwatch(channels_on, make_user):
    """A group channel of two members with a thread, and a stranger; the names are unique."""
    member, other_member, stranger = make_user(), make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    with member.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={
                "name": f"birdwatch-{tag}",
                "description": f"heron counts {tag}",
                "type": "group",
                "user_ids": [other_member.id],
            },
        )
    assert created.status_code == 200, created.text
    channel_id = created.json()["id"]
    parent_id = post(member, channel_id, f"heron at the jetty {tag}")
    first_reply = post(other_member, channel_id, f"two more herons {tag}", parent_id)
    second_reply = post(member, channel_id, "and a grebe", parent_id)
    return {
        "member": member,
        "stranger": stranger,
        "tag": tag,
        "channel_id": channel_id,
        "parent_id": parent_id,
        "reply_ids": [first_reply, second_reply],
    }


def test_a_channel_is_found_by_name_and_by_description(birdwatch, upstream):
    member, tag = birdwatch["member"], birdwatch["tag"]

    by_name = call(member, upstream, "search_channels", query=f"birdwatch-{tag}")
    by_description = call(member, upstream, "search_channels", query=f"counts {tag}")

    assert [channel["id"] for channel in by_name] == [birdwatch["channel_id"]]
    assert by_description == by_name, f"the description did not match: {by_description}"
    assert by_name[0]["type"] == "group"


def test_a_stranger_finds_neither_the_channel_nor_its_messages(birdwatch, upstream):
    stranger, tag = birdwatch["stranger"], birdwatch["tag"]

    assert call(stranger, upstream, "search_channels", query=tag) == []
    assert call(stranger, upstream, "search_channel_messages", query=tag) == []


def test_messages_and_thread_replies_are_found(birdwatch, upstream):
    found = call(birdwatch["member"], upstream, "search_channel_messages", query=birdwatch["tag"])

    by_id = {message["message_id"]: message for message in found}
    assert set(by_id) == {birdwatch["parent_id"], birdwatch["reply_ids"][0]}, found
    assert by_id[birdwatch["parent_id"]]["is_thread_reply"] is False
    reply = by_id[birdwatch["reply_ids"][0]]
    assert (reply["is_thread_reply"], reply["parent_id"]) == (True, birdwatch["parent_id"])
    assert reply["channel_name"] == f"birdwatch-{birdwatch['tag']}"


def test_a_message_is_read_by_a_member_and_refused_to_a_stranger(birdwatch, upstream):
    member, stranger = birdwatch["member"], birdwatch["stranger"]

    viewed = call(member, upstream, "view_channel_message", message_id=birdwatch["parent_id"])
    refused = call(stranger, upstream, "view_channel_message", message_id=birdwatch["parent_id"])

    assert viewed["content"] == f"heron at the jetty {birdwatch['tag']}"
    assert (viewed["reply_count"], viewed["user_name"]) == (2, member.name)
    assert refused == {"error": "Access denied"}, f"a stranger read a channel message: {refused}"
    assert call(member, upstream, "view_channel_message", message_id="nope") == {
        "error": "Message not found"
    }


def test_a_thread_is_read_in_the_order_it_was_written(birdwatch, upstream):
    member, stranger = birdwatch["member"], birdwatch["stranger"]

    thread = call(member, upstream, "view_channel_thread", parent_message_id=birdwatch["parent_id"])
    refused = call(
        stranger, upstream, "view_channel_thread", parent_message_id=birdwatch["parent_id"]
    )

    ids = [message["id"] for message in thread["messages"]]
    assert ids == [birdwatch["parent_id"], *birdwatch["reply_ids"]], thread
    assert thread["message_count"] == 3
    assert [message["is_parent"] for message in thread["messages"]] == [True, False, False]
    assert refused == {"error": "Access denied"}


def test_the_channel_tools_follow_the_channels_switch(make_user, upstream, admin, preserve):
    preserve("admin_config")
    account = make_user()
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_CHANNELS": False}
        ).raise_for_status()
    with account.client() as client:
        switched_off = offered_tools(client, upstream)
    with admin.client() as client:
        enable_channels(client)
    with account.client() as client:
        switched_on = offered_tools(client, upstream)

    assert not CHANNEL_TOOLS & switched_off, f"channels are off, yet {switched_off} was offered"
    assert CHANNEL_TOOLS <= switched_on
